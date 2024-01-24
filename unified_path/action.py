import torch
import einops
import abc
from torch.distributions import MultivariateNormal
from .lattice import plaquette


class Action(torch.nn.Module, abc.ABC):
    @abc.abstractmethod
    def evaluate(self, field) -> torch.Tensor:
        pass

    def __call__(self, phi):
        return self.evaluate(phi)


class MGM(Action):
    """
    The MGM action is defined as the log of the sum of two Gaussian distributions
    for each dimension `i` in the input `x`.

    The equation for the action is:
        L = log(Σ(N(x_i; -cluster_loc, cluster_scale) + N(x_i; cluster_loc, cluster_scale)))
    where N(x; μ, σ) is the normal distribution with mean μ and standard deviation σ,
    x_i is the ith element of the input, and the sum is over the dimension `d` of the input.

    """

    def __init__(
        self, device="cpu", clusters=2, dim=2, cluster_loc=1.0, cluster_scale=0.5
    ):
        """
        Parameters:
            device (str): The device to run the calculations on, default is "cpu".
            clusters (int): The number of clusters, default is 2.
            dim (int): The dimensionality of the input, default is 2.
            cluster_loc (float): The location (mean) of the clusters, default is 1.0.
            cluster_scale (float): The scale (standard deviation) of the clusters, default is 0.5.
        """
        Action.__init__(self)
        torch.nn.Module.__init__(self)

        assert dim >= 2, "Dimensionality has to be >=2"
        self.dim = dim
        self.device = device
        # Outer product of cluster locations
        locs = torch.meshgrid(
            dim * [torch.linspace(-cluster_loc, cluster_loc, clusters)], indexing="ij"
        )
        loc = torch.stack([tmp.ravel() for tmp in locs]).T

        self.num_clusters = loc.shape[0]

        cov = (
            einops.repeat(torch.eye(dim), "i j -> c i j", c=self.num_clusters)
            * cluster_scale
        )
        self.loc = loc
        self.scale = cluster_scale
        self.cov = cov
        self.weights = torch.ones(self.num_clusters)  # [c]
        self.weights /= self.weights.sum()  # normalize

        self.dist = MultivariateNormal(loc=self.loc, covariance_matrix=self.cov)

        self.dist.loc = self.dist.loc.to(device)
        self.dist._unbroadcasted_scale_tril = self.dist._unbroadcasted_scale_tril.to(
            device
        )
        self.weights = self.weights.to(device)

    def evaluate(self, field):
        return -self.log_prob(field)

    def log_prob(self, data):
        """LogSumExp trick used to stabilize for weird values we sum over the clusters and divide
        by the number.

        :param data: [BS, N]
        p(x)    = log sum_c w_c * p(x| μ_c, σ_c)
                = log sum_c exp log[ w_c * p(x| μ_c, σ_c) ]
                = log sum_c exp ( log w_c + log p(x|μ_c, σ_c) )
                = logsumexp( log w_c + log p(x|μ_c, σ_c), c)
        """

        data = data.unsqueeze(1)  # [BS, Nd] -> [BS, c=1, Nd]
        log_probs_cluster = torch.log(self.weights).unsqueeze(0)  # [c]
        log_probs_data = self.dist.log_prob(data)  # [b, c]
        log_probs = log_probs_cluster + log_probs_data
        log_probs = torch.logsumexp(log_probs, dim=-1)  # [b, c] -> [b]
        return log_probs

    def sample(self, num_samples=(1,)):
        chosen_cluster = torch.randint(
            low=0, high=self.num_clusters, size=num_samples, device=self.device
        )

        samples = (
            self.dist.sample(num_samples).squeeze(1).to(self.device)
        )  # [BS, 1, c=num_gaussians, nd] -> [BS, c=num_gaussians, nd]
        chosen_cluster = (
            chosen_cluster.unsqueeze(-1).unsqueeze(-1).expand(*num_samples, 1, self.dim)
        )

        if self.num_clusters > 1:
            samples = torch.gather(input=samples, dim=1, index=chosen_cluster).squeeze(
                1
            )

        return samples


class U1GaugeAction(Action):
    """
    U1 Wilson Action
    Original code from https://arxiv.org/abs/2101.08176"""

    def __init__(self, beta):
        """
        Parameters:
            beta (float): Inverse Gauge Coupling parameter
        """
        self.beta = beta

    def actions_per_site(self, theta):
        Nd = theta.shape[1]
        action = 0
        for mu in range(Nd):
            for nu in range(mu + 1, Nd):
                action = action + torch.cos(plaquette(theta, mu, nu))
        return action

    def evaluate(self, field):
        Nd = field.shape[1]
        return -self.beta * torch.sum(
            self.actions_per_site(field), dim=tuple(range(1, Nd + 1))
        )
