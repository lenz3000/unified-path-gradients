from argparse import Namespace
from typing import Union, Tuple, List, Iterable
import abc
import torch

from unified_path.action import Action


class Loss(torch.nn.Module, abc.ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        action: Action,
        lat_shape: List[int],
        batch_size: int,
    ):
        super().__init__()
        if not hasattr(model, "sample_with_logprob"):
            raise TypeError("Model does not support logprob sampling")
        self.sampler = model
        self.action = action
        self.lat_shape = lat_shape
        self.batch_size = batch_size

    @abc.abstractmethod
    def forward(
        self, samples: Union[torch.Tensor, None] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            samples: Samples from the sampling distribution (optional), useful cross-checking losses
        """
        pass

    @abc.abstractmethod
    def __str__(self) -> str:
        pass


class RepQP(Loss):
    r"""Calculates the reverse KL(q|p) loss.
    Differentiation results in reparametrized gradient estimator.
    """

    def forward(self, samples=None):
        if samples is None:
            samples, log_probs = self.sampler.sample_with_logprob(self.batch_size)
        else:
            samples, log_probs = self.sampler.forward(samples)

        actions = self.action(samples.reshape(self.batch_size, *self.lat_shape))
        neg_lwt = actions + log_probs

        loss_mean = neg_lwt.mean()

        return loss_mean, -neg_lwt, actions

    def __str__(self):
        return "Reparam KL(q|p)"


class DropInPathQP(Loss):
    """
    Calculates the reverse KL(q|p) loss.
    Differentiation results in path gradients for the reverse KL, a.k.a STL
    Simple drop-in path gradient estimator for KL(q||p) where q is the variational distribution
    Algorithm 2 in Paper
    """

    def forward(self, samples=None):
        with torch.no_grad():
            if samples is None:
                samples = self.sampler.sample_base(self.batch_size)
            x_prime, _ = self.sampler.forward(samples)

        x_prime = x_prime.detach().clone().requires_grad_()
        log_probs = self.sampler.log_prob(x_prime)

        actions = self.action(x_prime)
        unnorm_target = -1.0 * actions
        log_w_tilde = unnorm_target - log_probs

        grad = torch.autograd.grad(outputs=-log_w_tilde.mean(), inputs=x_prime)[0]

        x, _ = self.sampler.g(samples)
        grad_term = (x * grad).sum()
        loss_mean = -log_w_tilde.detach().mean() + grad_term - grad_term.detach()

        return loss_mean, log_w_tilde, actions

    def __str__(self):
        return "STL KL(q|p)"


class MaximumLikelihood(torch.nn.Module):
    r"""Calculates forward KL(p|q) loss
    Differentiation results in maximum likelihood gradient estimators
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config_sampler: Iterable,
        action: Action,
        lat_shape: List[int],
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.model = model
        self.action = action
        self.config_sampler = config_sampler
        self.lat_shape = lat_shape
        self.device = device

    def forward(self, samples=None):
        if samples is None:
            samples = next(self.config_sampler).to(self.device)
        # The action only appears here for comparing the loss
        actions = self.action(samples)
        loss = -self.model.log_prob(samples)
        loss_mean = loss.mean(0)

        return (loss - actions.detach()).mean(), loss_mean, actions

    def __str__(self) -> str:
        return "MaximumLikelihood"


class FastDropInPathPQ(MaximumLikelihood):
    r"""Calculates forward KL(p|q) loss
    Differentiation results in path gradients for the forward KL, a.k.a. GDReG estimator
    Algorithm 3 in Paper
    """

    def forward(self, samples=None):
        if samples is None:
            samples = next(self.config_sampler).to(self.device)
        x1 = samples.detach().clone().requires_grad_()

        z1, logq = self.model.f(x1)
        actions = self.action(x1)
        nlwt = logq + actions
        grad = torch.autograd.grad(nlwt.mean(), x1)[0]

        z1 = z1.detach().clone().requires_grad_()

        x_, _ = self.model(z1)
        gradterm = (x_ * grad.detach()).reshape((samples.shape[0], -1)).sum()
        return (
            (-logq.detach() - actions.detach()).mean() + gradterm - gradterm.detach(),
            -logq,
            actions,
        )

    def __str__(self) -> str:
        return "GDReG KL(P|Q)"


class FastPath(Loss):
    """
    Efficient implementation of the path gradient estimator for the forward and reverse KL
    Implements Algorithm 1 in paper.
    If reverse is True, then the forward call computes the reverse KL, and differentiation gives back Path Gradients
    else, the forward KL and its path gradients are computed

    At the core of the algorithm is the recursive operation that computes the gradient of the path along with the sampling.
    Since the former is passed along with the sampling, we call it piggy backing
    """

    def __init__(
        self,
        model: torch.nn.Module,
        action: Action,
        lat_shape: List[int],
        batch_size: int,
        reverse=True,
        config_sampler: Union[None, Iterable] = None,
        device=torch.device("cpu"),
    ):
        super().__init__(model, action, lat_shape, batch_size)
        self.device = device
        if not hasattr(self.sampler, "piggy_back_forward"):
            raise TypeError("Model not a path gradient flow")

        self.reverse_kl = reverse
        if self.reverse_kl:
            self.alternative = DropInPathQP(model, action, lat_shape, batch_size)
        else:
            # If the forward KL is used, we need samples from the target distribution
            self.config_sampler = config_sampler
            assert config_sampler is not None, "Need config_sampler for PQ"
            self.alternative = FastDropInPathPQ(
                model, config_sampler, action, lat_shape, device=device
            )

    def forward(self, samples=None):
        # When evaluating, we simply call the faster alternative loss function
        if not self.training or not torch.is_grad_enabled():
            return self.alternative()

        # If samples are not given, sample from given initial distribution
        if samples is None:
            if self.reverse_kl:
                # Either from the base density
                samples = self.sampler.sample_base(self.batch_size)
            else:
                # Or from the target
                samples = next(self.config_sampler).to(self.device)

        # Gather initial values, including the force of the initial distribution
        if self.reverse_kl:
            x_ = torch.autograd.Variable(samples, requires_grad=True)
            lq0x = self.sampler._base_log_prob(samples)
            initial_px = self.sampler._base_log_prob(x_)
            sign = -1
        else:
            x_ = torch.autograd.Variable(samples, requires_grad=True)
            actions = self.action(samples)
            initial_px = -self.action(x_)
            sign = +1

        dlogqdx = torch.autograd.grad(
            initial_px.sum(), x_, allow_unused=True, retain_graph=True
        )[0]

        # Perform piggy back forward or revserse, to get \partial log rho_\theta(x)/ \partial x
        x, delta_trans, dlogqdx = self.sampler.piggy_back_forward(
            samples, dlogqdx, reverse=not self.reverse_kl
        )

        # Get final values, including the force of the final distribution
        if self.reverse_kl:
            actions = self.action(x.reshape(-1, *self.lat_shape))
            final_px = -actions
        else:
            lq0x = self.sampler._base_log_prob(x)
            final_px = -lq0x
        log_q = lq0x - sign * delta_trans
        dfinalpdx = torch.autograd.grad(
            sign * final_px.sum(), x, allow_unused=True, retain_graph=True
        )[0]

        # Combine the gradients
        dLdx = (dlogqdx + dfinalpdx) / self.batch_size
        log_weights = -(actions + log_q).detach()
        loss_mean = sign * log_weights.mean()

        # Add the gradient-yielding term
        grad_term = (x * dLdx.detach()).sum()
        loss_mean += grad_term - grad_term.detach()

        return (
            loss_mean,
            (log_weights).detach().cpu(),
            actions.detach().cpu(),
        )

    def __str__(self):
        return f"FastPath-{'QP' if self.reverse_kl else 'PQ'}"


def load_loss(
    cfg: Namespace,
    flow: torch.nn.Module,
    action: Action,
    device: torch.device,
    config_sampler: Union[Iterable, None] = None,
):
    """
    Initializes and loads Loss
    """
    if cfg.gradient_estimator == "FastDropInPQ":
        loss = FastDropInPathPQ(
            model=flow,
            config_sampler=config_sampler,
            action=action,
            lat_shape=cfg.lat_shape,
            device=device,
        )
    elif "fastPath" in cfg.gradient_estimator:
        reverse = "qp" in cfg.gradient_estimator.lower()
        loss = FastPath(
            model=flow,
            config_sampler=config_sampler,
            action=action,
            reverse=reverse,
            lat_shape=cfg.lat_shape,
            batch_size=cfg.batch_size,
            device=device,
        )
    elif cfg.gradient_estimator == "ML":
        loss = MaximumLikelihood(
            model=flow,
            config_sampler=config_sampler,
            action=action,
            lat_shape=cfg.lat_shape,
            device=device,
        )
    elif cfg.gradient_estimator == "RepQP":
        loss = RepQP(
            model=flow,
            action=action,
            lat_shape=cfg.lat_shape,
            batch_size=cfg.batch_size,
        )
    elif cfg.gradient_estimator == "DropInQP":
        loss = DropInPathQP(
            model=flow,
            action=action,
            lat_shape=cfg.lat_shape,
            batch_size=cfg.batch_size,
        )

    else:
        raise ValueError(
            f"{cfg.gradient_estimator} not in ['RepQP', 'ML', 'fastPathPQ', 'fastPathQP', 'DropInQP', 'FastDropInPQ']"
        )
    return loss
