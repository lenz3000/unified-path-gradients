import torch
from torch.distributions import Normal as norm_dist
from torch.utils.data import DataLoader
from argparse import Namespace

from unified_path.action import Action
from unified_path.models.realNVP import (
    load_RealNVP,
    infinite_sampling,
    transfer_path_flow_to_normal_flow,
)
from unified_path.loss import load_loss
from unified_path.models import U1Flow_Path
from unified_path.utils import gather_grads


class Normal(Action):
    def __init__(self, device="cpu"):
        self.normal = norm_dist(
            torch.tensor([1.0], device=device), torch.tensor([2.0], device=device)
        )

    def evaluate(self, x):
        if len(x.shape) == 1:
            x = x[:, None]
        return -self.normal.log_prob(x).sum(-1)

    def sample(self, n):
        return self.normal.rsample(n)[..., 0]


device = "cpu"
norm_action = Normal(device)
torch.manual_seed(0)
n_samples = 16


gen_dict = {
    "gradient_estimator": "fastPathQP",
    "dim": 2,
    "n_coupling_layers": 4,
    "n_blocks": 2,
    "hidden": 32,
}


def get_fast_slow_twins():
    gen_dict_ = dict(gen_dict)

    fast_flow = load_RealNVP(Namespace(**gen_dict)).to(device)
    gen_dict_["gradient_estimator"] = "RepQP"
    slow_flow = load_RealNVP(Namespace(**gen_dict_)).to(device)

    transfer_path_flow_to_normal_flow(slow_flow, fast_flow)
    fast_flow.zero_grad()
    slow_flow.zero_grad()
    return fast_flow, slow_flow


def test_fastLoss():
    fast_flow, slow_flow = get_fast_slow_twins()
    eps_train = fast_flow.sample_base(n_samples)

    fast_pathQP = load_loss(
        Namespace(
            gradient_estimator="fastPathQP",
            lat_shape=[2],
            batch_size=n_samples,
        ),
        fast_flow,
        norm_action,
        device,
    )
    repQP = load_loss(
        Namespace(
            gradient_estimator="RepQP",
            lat_shape=[2],
            batch_size=n_samples,
        ),
        slow_flow,
        norm_action,
        device,
    )
    floss = fast_pathQP(eps_train)
    sloss = repQP(eps_train)
    assert torch.allclose(floss[0], sloss[0], atol=1e-5)


def check_fastPath_RealNVP_grads(direction):
    fast_flow, slow_flow = get_fast_slow_twins()
    eps_train = fast_flow.sample_base(n_samples)
    config_sampler = None
    if direction == "PQ":
        config_sampler = infinite_sampling(
            DataLoader(
                fast_flow.sample_base(2 * n_samples),
                batch_size=n_samples,
                drop_last=False,
            )
        )
    fast_pathQP = load_loss(
        Namespace(
            gradient_estimator=f"fastPath{direction}",
            lat_shape=[2],
            batch_size=n_samples,
        ),
        fast_flow,
        norm_action,
        device,
        config_sampler,
    )
    slow_pathQP = load_loss(
        Namespace(
            gradient_estimator=f"{'Fast' if direction == 'PQ' else ''}DropIn{direction}",
            lat_shape=[2],
            batch_size=n_samples,
        ),
        slow_flow,
        norm_action,
        device,
        config_sampler,
    )
    floss = fast_pathQP(eps_train)
    sloss = slow_pathQP(eps_train)
    assert torch.allclose(floss[0], sloss[0], atol=1e-5)
    floss[0].backward()
    sloss[0].backward()

    fgrad = gather_grads(fast_flow)
    sgrad = gather_grads(slow_flow)
    assert torch.allclose(fgrad, sgrad, atol=1e-5)


def test_fastPath_RealNVP():
    check_fastPath_RealNVP_grads("QP")
    check_fastPath_RealNVP_grads("PQ")


def test_piggyBack_CouplingFlows():
    def piggy_back_eval(flow):
        eps_train = flow.sample_base(n_samples)
        x, logq = flow.g(eps_train)
        logdet = logq - flow._prior_log_prob(eps_train)
        x2, logdet2, _ = flow.piggy_back_forward(eps_train, torch.zeros_like(eps_train))
        assert torch.allclose(x, x2, atol=1e-5)
        assert torch.allclose(logdet, logdet2, atol=1e-5)

    flow = load_RealNVP(Namespace(**gen_dict)).to(device)
    piggy_back_eval(flow)

    flow = U1Flow_Path(
        lat_shape=[4, 4],
        n_mixture_comps=2,
        n_layers=2,
    )
    piggy_back_eval(flow)


def test_invert_CouplingFlows():
    def invert_eval(flow):
        eps_train = flow.sample_base(n_samples)
        x, logq = flow.g(eps_train)
        eps2, logq2 = flow.f(x)

        assert torch.allclose(eps_train, eps2, atol=1e-5)
        assert torch.allclose(logq, logq2, atol=1e-5)

    flow = load_RealNVP(Namespace(**gen_dict)).to(device)
    invert_eval(flow)

    flow = U1Flow_Path(
        lat_shape=[4, 4],
        n_mixture_comps=2,
        n_layers=2,
    )
    invert_eval(flow)
