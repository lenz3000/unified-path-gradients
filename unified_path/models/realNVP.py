import numpy as np
import torch
import torch.nn as nn
from functools import partial

from .modules import Tanhgrow, Unsqueeze, Squeeze

COUPLINGS = {}


def register_base_dist(self, base_dist_str):
    # changeable with extra line self.dist_params.requires_grad = False
    if base_dist_str == "Uniform":
        self.dist_params = nn.Parameter(
            torch.tensor([-10.0, 10.0], requires_grad=False)
        )
        self.dist_params.requires_grad_(False)
        self.base_dist = lambda: torch.distributions.Uniform(*self.dist_params)
    else:
        self.dist_params = nn.Parameter(torch.tensor([0.0, 1.0], requires_grad=False))
        self.dist_params.requires_grad_(False)
        self.base_dist = lambda: torch.distributions.Normal(*self.dist_params)


def register_coupling(name):
    def wrap(clss):
        COUPLINGS[name] = clss
        return clss

    return wrap


class Coupling(nn.Module):
    def __init__(self, mask_config):
        """Initialize a coupling layer.

        Args:
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super().__init__()
        self.mask_config = mask_config

    @classmethod
    def partial(clss, *args, **kwargs):
        return partial(clss, *args)


@register_coupling("conv")
class ConvCoupling(Coupling):
    def __init__(self, lat_shape, mask_config, channels=8, nblocks=4, bias=False):
        super().__init__(mask_config=mask_config)
        PConv2d = partial(
            nn.Conv2d, kernel_size=3, padding=1, padding_mode="circular", bias=bias
        )
        self.layers = nn.Sequential(
            *[
                Unsqueeze(),
                PConv2d(in_channels=1, out_channels=channels),
                Tanhgrow(),
                nn.Sequential(
                    *[
                        nn.Sequential(
                            PConv2d(in_channels=channels, out_channels=channels),
                            Tanhgrow(),
                        )
                        for _ in range(nblocks)
                    ]
                ),
                PConv2d(in_channels=channels, out_channels=1),
                Squeeze(),
            ]
        )

    def forward(self, input):
        mask = self._get_mask(input)
        return input + (1 - mask) * self.layers(input * mask)

    def reverse(self, input):
        mask = self._get_mask(input)
        return input - (1 - mask) * self.layers(input * mask)

    def _get_mask(self, input):
        return torch.from_numpy(
            (np.indices(input.shape).sum(0) + self.mask_config) % 2
        ).to(input.device)


@register_coupling("AltFCS")
class AltFCSCoupling(Coupling):
    def __init__(
        self,
        lat_shape,
        mask_config,
        mid_dim=1000,
        nblocks=4,
        bias=True,
        activation="tanh",
    ):
        super().__init__(mask_config=mask_config)

        self.lat_shape = lat_shape
        if activation == "tanh":
            act = nn.Tanh
        elif activation == "ReLU":
            act = nn.ReLU
        else:
            act = nn.Softplus

        in_out_dim = np.prod(lat_shape)
        self.layers = nn.Sequential(
            nn.Linear(in_out_dim // 2, mid_dim, bias=bias),
            act(),
            nn.Sequential(
                *[
                    nn.Sequential(
                        nn.Linear(mid_dim, mid_dim, bias=bias),
                        act(),
                    )
                    for _ in range(nblocks)
                ]
            ),
            nn.Linear(mid_dim, in_out_dim, bias=bias),
        )

    def forward(self, x):
        return self._apply_coupling(x)

    def reverse(self, x):
        return self._apply_coupling(x, reverse=True)

    def _apply_coupling(self, x, reverse=False):
        x = x.reshape(x.shape[0], -1)

        on, off = self._split(x)

        shiftscale = self.layers(off)
        shift, logscale = self._split(shiftscale)
        if reverse:
            on = (on - shift) / logscale.exp()
        else:
            on = on * logscale.exp() + shift
        logdet = logscale.sum(-1)

        x = self._join(on, off)

        return x.reshape((x.shape[0],) + tuple(self.lat_shape)), logdet

    def _split(self, x):
        B, W = x.shape
        x = x.reshape((B, W // 2, 2))
        if self.mask_config:
            on, off = x[:, :, 0], x[:, :, 1]
        else:
            off, on = x[:, :, 0], x[:, :, 1]

        return on, off

    def _join(self, on, off):
        if self.mask_config:
            x = torch.stack((on, off), dim=2)
        else:
            x = torch.stack((off, on), dim=2)

        return x


@register_coupling("NormAltFCS")
class NormAltFCSCoupling(AltFCSCoupling):
    """Only difference is that we apply torch.nn.utils.parametrizations.weight_norm to the Linear layers during
    construction."""

    def __init__(
        self,
        lat_shape,
        mask_config,
        mid_dim=1000,
        nblocks=4,
        bias=True,
        activation="tanh",
    ):
        super().__init__(
            lat_shape=lat_shape,
            mask_config=mask_config,
            mid_dim=mid_dim,
            nblocks=nblocks,
            bias=bias,
            activation=activation,
        )

        self.lat_shape = lat_shape
        if activation == "tanh":
            act = nn.Tanh
        elif activation == "ReLU":
            act = nn.ReLU
        else:
            act = nn.Softplus

        in_out_dim = np.prod(lat_shape)
        self.layers = nn.Sequential(
            nn.Linear(in_out_dim // 2, mid_dim, bias=bias),
            act(),
            nn.Sequential(
                *[
                    nn.Sequential(
                        # nn.BatchNorm1d(mid_dim, momentum=0.01, affine=True),
                        torch.nn.utils.parametrizations.weight_norm(
                            nn.Linear(mid_dim, mid_dim, bias=bias)
                        ),
                        act(),
                    )
                    for _ in range(nblocks)
                ]
            ),
            nn.Linear(mid_dim, in_out_dim, bias=bias),
        )


class RealNVP(nn.Module):
    def __init__(
        self,
        lat_shape,
        coupling_factory,
        ncouplings=6,
        nblocks=4,
        mask_config=1,
        bias=True,
        n_hidden=1000,
        activation="tanh",
        base_dist="Gaussian",
        loss=None,
        gen_wrapper=None,
    ):
        super().__init__()
        self.lat_shape = lat_shape

        register_base_dist(self, base_dist)
        if type(coupling_factory) is str:
            coupling_factory = COUPLINGS[coupling_factory]
        assert coupling_factory in (
            AltFCSCoupling,
            NormAltFCSCoupling,
        ), f"RealNVP needs scaling but is {coupling_factory} but should be (AltFCSCoupling, NormAltFCSCoupling)"
        self.couplings = nn.ModuleList(
            [
                coupling_factory(
                    lat_shape=lat_shape,
                    nblocks=nblocks,
                    mask_config=(mask_config + i) % 2,
                    mid_dim=n_hidden,
                    bias=bias,
                    activation=activation,
                )
                for i in range(ncouplings)
            ]
        )

    def g(self, x):
        """
        g: Z -> X
        Evaluates forward flow, transforming z to x
        simultaneous applies change of variable to probability
        :param x: latent dim z which is transformed to x
        :return:
        """
        log_det = self._prior_log_prob(x)  # log q(z)

        for coupling in self.couplings:
            # evaluate coupling layers and adding log det Jacobian to log det
            x, c_log_det = coupling(x)
            log_det -= c_log_det

        return x, log_det

    def f(self, x):
        """Evaluate transformation f: X -> Z.

        :param x:
        :return:
        """
        device = list(self.parameters())[0].device
        log_det = torch.zeros((x.shape[0],), device=device)
        for coupling in reversed(self.couplings):
            x, c_log_det = coupling.reverse(x)
            log_det -= c_log_det

        # add prior entropy estimate
        log_prob = self._prior_log_prob(x) + log_det

        return x, log_prob

    def log_prob(self, x):
        return self.f(x)[1]

    def sample_base(self, batch_size):
        base = self.base_dist()
        self.last_sample = base.sample((batch_size, *self.lat_shape))
        return self.last_sample

    def sample_with_logprob(self, batch_size):
        """Generates samples.

        Args:
                batch_size: number of samples to generate.
        Returns:
                samples from the data space X and their log probs.
        """
        z = self.sample_base(batch_size)
        return self.g(z)

    def forward(self, z):
        return self.g(z)

    def reverse(self, x):
        return self.f(x)

    def _prior_log_prob(self, z):
        z = z.reshape(z.shape[0], -1)

        log_prob_per_sample = self.base_dist().log_prob(z)
        return log_prob_per_sample.sum(-1)

    def load(self, checkpoint, device):
        self.load_state_dict(torch.load(checkpoint, map_location=device)["net"])


class PathCouplingWrapper(torch.nn.Module):
    """Wrapper for coupling layers for implementing the path gradient Only when training and for
    the forward pass the path gradient is applied."""

    def __init__(self, coupling: AltFCSCoupling):
        super().__init__()
        self.coupling = coupling

    def forward(self, z, dlogqdz=None):
        if not self.coupling.training or not torch.is_grad_enabled() or dlogqdz is None:
            return self.coupling(z)
        return self._apply_coupling(z, dlogqdz)

    def reverse(self, x, dlogqdx=None):
        if not self.coupling.training or not torch.is_grad_enabled() or dlogqdx is None:
            return self.coupling.reverse(x)
        return self._apply_coupling(x, dlogqdx, reverse=True)

    def _apply_coupling(self, z, dlogqdz, reverse=False):
        z = z.reshape(z.shape[0], -1)
        dlogqdz = dlogqdz.reshape(dlogqdz.shape[0], -1)

        on, off = self.coupling._split(z)
        dLdz_on, dLdz_off = self.coupling._split(dlogqdz)

        off.requires_grad_(True)
        on.requires_grad_(True)

        shiftscale = self.coupling.layers(off)
        shift, logscale = self.coupling._split(shiftscale)
        if reverse:
            on = (on - shift) / logscale.exp()
            # The detach here has probably an effect of -0.1% runtime
            on_back = on.detach() * logscale.exp() + shift
            sign = +1
        else:
            on = on * logscale.exp() + shift
            on_back = (on.detach() - shift) / logscale.exp()
            sign = -1
        logdet = logscale.sum(-1)
        dLdz_off1 = torch.autograd.grad(
            (dLdz_on.detach() * on_back).sum() + sign * logdet.sum(),
            off,
            retain_graph=True,
            allow_unused=False,
        )[0]
        dLdx_off = dLdz_off + dLdz_off1
        if reverse:
            dLdx_on = (dLdz_on * logscale.exp()).detach()
        else:
            dLdx_on = (dLdz_on / logscale.exp()).detach()

        x = self.coupling._join(on, off)
        dLdx = self.coupling._join(dLdx_on, dLdx_off).detach()

        return (
            x.reshape((z.shape[0],) + tuple(self.coupling.lat_shape)),
            logdet,
            dLdx.reshape((z.shape[0],) + tuple(self.coupling.lat_shape)),
        )


class GeneralPathCouplingWrapper(PathCouplingWrapper):
    def _apply_coupling(self, z, dlogqdz, reverse=False):
        z = z.reshape(z.shape[0], -1)
        dlogqdz = dlogqdz.reshape(dlogqdz.shape[0], -1)

        # Preparing the variables
        z.requires_grad_(True)
        z_on, z_off = self.coupling._split(z)
        z_ = self.coupling._join(z_on, z_off)
        dLdz_on, dLdz_off = self.coupling._split(dlogqdz)

        # Applying the coupling
        x, c_log_det = self.coupling._apply_coupling(z_, reverse=reverse)

        # Redoing the splitting
        x_on, x_off = self.coupling._split(x)

        # Getting the partial derivatives
        dx_ondz_on = torch.autograd.grad(x_on.sum(), z_on, retain_graph=True)[0]
        ddldz = torch.autograd.grad(
            c_log_det.sum() if reverse else -c_log_det.sum(),
            (z_on, z_off),
            retain_graph=True,
            allow_unused=False,
        )
        # Implicit gradients
        dLdx_on = (dLdz_on + ddldz[0]) / dx_ondz_on
        ddx_ondz_off = torch.autograd.grad(
            x_on,
            z_off,
            grad_outputs=dLdx_on,
            retain_graph=True,
        )[0]
        # Since the off part is passed through, so are the gradients
        dLdx_off = dLdz_off - ddx_ondz_off + ddldz[1]

        # Appending the piggy back gradients to the inputs

        x = self.coupling._join(x_on, x_off)
        dLdx = self.coupling._join(dLdx_on, dLdx_off).detach()

        return (
            x.reshape((x.shape[0],) + tuple(self.coupling.lat_shape)),
            c_log_det,
            dLdx.reshape((x.shape[0],) + tuple(self.coupling.lat_shape)),
        )


class RealNVP_Path(RealNVP):
    def __init__(self, *args, gen_wrapper=False, **kwargs):
        super().__init__(*args, **kwargs)
        if gen_wrapper:
            wrapper = GeneralPathCouplingWrapper
        else:
            wrapper = PathCouplingWrapper
        self.couplings = nn.ModuleList(
            [wrapper(coupling) for coupling in self.couplings]
        )
        self.loss = kwargs["loss"]
        assert self.loss in [
            "fastPathPQ",
            "fastPathQP",
        ], f"Loss must be one of fastPathQP or fastPathPQ - is {self.loss}"

    def piggy_back_forward(self, x, dLdx, reverse=False):
        if not self.training or not torch.is_grad_enabled():
            return super().g(x)

        log_det = torch.zeros(x.shape[0], device=x.device)

        sign = 1 if reverse else -1

        couplings = self.couplings[::-1] if reverse else self.couplings

        # Propagate x, log q and force through flow couplings
        for coupling in couplings:
            x, c_log_det, dLdx = coupling._apply_coupling(x, dLdx, reverse=reverse)
            log_det += sign * c_log_det

        return x, log_det, dLdx

    def g(self, x):
        """
        g: Z -> X
        Evaluates forward flow, transforming z to x
        simultaneous applies change of variable to probability
        :param x: latent dim z which is transformed to x
        :return:
        """
        log_det = self._prior_log_prob(x)  # log q(z)

        for coupling in self.couplings:
            # evaluate coupling layers and adding log det Jacobian to log det
            x, c_log_det = coupling(x)
            log_det -= c_log_det

        return x, log_det

    def f(self, x):
        """Evaluate transformation f: X -> Z.

        :param x:
        :return:
        """
        device = list(self.parameters())[0].device
        log_det = torch.zeros((x.shape[0],), device=device)
        for coupling in reversed(self.couplings):
            x, c_log_det = coupling.reverse(x)
            log_det -= c_log_det

        # add prior entropy estimate
        log_prob = self._prior_log_prob(x) + log_det

        return x, log_prob
