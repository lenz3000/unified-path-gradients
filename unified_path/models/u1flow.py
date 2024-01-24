import torch

import torch.nn as nn
import numpy as np

from ..utils import invert_transform_bisect_wgrad
from ..lattice import plaquette, single_stripes, double_stripes
from .modules import ConvNet


class U1Flow(nn.Module):
    # Originally implemented in https://arxiv.org/abs/2101.08176
    def __init__(
        self,
        lat_shape,
        n_layers=16,
        n_mixture_comps=2,
        hidden_sizes=[8, 8],
        kernel_size=3,
        inv_prec=1e-6,
    ):
        super().__init__()

        self.lat_shape = lat_shape
        self.link_shape = (2, *lat_shape)
        self.register_buffer("high", torch.tensor(2.0 * np.pi))
        self.register_buffer("low", torch.tensor(0.0))

        layers = []
        for i in range(n_layers):
            mu = i % 2
            off = (i // 2) % 4
            in_channels = 2  # x - > (cos(x), sin(x))
            out_channels = n_mixture_comps + 1  # for mixture s and t, respectively
            net = ConvNet(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_sizes=hidden_sizes,
                kernel_size=kernel_size,
            )
            plaq_coupling = PlaquetteCoupling(
                net,
                mask_shape=lat_shape,
                mask_mu=mu,
                mask_off=off,
                inv_prec=inv_prec,
                inv_max_iter=50000,
            )
            link_coupling = U1Coupling(
                lattice_shape=lat_shape,
                mask_mu=mu,
                mask_off=off,
                plaquette_coupling=plaq_coupling,
            )
            layers.append(link_coupling)

        self.couplings = nn.ModuleList(layers)

    @property
    def base_distribution(self):
        return torch.distributions.Uniform(self.low, self.high)

    def sample_base(self, batch_size):
        return self.base_distribution.sample((batch_size, *self.link_shape))

    def _prior_log_prob(self, z):
        # Adding 0. so that there are some gradients
        return self.base_distribution.log_prob(z).sum(dim=(1, 2, 3)) + 0.0 * z.sum(
            dim=(1, 2, 3)
        )

    def sample_with_logprob(self, batch_size):
        z = self.base_distribution.sample((batch_size, *self.link_shape))

        axes = tuple(range(1, len(z.shape)))
        log_q_prior = self.base_distribution.log_prob(z).sum(dim=axes)

        z, log_q_flow = self.forward(z)

        return z, log_q_flow + log_q_prior

    def forward(self, z):
        logq = 0.0
        for coupling in self.couplings:
            z, logJ = coupling(z)
            logq = logq - logJ

        return z, logq

    def g(self, z):
        axes = tuple(range(1, len(z.shape)))
        log_q_prior = self.base_distribution.log_prob(z).sum(dim=axes)

        z, log_q_flow = self.forward(z)

        return z, log_q_flow + log_q_prior

    def reverse(self, x):
        logq = 0.0
        for coupling in reversed(self.couplings):
            x, logJ = coupling.reverse(x)
            logq = logq + logJ

        return x, logq

    def f(self, z):
        z, logdet = self.reverse(z)
        axes = tuple(range(1, len(z.shape)))
        log_q_prior = self.base_distribution.log_prob(z).sum(dim=axes)
        return z, logdet + log_q_prior

    def log_prob(self, x):
        z, logq = self.reverse(x)
        axes = tuple(range(1, len(z.shape)))
        log_q_prior = self.base_distribution.log_prob(z).sum(dim=axes)
        return logq + log_q_prior


class U1Flow_Path(U1Flow):
    def __init__(
        self,
        lat_shape,
        n_layers=16,
        n_mixture_comps=2,
        hidden_sizes=[8, 8],
        kernel_size=3,
    ):
        super().__init__(
            lat_shape, n_layers, n_mixture_comps, hidden_sizes, kernel_size
        )
        self.couplings = nn.ModuleList(
            [U1CouplingPathWrapper(c) for c in self.couplings]
        )

    def piggy_back_forward(self, x, dLdx, reverse=False):
        assert not reverse, "This PathWrapperFlow is not made for reverse piggy_back"
        if not self.training or not torch.is_grad_enabled():
            return super().forward(x)

        log_det = torch.zeros(x.shape[0], device=x.device)

        # Propagate x, log q and force through flow couplings
        for coupling in self.couplings:
            x, c_log_det, dLdx = coupling._apply_coupling(x, dLdx)
            log_det -= c_log_det

        return x, log_det, dLdx


class U1CouplingPathWrapper(nn.Module):
    def __init__(self, coupling) -> None:
        super().__init__()
        self.coupling = coupling

    def forward(self, z, dlogqdz=None):
        if not self.coupling.training or not torch.is_grad_enabled() or dlogqdz is None:
            return self.coupling(z)
        return self._apply_coupling(z, dlogqdz)

    def reverse(self, x, dlogqdx=None):
        if not self.coupling.training or not torch.is_grad_enabled() or dlogqdx is None:
            return self.coupling.reverse(x)
        raise NotImplementedError("Reverse piggyback not implemented")

    def _apply_coupling(self, z, dlogqdz):
        # Preparing the variables
        z.requires_grad_(True)

        # Due to its gauge invariance, this coupling mask is more complex than a standard one
        m_trans = self.coupling.active_mask
        m_cond = 1 - m_trans

        def split(x):
            x_cond = torch.masked_select(x, m_cond.bool()).reshape(
                (*dlogqdz.shape[:1], -1)
            )
            x_trans = torch.masked_select(x, m_trans.bool()).reshape(
                (*dlogqdz.shape[:1], -1)
            )
            return x_cond, x_trans

        def join(x_cond, x_trans):
            x = torch.zeros_like(z)
            x.masked_scatter_(m_cond.bool(), x_cond.reshape(-1))
            x.masked_scatter_(m_trans.bool(), x_trans.reshape(-1))
            return x

        cond_z, trans_z = split(z)
        z_ = join(cond_z, trans_z)
        dlogqdz_cond, dlogqdz_trans = split(dlogqdz)

        # Applying the coupling
        x, c_log_det_d = self.coupling(z_, sum_d=False)
        c_log_det = c_log_det_d.reshape((x.shape[0], -1)).sum(1)
        c_log_det_d = c_log_det_d[:, self.coupling.plaquette_coupling.m_active.bool()]

        # Redoing the splitting
        _, trans_x = split(x)

        dx_transdz_trans = c_log_det_d.exp()
        # Gradient resulting from log det Jacobian
        ddldz = torch.autograd.grad(
            -c_log_det.sum(),
            (cond_z, trans_z),
            retain_graph=True,
            allow_unused=False,
        )
        # Implicit gradients
        dLdx_trans = (dlogqdz_trans + ddldz[1]) / dx_transdz_trans
        ddx_transdz_cond = torch.autograd.grad(
            trans_x,
            cond_z,
            grad_outputs=dLdx_trans,
            retain_graph=True,
        )[0]

        dLdx_cond = dlogqdz_cond - ddx_transdz_cond + ddldz[0]

        # Appending the piggy back gradients to the inputs
        dLdx = join(dLdx_cond, dLdx_trans)

        return x, c_log_det, dLdx


class U1Coupling(nn.Module):
    def __init__(self, lattice_shape, mask_mu, mask_off, plaquette_coupling, alt=False):
        super().__init__()

        self.register_buffer(
            "active_mask",
            self._active_link_mask(
                (len(lattice_shape), *lattice_shape), mask_mu, mask_off
            ),
        )
        self.plaquette_coupling = plaquette_coupling
        self.alt = alt

    def forward(self, u, sum_d=True):
        plaq = plaquette(u)
        new_plaq, logJ = self.plaquette_coupling(plaq, sum_d=sum_d)
        uprime = self._update_links(u, new_plaq, plaq)
        return uprime, logJ

    def reverse(self, uprime):
        new_plaq = plaquette(uprime)
        plaq, logJ = self.plaquette_coupling.reverse(new_plaq)
        u = self._update_links(uprime, new_plaq, plaq, reverse=True)
        return u, logJ

    def _update_links(self, links, new_plaq, plaq, reverse=False):
        """[summary]
        Updates the links by using new plaquette and old plaquette
        Args:
            links ([torch.tensor]): gauge links of shape [B, 2, H, W]
            new_plaq ([torch.tensor]): new plaquette values of shape [B, H, W]
            plaq ([torch.tensor]): old plaquette values of shape [B, H, W]
            reverse (bool, optional): Forward or reverse direction. Defaults to False.

        Returns:
            [torch.tensor]: updated links of shape [B, 2, H, W]
        """
        delta_plaq = new_plaq - plaq
        if reverse:
            delta_plaq = -delta_plaq
        # active plaquette before active link
        if self.alt:
            delta_plaq = -1.0 * delta_plaq
            delta_plaq_y = torch.roll(delta_plaq, 1, 1)
            delta_plaq_x = torch.roll(delta_plaq, 1, 2)
            delta_links = torch.stack((delta_plaq_x, -delta_plaq_y), dim=1)
        else:
            delta_links = torch.stack(
                (delta_plaq, -delta_plaq), dim=1
            )  # signs for U vs Udagger

        return (
            self.active_mask * torch.remainder(delta_links + links, 2 * np.pi)
            + (1 - self.active_mask) * links
        )

    def _active_link_mask(self, shape, mu, off):
        """
        Stripes mask looks like in the `mu` channel (mu-oriented links)::

            1 0 0 0 1 0 0 0 1 0 0
            1 0 0 0 1 0 0 0 1 0 0
            1 0 0 0 1 0 0 0 1 0 0

        where vertical stripe is the `mu` direction and the pattern is offset in the nu
        direction by `off` (mod 4). The other channel is identically 0.
        """
        assert len(shape) == 3, "need to pass shape suitable for 2D gauge theory"
        assert shape[0] == len(shape[1:]), "first dim of shape must be Nd"
        assert mu in (0, 1), "mu must be 0 or 1"
        mask = np.zeros(shape)
        if mu == 0:
            mask[mu, :, 0::4] = 1
        elif mu == 1:
            mask[mu, 0::4] = 1
        nu = 1 - mu
        mask = np.roll(mask, off, axis=nu + 1)
        return torch.from_numpy(mask).float()


def tan_transform(x, s):
    return torch.remainder(2 * torch.atan(torch.exp(s) * torch.tan(x / 2)), 2 * np.pi)


def tan_transform_logJ(x, s):
    return -torch.log(
        torch.exp(-s) * torch.cos(x / 2) ** 2 + torch.exp(s) * torch.sin(x / 2) ** 2
    )


def mixture_tan_transform(x, s):
    assert len(x.shape) == len(s.shape), f"dim mismatch x and s {x.shape} vs {s.shape}"
    return torch.mean(tan_transform(x, s), dim=1, keepdim=True)


def mixture_tan_transform_logJ(x, s):
    assert len(x.shape) == len(s.shape), f"dim mismatch x and s {x.shape} vs {s.shape}"
    return torch.logsumexp(tan_transform_logJ(x, s), dim=1) - np.log(s.shape[1])


class PlaquetteCoupling(nn.Module):
    def __init__(
        self, net, mask_shape, mask_mu, mask_off, inv_prec=5e-5, inv_max_iter=1000
    ):
        super().__init__()
        assert len(mask_shape) == 2, "only 2d supported"
        self.net = net
        self.inv_prec = inv_prec
        self.inv_max_iter = inv_max_iter

        active, passive, frozen = self._plaquette_masks(mask_shape, mask_mu, mask_off)
        self.register_buffer("m_active", active)
        self.register_buffer("m_passive", passive)
        self.register_buffer("m_frozen", frozen)

    def forward(self, p, sum_d=True):
        frozen = self.m_frozen * p
        active = self.m_active * p
        passive = self.m_passive * p

        weights = self.net(self._frozen_preproc(p))
        assert weights.shape[1] >= 2, "CNN must output n_mix (s_i) + 1 (t) channels"
        s, t = weights[:, :-1], weights[:, -1]

        logJ = self.m_active * mixture_tan_transform_logJ(active.unsqueeze(1), s)
        axes = tuple(range(1, len(logJ.shape)))
        if sum_d:
            logJ = torch.sum(logJ, dim=axes)

        p_prime = self.m_active * mixture_tan_transform(active.unsqueeze(1), s).squeeze(
            1
        )
        p_prime = (
            self.m_active * torch.remainder(p_prime + t, 2 * np.pi) + passive + frozen
        )
        return p_prime, logJ

    def reverse(self, p_prime):
        frozen = self.m_frozen * p_prime
        passive = self.m_passive * p_prime

        weights = self.net(self._frozen_preproc(p_prime))
        assert weights.shape[1] >= 2, "CNN must output n_mix (s_i) + 1 (t) channels"
        s, t = weights[:, :-1], weights[:, -1]

        p1 = torch.remainder(self.m_active * (p_prime - t).unsqueeze(1), 2 * np.pi)

        def transform(p):
            return self.m_active * mixture_tan_transform(p, s)

        p1 = invert_transform_bisect_wgrad(
            p1, f=transform, tol=self.inv_prec, max_iter=self.inv_max_iter
        )
        logJ = self.m_active * mixture_tan_transform_logJ(p1, s)
        axes = tuple(range(1, len(logJ.shape)))
        logJ = -torch.sum(logJ, dim=axes)
        p1 = p1.squeeze(1)

        x = self.m_active * p1 + passive + frozen

        return x, logJ

    def _frozen_preproc(self, p):
        frozen = self.m_frozen * p
        return torch.stack((torch.cos(frozen), torch.sin(frozen)), dim=1)

    def _plaquette_masks(self, mask_shape, mask_mu, mask_off):
        frozen = double_stripes(mask_shape, mask_mu, mask_off + 1)
        active = single_stripes(mask_shape, mask_mu, mask_off)
        passive = 1 - frozen - active

        return active, passive, frozen
