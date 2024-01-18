import numpy as np
import torch


def plaquette(links, mu=0, nu=1):
    """
    Compute U(1) plaquette in the (mu,nu) plane given `links` = arg(U)
    """
    return (
        links[:, mu]
        + torch.roll(links[:, nu], -1, mu + 1)
        - torch.roll(links[:, mu], -1, nu + 1)
        - links[:, nu]
    )


def single_stripes(shape, mu, off):
    """1 0 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 0 1 0 0.

    where vertical is the `mu` direction. Vector of 1 is repeated every 4. The pattern is offset in
    perpendicular to the mu direction by `off` (mod 4).
    """
    assert len(shape) == 2, "need to pass 2D shape"
    assert mu in (0, 1), "mu must be 0 or 1"

    mask = np.zeros(shape).astype(np.uint8)
    if mu == 0:
        mask[:, 0::4] = 1
    elif mu == 1:
        mask[0::4] = 1

    mask = np.roll(mask, off, axis=1 - mu)
    return torch.from_numpy(mask).float()


def double_stripes(shape, mu, off):
    """1 0 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 0 1 0 0 1 0 0 0 1 0 0 0 1 0 0.

    where vertical is the `mu` direction. Vector of 1 is repeated every 4. The pattern is offset in
    perpendicular to the mu direction by `off` (mod 4).
    """
    assert len(shape) == 2, "need to pass 2D shape"
    assert mu in (0, 1), "mu must be 0 or 1"

    mask = np.zeros(shape).astype(np.uint8)
    if mu == 0:
        mask[:, 0::4] = 1
        mask[:, 1::4] = 1
    elif mu == 1:
        mask[0::4] = 1
        mask[1::4] = 1

    mask = np.roll(mask, off, axis=1 - mu)
    return torch.from_numpy(mask).float()
