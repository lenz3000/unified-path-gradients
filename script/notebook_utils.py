import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_action(
    action,
    sampler=None,
    nlevels=10,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
):
    lim = 3.5
    step_size = 0.05
    base = np.arange(-lim, lim, step_size)
    x, y = np.meshgrid(base, base)
    grid = np.concatenate([x.flatten()[:, None] for x in [x, y]], axis=-1)  # [..., None]
    tensor = torch.tensor(grid, device=device, dtype=torch.float)
    logp_x = -action(tensor).cpu()
    log_q = None
    p_x = (logp_x).exp().detach().numpy()
    kl_pq = None
    zTrue = sum(p_x) * step_size**2
    im = plt.contourf(x, y, p_x.reshape(x.shape), levels=nlevels)
    if sampler is not None:
        sampler = sampler.to("cpu")
        log_q = sampler.log_prob(tensor.cpu())

        kl_pq = step_size**2 * np.sum(
            np.where(p_x > 1e-5, p_x * (logp_x.cpu() - log_q).detach().cpu().numpy(), 0)
        )
        sampler = sampler.to(device)
        im = plt.contour(
            x,
            y,
            log_q.exp().detach().cpu().numpy().reshape(x.shape),
            alpha=0.6,
            linestyles="--",
            cmap="Greys",
            levels=nlevels,
        )
        # plt.gca().clabel(im, inline=True, fontsize=10, colors='green')
    # plt.colorbar(im)
    plt.xlim([-lim, lim])
    plt.ylim([-lim, lim])
    return {"x": x, "y": x, "log p(x)": logp_x, "log q(x)": log_q, "kl(p|q)": kl_pq, "Z": zTrue}