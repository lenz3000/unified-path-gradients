#!/usr/bin/env python
from argparse import Namespace

import click
import torch
from tqdm.auto import tqdm

from unified_path.action import U1GaugeAction
from unified_path.importsamp import estimate_ess_q
from unified_path.models import U1Flow, U1Flow_Path
from unified_path.loss import load_loss, Loss
from torch.optim.lr_scheduler import ReduceLROnPlateau


def load_flow(cfg):
    model = U1Flow if "fastPath" not in cfg.gradient_estimator else U1Flow_Path
    return model(
        lat_shape=cfg.lat_shape[1:],
        n_mixture_comps=cfg.nmixtures,
        n_layers=cfg.n_coupling_layers,
    )


def train(kl_loss: Loss, n_steps):
    optimizer = torch.optim.Adam(kl_loss.sampler.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=0.92,
        verbose=True,
        patience=3000,
        threshold=1e-4,
        min_lr=1e-6,
    )
    with tqdm(range(n_steps)) as pbar:
        for step in pbar:
            optimizer.zero_grad()

            loss_mean, log_w_tilde, actions = kl_loss()

            if loss_mean is None:
                step -= 1
                continue

            loss_mean.backward()
            optimizer.step()
            scheduler.step(log_w_tilde.std())

            if step % 4000 == 0:
                ess = estimate_ess_q(
                    kl_loss.sampler,
                    kl_loss.action,
                    kl_loss.lat_shape,
                    batch_size=2 * actions.shape[0],
                    n_iter=50,
                )
                ess_string = f"ESSq @ {step}: {ess:.3f}"

            pbar.set_description(
                f"step: {step} "
                f"ELBO: {log_w_tilde.mean():8.2e} "
                f"action: {actions.mean():6.2e}+-{actions.std():4.1e}"
                f" {ess_string}"
            )


@click.command()
@click.option("--seed", default=0, help="")
@click.option("--steps", default=1_000, help="Number of iterations")
@click.option("--batch-size", default=1_000, help="Batch size")
@click.option(
    "--gradient-estimator",
    type=click.Choice(["RepQP", "fastPathQP", "DropInQP", "FastDropInPQ"]),
    default="fastPathQP",
    help="Gradient Estimator",
)
@click.option("--n-coupling-layers", default=24, help="Number of coupling layers")
@click.option("--nmixtures", default=6, help="blocks per coupling layer")
@click.option("--L", default=16, type=int, help="Lattice extent")
@click.option("--beta", default=3.0, type=float, help="inverse gauge coupling")
def main(**cfg):
    cfg = Namespace(**cfg)
    cfg.lat_shape = [2, cfg.l, cfg.l]
    print(cfg)

    torch.manual_seed(cfg.seed)
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    action = U1GaugeAction(beta=cfg.beta)
    flow = load_flow(cfg).to(device)

    loss = load_loss(
        cfg=cfg, flow=flow, config_sampler=None, action=action, device=torch.device
    )

    train(loss, cfg.steps)


if __name__ == "__main__":
    main()
