#!/usr/bin/env python
from argparse import Namespace

import click
import torch

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from unified_path.action import MGM
from unified_path.importsamp import estimate_reverse_ess, estimate_forward_ess
from unified_path.loss import load_loss
from unified_path.models.realNVP import load_RealNVP
from unified_path.utils import infinite_sampling


def train(
    flow,
    train_data,
    test_data,
    kl,
    steps=1000,
    batch_size=500,
):
    optim = torch.optim.Adam(flow.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=steps, eta_min=1e-6
    )
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, drop_last=False
    )
    train_losses, test_losses = [], []
    test_nll = torch.Tensor([0.0])
    train_nll = torch.Tensor([0.0])
    ess = -1.0
    forward_ess = -1.0
    pbar = tqdm(range(steps))
    for i in pbar:
        flow.zero_grad()
        loss, _, _ = kl()

        if loss is not None:
            loss.backward()
        else:
            loss = torch.Tensor([0.0])
        optim.step()
        scheduler.step()
        if (i % 10) == 0 or i == steps:
            with torch.no_grad():
                test_nll = -flow.log_prob(
                    test_data
                ).mean()  # flow probability log q_θ(x)
                nll_info = f" test {test_nll.item():.3e}"
                if train_data is not None:
                    train_nll = -flow.log_prob(train_data).mean()
                    nll_info += f" train {train_nll.item():.3e}"
            ess = estimate_reverse_ess(
                flow,
                kl.action,
                lat_shape=test_data.shape[1:],
                batch_size=1_000,
                n_iter=5,
            )
            forward_ess = estimate_forward_ess(
                model=flow,
                config_sampler=test_dataloader,
                action=kl.action,
                lat_shape=test_data.shape[1:],
                device=loss.device,
            )

            pbar.set_description(
                f"loss: {loss.item():.3e} || NLL {nll_info}"
                f"|| ESS q:{ess:.3f} p:{forward_ess:.3f}"
            )

        train_losses.append(loss.item())
        test_losses.append(test_nll.item())

    return train_losses, test_losses


@click.command()
@click.option("--seed", default=0, help="")
@click.option("--steps", default=1_000, help="Number of iterations")
@click.option("--batch-size", default=1_000, help="Batch size")
@click.option("--hidden", default=1_000, help="Hidden size")
@click.option(
    "--gradient-estimator",
    type=click.Choice(
        ["RepQP", "ML", "fastPathPQ", "fastPathQP", "DropInQP", "FastDropInPQ"]
    ),
    default="fastPathQP",
    help="Gradient Estimator",
)
@click.option("--n-coupling-layers", default=6, help="Number of coupling layers")
@click.option("--n-blocks", default=6, help="blocks per coupling layer")
@click.option("--nsamples", default=10_000, help="Number of samples")
@click.option("--mgm-loc", default=1.0, help="MGM cluster loc")
@click.option("--mgm-scale", default=0.5, help="MGM cluster scale")
@click.option("--dim", default=6, type=int, help="dim of data")
def main(**cfg):
    """Function to get all CLI arguments into script."""

    cfg = Namespace(**cfg)
    cfg.lat_shape = [cfg.dim]
    print(cfg)

    torch.manual_seed(cfg.seed)
    print("Setting up action and sampling ...")
    num_test_samples = 10_000

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    target_action = MGM(
        dim=cfg.dim,
        device=device,
        clusters=2,
        cluster_loc=cfg.mgm_loc,
        cluster_scale=cfg.mgm_scale,
    ).to(device)

    train_data = None
    if cfg.gradient_estimator in ["ML", "FastDropInPQ", "fastPathPQ"]:
        train_data = target_action.sample((cfg.nsamples,)).to(device)
        config_sampler = infinite_sampling(
            DataLoader(
                train_data, batch_size=cfg.batch_size, shuffle=True, drop_last=False
            )
        )
    else:
        config_sampler = None
    test_data = target_action.sample((num_test_samples,)).to(device)

    flow = load_RealNVP(cfg).to(device)

    loss = load_loss(cfg, flow, target_action, device, config_sampler=config_sampler)

    train(
        flow=flow,
        train_data=train_data,
        test_data=test_data,
        kl=loss,
        steps=cfg.steps,
        batch_size=cfg.batch_size,
    )


if __name__ == "__main__":
    main()
