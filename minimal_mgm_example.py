#!/usr/bin/env python
from argparse import Namespace

import click
import torch

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from unified_path.action import MGM
from unified_path.importsamp import estimate_ess_q, estimate_ess_p
from unified_path.loss import load_loss
from unified_path.models import RealNVP, RealNVP_Path, COUPLINGS


def infinite_sampling(dataloader):
    while True:
        yield from iter(dataloader)


def load_flow(cfg):
    model = RealNVP if not "fastPath" in cfg.gradient_estimator else RealNVP_Path
    flow = model(
        lat_shape=[cfg.dim],
        coupling_factory=COUPLINGS["NormAltFCS"],
        ncouplings=cfg.n_coupling_layers,
        nblocks=cfg.n_blocks,
        n_hidden=cfg.hidden,
        loss=cfg.gradient_estimator,
    )
    return flow


def train(
    flow,
    X_train,
    X_test,
    kl,
    steps=1000,
    batch_size=500,
    v=True,
):
    optim = torch.optim.Adam(flow.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=steps, eta_min=1e-6
    )
    dataloader = infinite_sampling(
        DataLoader(X_train, batch_size=batch_size, shuffle=True, drop_last=False)
    )
    fw_ESS_dataloader = DataLoader(
        X_test, batch_size=batch_size, shuffle=True, drop_last=False
    )
    tr_l, te_l = [], []
    test_nll = torch.Tensor([0.0])
    train_nll = torch.Tensor([0.0])
    ess = -1.0
    forward_ess = -1.0
    pbar = range(steps)
    if v:
        pbar = tqdm(pbar)
    for i in pbar:
        batch = next(dataloader)
        flow.zero_grad()
        loss, logq, action = kl(batch)

        if loss is not None:
            loss.backward()
        else:
            loss = torch.Tensor([0.0])
        optim.step()
        scheduler.step()
        if v and (i % 10) == 0 or i == steps:
            with torch.no_grad():
                train_nll = -flow.log_prob(batch).mean()
                test_nll = -flow.log_prob(X_test).mean()  # flow probability log q_θ(x)

            ess = estimate_ess_q(
                flow, kl.action, lat_shape=batch.shape[1:], batch_size=1_000, n_iter=5
            )
            forward_ess = estimate_ess_p(
                model=flow,
                config_sampler=fw_ESS_dataloader,
                action=kl.action,
                lat_shape=batch.shape[1:],
                device=loss.device,
            )
            pbar.set_description(
                f"loss: {loss.item():.3e} || NLL train {train_nll.item():.3e} test {test_nll.item():.3e} || ESS q:{ess:.3f} p:{forward_ess:.3f}"
            )

        tr_l.append(loss.item())
        te_l.append(test_nll.item())

    return tr_l, te_l


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
@click.option("--verbose", default=True, help="Verbose")
@click.option("--dim", default=6, type=int, help="dim of data")
def main(**cfg):
    """Function to get all CLI arguments into script."""

    cfg = Namespace(**cfg)
    cfg.lat_shape = [cfg.dim]
    print(cfg)

    torch.manual_seed(cfg.seed)
    print(f"Setting up action and sampling ...")
    num_test_samples = 1_000

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    norm_action = MGM(
        dim=cfg.dim,
        device=device,
        clusters=2,
        cluster_loc=cfg.mgm_loc,
        cluster_scale=cfg.mgm_scale,
    ).to(device)
    X_train = norm_action.sample((cfg.nsamples,)).to(device)
    X_test = norm_action.sample((num_test_samples,)).to(device)

    flow = load_flow(cfg).to(device)

    config_sampler = infinite_sampling(
        DataLoader(X_train, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    )

    loss = load_loss(cfg, flow, config_sampler, norm_action, device)

    train(
        flow=flow,
        X_train=X_train,
        X_test=X_test,
        kl=loss,
        steps=cfg.steps,
        batch_size=cfg.batch_size,
    )


if __name__ == "__main__":
    main()
