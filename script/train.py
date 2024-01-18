import json
import random
import numpy as np
import logging
import os
from argparse import Namespace
from datetime import datetime

import click
import wandb
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from unified_path.models.realNVP import COUPLINGS
from unified_path.training import (
    create_sampler_from_config,
    load_action,
    load_loss,
    load_scheduler,
    train,
)
from unified_path.utils import args_to_str


def geometry(string):
    if isinstance(string, int):
        return [string]
    if isinstance(string, list):
        return string
    return [int(obj) for obj in string.split("x")]


@click.command()
@click.option("--cuda/--cpu", help="runs the code on GPU if available")
@click.option("--last-step", type=int, default=0, help="last training step")
@click.option(
    "--debug-step",
    type=int,
    default=None,
    help="If you know it crashes here, we will save the step and the culprit batch",
)
@click.option(
    "--lat-shape", type=geometry, default="8x8", help="lattice dimensions as TxL"
)
@click.option(
    "--save-interval", type=int, default=100000, help="save every interval steps"
)
@click.option("--kappa", type=float, default=0.21, help="kappa of phi4")
@click.option("--lam", type=float, default=1.0, help="lambda of phi4")
@click.option("--ncouplings", type=int, default=6, help="number of coupling layers")
@click.option("--nblocks", type=int, default=4, help="number of coupling layer blocks")
@click.option("--nmixtures", type=int, default=4, help="number of mixtures for u1flow")
@click.option(
    "--hidden_width", type=int, default=1000, help="width of the hidden coupling layers"
)
@click.option(
    "--coupling",
    type=click.Choice(list(COUPLINGS)),
    default="fc",
    help="coupling layer for nice",
)
@click.option("--n-steps", type=int, default=1000000, help="number of iterations.")
@click.option("--batch-size", type=int, default=256, help="number of samples")
@click.option("--optim", type=click.Choice(["ADAM"]), default="ADAM", help="Optimizer")
@click.option("--lr", type=float, default=5e-4, help="initial learning rate.")
@click.option(
    "--lr-schedule",
    type=click.Choice(["cosine", "plateau", "exp", "cosineWR"]),
    default="plateau",
    help="LR scheduler",
)
@click.option("--ess-interval", type=int, default=1000, help="measures ess at interval")
@click.option(
    "--min-lr", type=float, default=1e-7, help="minimum value for learning rate."
)
@click.option(
    "--patience", type=int, default=3000, help="set patience for lr scheduler."
)
@click.option(
    "--sampler",
    type=click.Choice(["RealNVP", "u1flow", "u1flow-path", "RealNVP-path"]),
    default="RealNVP",
)
@click.option(
    "--activation",
    type=click.Choice(["tanh", "ReLU", "Softplus"]),
    default="tanh",
    help="activation function for RealNVP",
)
@click.option(
    "--base-dist",
    type=click.Choice(["Gaussian", "Uniform"]),
    default="Gaussian",
    help="Which base distribution?",
)
@click.option("--action", type=click.Choice(["phi4", "u1"]), default="phi4")
@click.option(
    "--loss",
    type=click.Choice(
        [
            "RepQP",
            "MaxLik",
            "STLQP",
            "fastpathPQ",
            "fastpathQP",
        ]
    ),
    default="RepQP",
)
@click.option("--beta", type=float, default=1.0, help="inverse coupling")
@click.option("--outdir", type=str, help="rename default output directory.")
@click.option("--output-path", type=click.Path(writable=True), default="runs")
@click.option("--seed", type=int, default=0)
@click.option(
    "--chkp-path", type=click.Path(dir_okay=False), help="full path to checkpoint"
)
@click.option(
    "--load-path-from-standard",
    type=bool,
    default=False,
    help="Use this if you want to load a pathflow model from a standard model",
)
def main(**kwargs):
    args = Namespace(**kwargs)

    if args.outdir is not None:
        out_dir = args.outdir
    else:
        out_dir = args_to_str(args)[:100]
    out_dir += str(datetime.now())[5:].split(".")[0]
    out_dir = os.path.join(args.output_path, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logging.info(vars(args))

    writer = SummaryWriter(out_dir)

    fname_args = os.path.join(out_dir, "args.json")
    with open(fname_args, "w") as fd:
        json.dump(vars(args), fd, indent=2)

    device = torch.device("cuda" if args.cuda else "cpu")
    # Loads sampler weights
    sampler = create_sampler_from_config(args.__dict__)
    sampler.to(device)
    if args.optim == "ADAM":
        optimizer = torch.optim.Adam(sampler.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(sampler.parameters(), lr=args.lr)

    if args.chkp_path is not None:
        state = torch.load(args.chkp_path)
        optimizer.load_state_dict(state["optim"])
        torch.set_rng_state(state["rng"])
    else:
        seed = args.seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    action = load_action(args, device)
    kl_loss = load_loss(args, sampler, action)

    scheduler = load_scheduler(args, optimizer)

    fn = train(kl_loss, optimizer, sampler.parameters, scheduler, writer, args, out_dir)
    if fn:
        logging.info(f"Saved under {fn}")


if __name__ == "__main__":
    main()
