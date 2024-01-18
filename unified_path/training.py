import json
import os
from argparse import Namespace

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from tqdm import tqdm
import time

from unified_path.action import U1GaugeAction
from unified_path.importsamp import estimate_ess_q
from unified_path.loss import (
    DropInPathQP,
    Loss,
    RepQP,
    FastPath,
    FastDropInPath,
)
from unified_path.models import (
    COUPLINGS,
    RealNVP,
    RealNVP_Path,
    U1Flow,
    U1Flow_Path,
)
from unified_path.utils import CosineAnnealingWithRestartsLR, calc_imp_weights


def load_model_action_args(fn, epoch, device, take_last=False, optim=False):
    kwargs = json.load(open(os.path.join(fn, "args.json")))

    if take_last:
        epoch = range(0, kwargs["n_steps"], kwargs["save_interval"])[-1]
    sampler = create_sampler_from_config(kwargs)
    if epoch >= 0:
        pth = torch.load(
            os.path.join(fn, f"checkpoint_{epoch}.pth"),
            map_location=torch.device(device),
        )
        sampler.load_state_dict(pth["net"])
        if hasattr(sampler, "base_dist"):
            sampler.dist_params.requires_grad_(False)
    sampler.to(device)
    args = Namespace(**kwargs)
    action = load_action(args, device)
    if not optim:
        return sampler, action, args
    if args.optim == "ADAM":
        optimizer = torch.optim.Adam(sampler.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(sampler.parameters(), lr=args.lr)
    if epoch >= 0:
        optimizer.load_state_dict(pth["optim"])
    return sampler, action, args, optimizer


def create_sampler(**kwargs):
    if kwargs["sampler"].lower() in ["realnvp", "realnvp-path"]:
        model = RealNVP if "path" not in kwargs["sampler"].lower() else RealNVP_Path
        coupling_factory = COUPLINGS[kwargs["coupling"]]
        sampler = model(
            lat_shape=kwargs["lat_shape"],
            coupling_factory=COUPLINGS[kwargs["coupling"]],
            ncouplings=kwargs["ncouplings"],
            nblocks=kwargs["nblocks"],
            n_hidden=kwargs["hidden_width"],
            bias=kwargs["bias"],
            activation=kwargs["activation"],
            base_dist=kwargs["base_dist"],
            loss=kwargs["loss"],
        )
    elif kwargs["sampler"].lower() in ["u1flow", "u1flow-path"]:
        model = U1Flow if "path" not in kwargs["sampler"].lower() else U1Flow_Path
        sampler = model(
            lat_shape=kwargs["lat_shape"],
            n_mixture_comps=kwargs["nmixtures"],
            n_layers=kwargs["ncouplings"],
        )
    else:
        raise NotImplementedError(f"{kwargs['sampler']} is not implemented")
    return sampler


def create_sampler_from_config(config):
    sampler = create_sampler(**config)

    state_dict = None
    if config.get("chkp_path", None) is not None:
        state_dict = torch.load(config["chkp_path"])
        if config["load_path_from_standard"]:
            og_state_dict = state_dict["net"]
            path_state_dict = {}
            for key, item in og_state_dict.items():
                if len(key.split(".")) > 1 and key.split(".")[0] == "couplings":
                    newkey = key.split(".")
                    newkey.insert(2, "coupling")
                    path_state_dict[".".join(newkey)] = item
                else:
                    path_state_dict[key] = item

            state_dict["net"] = path_state_dict
        sampler.load_state_dict(state_dict["net"])

    return sampler


def load_action(args, device):
    if args.action == "u1":
        action = U1GaugeAction(beta=args.beta)
    else:
        raise TypeError(f'Action "{args.action}" is not valid')
    return action


def load_loss(args, sampler, action):
    # For U1 we have complex numbers -> 2 values instead of one
    if isinstance(action, U1GaugeAction):
        args.lat_shape = (2, *args.lat_shape)

    elif args.loss == "RepQP":
        kl_loss = RepQP(sampler, action, args.lat_shape, args.batch_size)
    elif args.loss == "FastDropInPath":
        kl_loss = FastDropInPath(
            sampler,
            None,
            action,
            args.lat_shape,
            args.batch_size,
        )
    elif args.loss == "DropinPathQP":
        kl_loss = DropInPathQP(
            sampler,
            action,
            args.lat_shape,
            args.batch_size,
        )
    elif args.loss.lower() in ["fastpathPQ", "fastpathQP"]:
        kl_loss = FastPath(
            sampler,
            action,
            args.lat_shape,
            args.batch_size,
            args.loss.replace("fastpath", ""),
        )
    else:
        raise TypeError(f'Loss "{args.loss}" is not valid')
    return kl_loss


def load_scheduler(args, optimizer):
    scheduler = None
    if args.lr_schedule == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor=0.92,
            verbose=True,
            patience=args.patience,
            threshold=1e-4,
            min_lr=args.min_lr,
        )
    elif args.lr_schedule == "exp":
        scheduler = StepLR(optimizer, 1000, 0.96)
    elif args.lr_schedule == "cosine":
        scheduler = CosineAnnealingLR(optimizer, 1000, eta_min=args.min_lr)
    elif args.lr_schedule == "cosineWR":
        scheduler = CosineAnnealingWithRestartsLR(optimizer, T_max=1000)
    elif args.lr_scheduler is not None:
        raise TypeError(f'LR scheduler "{args.lr_schedule}" is not valid.')
    return scheduler


def train(kl_loss: Loss, optimizer, parameters, scheduler, writer, args, out_dir):
    chkpt_pth = None

    last_loss = None
    with tqdm(range(args.last_step, args.n_steps, 1)) as pbar:
        for step in pbar:
            optimizer.zero_grad()

            loss_mean, log_w_tilde, actions = kl_loss()

            if loss_mean is None:
                step -= 1
                continue
            loss_mean.backward()

            # Track gradient norm
            grad_norm = 0
            for p in parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm**0.5

            optimizer.step()

            if isinstance(scheduler, StepLR):
                scheduler.step()
            elif isinstance(
                scheduler,
                (CosineAnnealingLR, CosineAnnealingWithRestartsLR, ReduceLROnPlateau),
            ):
                scheduler.step(log_w_tilde.std())

            logZ = (
                torch.logsumexp(log_w_tilde, dim=0) - np.log(log_w_tilde.shape[0])
            ).detach()
            imp_weights = calc_imp_weights(log_w_tilde, include_factor_N=False).detach()
            kl_pq = imp_weights * log_w_tilde
            w = imp_weights / log_w_tilde.shape[0]
            stats = {
                "loss": loss_mean,
                "loss_std": loss_mean.std(),
                "ELBO": log_w_tilde.mean(),
                "KL(p|q)": kl_pq.mean() - logZ,
                "KL(p|q)_std": kl_pq.std(),
                "KL(q|p)": logZ - log_w_tilde.mean(),
                "KL(q|p)_std": log_w_tilde.std(),
                "log(Z)": logZ,
                "Z": logZ.exp(),
                "action": actions.mean(),
                "action_std": actions.std(),
                "w_max": w.max(),
                "w_median": w.median(),
                "var w": imp_weights.var(unbiased=False),
                "grad norm": grad_norm,
            }

            stats = {
                key: val.item() if not isinstance(val, float) else val
                for key, val in stats.items()
            }
            if args.ess_interval is not None and step % args.ess_interval == 0:
                (
                    stats["ess"],
                    stats["absmag"],
                    stats["mag"],
                ) = estimate_ess_q(
                    kl_loss.sampler,
                    kl_loss.action,
                    args.lat_shape,
                    batch_size=2 * args.batch_size,
                    n_iter=50,
                )
            stats["lr"] = optimizer.param_groups[0]["lr"]

            pbar.set_description(
                f"step: {step} "
                f'KL(q|p): {stats["KL(q|p)"]:8.2e}+-{stats["KL(q|p)_std"]:4.1e} '
                f'KL(p|q): {stats["KL(p|q)"]:8.2e}+-{stats["KL(p|q)_std"]:4.1e} '
                f'action: {stats["action"]:6.2e}+-{stats["action_std"]:4.1e}'
            )

            for key, value in stats.items():
                writer.add_scalar(key, value, step)

            if step % args.save_interval == 0:
                chkpt_pth = os.path.join(out_dir, f"checkpoint_{step}.pth")
                state = {"optim": optimizer.state_dict()}
                state["net"] = kl_loss.sampler.state_dict()
                state["rng"] = torch.get_rng_state()
                torch.save(state, chkpt_pth)
            last_loss = loss_mean

    return chkpt_pth


def prepare_model(args, device):
    sampler = create_sampler_from_config(args.__dict__)
    sampler.to(device)

    parameters = sampler.parameters
    return sampler, parameters
