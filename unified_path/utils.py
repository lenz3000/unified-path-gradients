import logging
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset


def estimate_importance_weights(log_weight_hist, include_factor_N=True):
    """Calculates normalized importance weights from logarithm of unnormalized weights.

    If include_factor_N is set, then the result will be multiplied by the number of weights, s.t.
    the expectation is simply the sum of weights and history. Otherwise, the average instead of the
    sum has to be taken.
    """
    # use exp-norm trick:
    # https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick
    log_weight_hist_max, _ = log_weight_hist.max(dim=0)
    log_weigth_hist_norm = log_weight_hist - log_weight_hist_max
    weight_hist = torch.exp(log_weigth_hist_norm)
    if include_factor_N:
        weight_hist = weight_hist / weight_hist.sum(dim=0)
    else:
        weight_hist = weight_hist / weight_hist.mean(dim=0)

    return weight_hist


def config_sampler(configs, batchsize, overrelax):
    if overrelax:
        configs = torch.cat([configs, -1.0 * configs], dim=0)
    dataset = TensorDataset(configs)
    loader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=batchsize,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    if len(loader) <= 1:
        raise RuntimeError("Empty loader.")
    while True:
        yield from loader


def get_new_name(loss):
    newnames = {"RepPQ": "MC-DReG-PQ", "DReG": "DReG-MC-PQ", "STLQP": "STLQP"}
    if loss in newnames:
        loss = newnames[loss]
    return loss


def invert_transform_bisect_wgrad(y, *, f, tol, max_iter, a=0, b=2 * np.pi):
    min_x = a * torch.ones_like(y)
    max_x = b * torch.ones_like(y)
    min_val = f(min_x)
    max_val = f(max_x)
    for i in range(max_iter):
        mid_x = (min_x + max_x) / 2
        mid_val = f(mid_x)
        greater_mask = (y > mid_val).float()
        err = torch.max(torch.abs(y - mid_val))
        if err < tol:
            return mid_x
        if torch.all((mid_x == min_x) + (mid_x == max_x)):
            logging.warning(
                "WARNING: Reached floating point precision before tolerance "
                f"(iter {i}, err {err})"
            )
            return mid_x
        min_x = greater_mask * mid_x + (1 - greater_mask) * min_x
        min_val = greater_mask * mid_val + (1 - greater_mask) * min_val
        max_x = (1 - greater_mask) * mid_x + greater_mask * max_x
        max_val = (1 - greater_mask) * mid_val + greater_mask * max_val
    logging.warning(
        f"WARNING: Did not converge to tol {tol} in {max_iter} iters! Error was {err}"
    )
    return mid_x


def invert_transform_bisect(y, *, f, tol, max_iter, a=0, b=2 * np.pi):
    min_x = a * torch.ones_like(y)
    max_x = b * torch.ones_like(y)
    min_val = f(min_x)
    max_val = f(max_x)
    with torch.no_grad():
        for i in range(max_iter):
            mid_x = (min_x + max_x) / 2
            mid_val = f(mid_x)
            greater_mask = (y > mid_val).float()
            err = torch.max(torch.abs(y - mid_val))
            if err < tol:
                return mid_x
            if torch.all((mid_x == min_x) + (mid_x == max_x)):
                logging.warning(
                    "WARNING: Reached floating point precision before tolerance "
                    f"(iter {i}, err {err})"
                )
                return mid_x
            min_x = greater_mask * mid_x + (1 - greater_mask) * min_x
            min_val = greater_mask * mid_val + (1 - greater_mask) * min_val
            max_x = (1 - greater_mask) * mid_x + greater_mask * max_x
            max_val = (1 - greater_mask) * mid_val + greater_mask * max_val
        logging.warning(
            f"WARNING: Did not converge to tol {tol} in {max_iter} iters! Error was {err}"
        )
        return mid_x


def gather_grads(model: torch.nn.Module):
    return torch.concat(
        tuple(p.grad.reshape(-1) for p in model.parameters() if p.grad is not None)
    )
