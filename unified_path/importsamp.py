import torch
from unified_path.utils import estimate_importance_weights


def estimate_reverse_ess(model, action, lat_shape, batch_size, n_iter):
    """Reverse ESS.
    ESS_q  = N / (sum_i w_i^2)

    :param model: Normalizing flow model
    :param action: Target action
    :param lat_shape: list of lattice dimensions
    """
    with torch.no_grad():
        log_weight_hist = []

        for _ in range(n_iter):
            # sample from model and calc probs and action diffs
            samples, log_prob = model.sample_with_logprob(batch_size)
            # samples = samples.cpu()
            # log_prob = log_prob.cpu()
            true_exp = (-1.0) * action(samples.reshape(batch_size, *lat_shape))

            lwt = true_exp - log_prob
            # append history
            log_weight_hist.append(lwt)

        log_weight_hist = torch.cat(log_weight_hist)

        weight_hist = estimate_importance_weights(log_weight_hist)
        n = weight_hist.shape[0]
        w = n * weight_hist

        ess = 1 / (w**2).mean()

    return ess.item()


def estimate_forward_ess(model, config_sampler, action, lat_shape, device="cpu"):
    """Forward ESS as used in https://arxiv.org/pdf/2107.00734.pdf Eq 31.

    ESS_p = N / ((sum_i w_i) * (sum_i 1/w_i))

    :param model: Normalizing flow model
    :param config_sampler: Sampler, which samples from the target distribution
    :param action: Target action
    :param lat_shape: list of lattice dimensions
    :param device:
    """
    with torch.no_grad():
        log_weight_hist = []
        for samples in config_sampler:
            # sample from model and calc probs and action diffs
            samples = samples.to(device)
            log_prob = model.log_prob(samples)
            true_exp = -action(samples.reshape(samples.shape[0], *lat_shape))

            log_weight_hist.append(true_exp - log_prob)

        log_weight_hist = torch.cat(log_weight_hist, dim=0)

        max_log_uw, _ = log_weight_hist.max(-1)
        w = torch.exp(log_weight_hist - max_log_uw)
        inv_w = 1 / w

    ess = 1 / w.mean() / inv_w.mean()

    return ess.item()
