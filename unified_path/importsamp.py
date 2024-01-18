import torch
from unified_path.utils import calc_imp_weights


def estimate_ess_q(model, action, lat_shape, batch_size, n_iter):
    """Backward ESS."""
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

        weight_hist = calc_imp_weights(log_weight_hist)
        n = weight_hist.shape[0]
        w = n * weight_hist

        ess = 1 / (w**2).mean()

    return ess.item()




def estimate_ess_p(model, config_sampler, action, lat_shape, device="cpu"):
    """Forward ESS Only difference to estimate_ess_mit() is that the sampler lost the zero index
    select.

    :param model:
    :param config_sampler:
    :param action:
    :param lat_shape:
    :param batch_size:
    :param n_iter:
    :param device:
    :return:
    """
    with torch.no_grad():
        log_weight_hist = []
        weight_hist = []
        for samples in config_sampler:
            # sample from model and calc probs and action diffs
            samples = samples.to(device)  # was next(config_sampler)[0].to(device)
            log_prob = model.log_prob(samples)
            true_exp = -action(samples.reshape(samples.shape[0], *lat_shape))
            # append history
            weight_hist.append(torch.exp(true_exp - log_prob))
            log_weight_hist.append(true_exp - log_prob)

        log_weight_hist = torch.cat(log_weight_hist, dim=0)

        # weight_hist = calc_imp_weights(log_weight_hist)
        max_log_uw, _ = log_weight_hist.max(-1)
        w = torch.exp(log_weight_hist - max_log_uw)
        inv_w = 1 / w

    ess = 1 / w.mean() / inv_w.mean()

    return ess.item()
