import torch


class Loss(torch.nn.Module):
    def __init__(self, model, action, lat_shape, batch_size):
        super().__init__()
        self.sampler = model
        self.action = action
        self.lat_shape = lat_shape
        self.batch_size = batch_size


class RepQP(Loss):
    r"""Calculates the loss.

    .. math::

        L = 1/N \sum_{i=1}^N ( S(g(z_i)) - log det dg/dz(z_i) + q_Z(z_i) )

    where :math:`z_i \sim p_Z`.
    Minimizing this loss is equivalent to minimizing KL(q, p) from the
    variational distribution q to the target Boltzmann distribution p=1/Z exp(-S).
    """

    def forward(self, eps=None):
        if not hasattr(self.sampler, "sample_with_logprob"):
            raise TypeError("Model does not support logprob sampling")

        if eps is None:
            samples, log_probs = self.sampler.sample_with_logprob(self.batch_size)
        else:
            samples, log_probs = self.sampler.forward(eps)

        actions = self.action(samples.reshape(self.batch_size, *self.lat_shape))
        neg_lwt = actions + log_probs

        loss_mean = neg_lwt.mean()

        return loss_mean, -neg_lwt, actions

    def __str__(self):
        return "Reparam KL(q|p)"


class DropInPathQP(Loss):
    def forward(self, eps=None):
        with torch.no_grad():
            if eps is None:
                eps = self.sampler.sample_base(self.batch_size)
            x_prime, _ = self.sampler.forward(eps)

        x_prime = x_prime.clone().detach().requires_grad_()
        log_probs = self.sampler.log_prob(x_prime)

        actions = self.action(x_prime)
        unnorm_target = -1.0 * actions
        log_w_tilde = unnorm_target - log_probs

        grad = torch.autograd.grad(outputs=-log_w_tilde.mean(), inputs=x_prime)[0]

        x, _ = self.sampler.g(eps)
        grad_term = (x * grad).sum()
        loss_mean = -log_w_tilde.detach().mean() + grad_term - grad_term.detach()

        return loss_mean, log_w_tilde, actions

    def __str__(self):
        return "STL KL(q|p)"


class MaximumLikelihood(torch.nn.Module):
    r"""Calculates the loss.

    .. math::

        L = - 1/N \sum_{i=1}^N (log det dg/dz(g^{-1}(\phi_i)) + log[q_Z(g^{-1}(\phi_i))] )

    where :math:`\phi_i \sim p`.
    Minimizing this loss is equivalent to minimizing KL(p||q) wrt params :math:`\lambda` of :math:`q_\lambda`
    to the target Boltzmann distribution p=1/Z exp(-S).
    """

    def __init__(self, model, config_sampler, action, lat_shape, device=False):
        super().__init__()
        self.model = model
        self.action = action
        self.config_sampler = config_sampler
        self.lat_shape = lat_shape
        self.device = device

    def forward(self, samples=None):
        if not hasattr(self.model, "log_prob"):
            raise TypeError("Model does not support log prob estimation")
        if samples is None:
            samples = next(self.config_sampler)[0].to(self.device)
        # The action only appears here for comparing the loss
        actions = self.action(samples)
        loss = -self.model.log_prob(samples)
        loss_mean = loss.mean(0)

        return (loss - actions.detach()).mean(), loss_mean, actions


class FastDropInPathPQ(MaximumLikelihood):
    def forward(self, samples=None):
        if not hasattr(self.model, "log_prob"):
            raise TypeError("Model does not support log prob estimation")
        if samples is None:
            samples = next(self.config_sampler)[0].to(self.device)
        x1 = samples.detach().clone().requires_grad_()

        z1, logq = self.model.f(x1)
        actions = self.action(x1)
        logp_t = -actions
        nlwt = logq - logp_t
        grad = torch.autograd.grad(nlwt.mean(), x1)[0]

        z1 = z1.detach().clone().requires_grad_()

        x_, _ = self.model(z1)
        gradterm = (x_ * grad.detach()).reshape((samples.shape[0], -1)).sum()
        return (
            (-logq.detach() - actions.detach()).mean() + gradterm - gradterm.detach(),
            -logq,
            actions,
        )


class FastPath(Loss):
    def __init__(self, model, action, lat_shape, batch_size, kind, config_sampler=None):
        super().__init__(model, action, lat_shape, batch_size)
        assert kind.lower() in (
            "pq",
            "qp",
        ), f"Unknown kind {kind} of path gradient loss"
        self.qp = "qp" == kind.lower()
        if self.qp:
            self.alternative = DropInPathQP(model, action, lat_shape, batch_size)
        else:
            self.config_sampler = config_sampler
            assert config_sampler is not None, "Need config_sampler for PQ"
            self.alternative = FastDropInPathPQ(
                model, config_sampler, action, lat_shape, batch_size
            )

    def forward(self, samples=None):
        if not self.training or not torch.is_grad_enabled():
            return self.alternative()

        if not hasattr(self.sampler, "piggy_back_forward"):
            raise TypeError(
                "Model doesn't support piggy_back_forward, probably not a path gradient flow"
            )

        # If samples are not given, sample from given initial distribution
        if samples is None:
            if self.qp:
                samples = self.sampler.sample_base(self.batch_size)
            else:
                samples = next(self.config_sampler)[0].to(self.device)

        # Gather initial values, including the force of the initial distribution
        if self.qp:
            x_ = torch.autograd.Variable(samples, requires_grad=True)
            lq0x = self.sampler._prior_log_prob(samples)
            initial_px = self.sampler._prior_log_prob(x_)
            sign = -1
        else:
            x_ = torch.autograd.Variable(samples, requires_grad=True)
            actions = self.action(samples)
            initial_px = -self.action(x_)
            sign = +1

        dlogqdx = torch.autograd.grad(
            initial_px.sum(), x_, allow_unused=True, retain_graph=True
        )[0]

        # Perform piggy back forward, to get \partial log rho_\theta(x)/ \partial x
        x, delta_trans, dlogqdx = self.sampler.piggy_back_forward(
            samples, dlogqdx, reverse=not self.qp
        )

        # Get final values, including the force of the final distribution
        if self.qp:
            actions = self.action(x.reshape(-1, *self.lat_shape))
            final_px = -actions
        else:
            lq0x = self.sampler._prior_log_prob(x)
            final_px = -lq0x
        log_q = lq0x - sign * delta_trans
        dfinalpdx = torch.autograd.grad(
            sign * final_px.sum(), x, allow_unused=True, retain_graph=True
        )[0]

        # Amalgamate the gradients
        dLdx = dlogqdx + dfinalpdx
        log_weights = -(actions + log_q).detach()
        dLdx = dLdx / self.batch_size
        loss_mean = sign * log_weights.mean()

        # Add the gradient-yielding term
        grad_term = (x * dLdx.detach()).sum()
        loss_mean += grad_term - grad_term.detach()

        return (
            loss_mean,
            (log_weights).detach().cpu(),
            actions.detach().cpu(),
        )

    def __str__(self):
        return f"FastPath-{'QP' if self.qp else 'PQ'}"
