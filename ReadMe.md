### Fast and Unified Path Gradient Estimators

This is the official, simple and fast implementation of path gradients for both forward and reverse KL's from the paper "Fast and Unified Path Gradient Estimators" by Lorenz Vaitl, Ludwig Winkler, Lorenz Richter and Pan Kessel.

> Lorenz Vaitl*, Ludwig Winkler*, Lorenz Richter, Pan Kessel. "Fast and Unified Path Gradient Estimators for Normalizing
Flows" _International Conference on Learning Representations (2024).
> [[arxiv]](https://openreview.net/pdf?id=zlkXLb3wpF) [[bibtex]]()

### Abstract

Recent work shows that path gradient estimators for normalizing flows have lower
variance compared to standard estimators for variational inference, resulting in
improved training. However, they are often prohibitively more expensive from a
computational point of view and cannot be applied to maximum likelihood train-
ing in a scalable manner, which severely hinders their widespread adoption. In
this work, we overcome these crucial limitations. Specifically, we propose a fast
path gradient estimator which improves computational efficiency significantly and
works for all normalizing flow architectures of practical relevance. We then show
that this estimator can also be applied to maximum likelihood training for which
it has a regularizing effect as it can take the form of a given target energy func-
tion into account. We empirically establish its superior performance and reduced
variance for several natural sciences applications.

### Code

#### Examples

`Example.ipynb` contains a simple example of how to use the path gradients in practice. The code is written in PyTorch and shows the improved convergence of path gradients compared to standard estimators including the score in a 2d Gaussian toy example with intuitive visualizations.

`script/mgm_train.py` lets you train your own model with path gradients on a Multivariate Gaussian Mixture (MGM) model.

`script/u1_train.py` lets you train your own flow with path gradients on a U1 lattice field problem.

#### Gradient Estimators

We provide the following gradient estimators:

- `RepQP`: standard reverse KL estimator without path gradient
- `ML`: standard forward KL estimator without path gradient
- `fastPathQP`: fast path gradient estimator for reverse KL with force evaluation
- `fastPathPQ`: fast path gradient estimator for forward KL with force evaluation
- `DropInQP`: slow, plug and play path gradient reverse for reverse KL with two evaluations
- `FastDropInPQ`: plug and play path gradient for forward KL


The `fastpath` family of estimators utilize the fast path gradient estimator from the paper and exploit redundant and recursive operations to obtain the benefits of path gradients.