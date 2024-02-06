### Fast and Unified Path Gradients

This is the official, simple and fast implementation of path gradients for both forward and reverse KL's from the paper "Fast and Unified Path Gradients for Shading and Lighting" by Lorenz Vaitl, Ludwig Winkler, Lorenz Richter and Pan Kessel.

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

`Example.ipynb` contains a simple example of how to use the path gradients in practice. The code is written in PyTorch and shows the improved convergence of path gradients compared to standard estimators including the score in a 2d Gaussian toy example.