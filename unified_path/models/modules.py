import torch
import torch.nn as nn

from torch.nn import Module, Conv2d, Parameter


class Join(Module):
    """Split layer; cf Glow figure 2 / RealNVP figure 4b
    Based on RealNVP multi-scale architecture: splits an input in half along the channel dim; half the vars are
    directly modeled as Gaussians while the other half undergo further transformations (cf RealNVP figure 4b).
    """

    def __init__(self, n_channels):
        super().__init__()
        self.gaussianize = Gaussianize(n_channels // 2)

    def forward(self, x1, z2):
        x2, logdet = self.gaussianize(z2, x1)
        x = torch.cat([x1, x2], dim=1)  # cat along channel dim
        return x, logdet

    def inverse(self, x):
        x1, x2 = x.chunk(2, dim=1)  # split input along channel dim
        z2, logdet = self.gaussianize.inverse(x1, x2)
        return x1, z2, logdet


class Gaussianize(Module):
    """Gaussianization per RealNVP sec 3.6 / fig 4b -- at each step half the variables are directly modeled as Gaussians.
    Model as Gaussians:
        x2 = z2 * exp(logs) + mu, so x2 ~ N(mu, exp(logs)^2) where mu, logs = f(x1)
    Here f(x1) is a conv layer initialized to identity.
    """

    def __init__(self, n_channels):
        super().__init__()
        # computes the parameters of Gaussian
        self.net = Conv2d(
            n_channels,
            2 * n_channels,
            kernel_size=3,
            bias=False,
            padding=1,
            padding_mode="circular",
        )
        #  learned scale (cf RealNVP sec 4.1 / Glow official code
        self.log_scale_factor = Parameter(torch.zeros(2 * n_channels, 1, 1))

        # initialize to identity
        self.net.weight.data.zero_()

    def forward(self, z2, x1):
        h = self.net(x1) * self.log_scale_factor.exp()
        m, logs = h[:, 0::2, :, :], h[:, 1::2, :, :]
        x2 = m + z2 * torch.exp(logs)
        logdet = logs.sum([1, 2, 3])
        return x2, logdet

    def inverse(self, x1, x2):
        h = (
            self.net(x1) * self.log_scale_factor.exp()
        )  # use x1 to model x2 as Gaussians; learnable scale
        m, logs = h[:, 0::2, :, :], h[:, 1::2, :, :]  # split along channel dims
        z2 = (x2 - m) * torch.exp(
            -logs
        )  # center and scale; log prob is computed at the model forward
        logdet = -logs.sum([1, 2, 3])
        return z2, logdet


class Channelize(torch.nn.Module):
    def forward(self, input):
        b, c, h, w = input.shape
        if h % 2 or w % 2:
            raise TypeError("Expected even spatial dims, got {}x{}".format(h, w))

        out = input.reshape(b, c, h // 2, 2, w // 2, 2)
        out = out.permute(0, 1, 3, 5, 2, 4)
        out = out.reshape(b, c * 4, h // 2, w // 2)

        return out

    def logdet(self, input):
        return 0.0


class DeChannelize(torch.nn.Module):
    def forward(self, input):
        b, c, h, w = input.shape
        if c % 4:
            raise TypeError(
                "Expected number of channels dividible by 4, got {}".format(c)
            )

        out = input.reshape(b, c // 4, 2, 2, h, w)
        out = out.permute(0, 1, 4, 2, 5, 3)
        out = out.reshape(b, c // 4, h * 2, w * 2)

        return out

    def logdet(self, input):
        return 0.0


class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.reshape(input.shape[0], -1)

    def logdet(self, input):
        return 0.0


class Unsqueeze(torch.nn.Module):
    def __init__(self, axis=1):
        self.axis = axis
        super().__init__()

    def forward(self, input):
        if isinstance(self.axis, int):
            return input.unsqueeze(self.axis)
        elif isinstance(self.axis, tuple):
            for i in self.axis:
                input = input.unsqueeze(i)
            return input
        else:
            raise TypeError("Invalid axis type.")

    def reverse(self, input):
        if isinstance(self.axis, int):
            return input.squeeze(self.axis)
        elif isinstance(self.axis, tuple):
            for i in self.axis:
                input = input.squeeze(i)
            return input
        else:
            raise TypeError("Invalid axis type.")

    def logdet(self, input):
        return 0.0


class Squeeze(Unsqueeze):
    def forward(self, input):
        return super().reverse(input)

    def reverse(self, input):
        return super().forward(input)


class Reshape(torch.nn.Module):
    def __init__(self, shape):
        self.shape = shape
        super().__init__()

    def forward(self, input):
        return input.view(tuple(input.shape[:1]) + tuple(self.shape))

    def logdet(self, input):
        return 0.0


class Tanhgrow(torch.nn.Module):
    def __init__(self, scale=2.0):
        super().__init__()
        self.scale = scale

    def forward(self, input):
        return input + self.scale * torch.tanh(input)

    def logdet(self, input):
        return (
            (1 + self.scale / torch.cosh(input) ** 2)
            .abs()
            .log()
            .sum(dim=tuple(range(1, len(input.shape))))
        )


class ConvNet(nn.Sequential):
    def __init__(
        self, hidden_sizes, kernel_size, in_channels, out_channels, circular=True
    ):
        assert kernel_size % 2 == 1, "kernel size must be odd for PyTorch >= 1.5.0"
        pad = kernel_size // 2
        pad_mode = "circular" if circular else "zeros"

        super().__init__(
            *[
                nn.Conv2d(
                    in_channels,
                    hidden_sizes[0],
                    kernel_size,
                    padding=pad,
                    padding_mode=pad_mode,
                ),
                nn.LeakyReLU(),
                nn.Sequential(
                    *[
                        nn.Sequential(
                            nn.Conv2d(
                                hidden_sizes[i],
                                hidden_sizes[i + 1],
                                kernel_size,
                                padding=pad,
                                padding_mode=pad_mode,
                            ),
                            nn.LeakyReLU(),
                        )
                        for i in range(len(hidden_sizes) - 1)
                    ]
                ),
                nn.Conv2d(
                    hidden_sizes[-1],
                    out_channels,
                    kernel_size,
                    padding=pad,
                    padding_mode=pad_mode,
                ),
            ]
        )
