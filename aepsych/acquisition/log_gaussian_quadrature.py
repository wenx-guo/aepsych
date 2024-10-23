# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import math
import torch
from torch.nn import Module

from gpytorch import settings
from gpytorch.utils.quadrature import _pad_with_singletons
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D
from botorch.utils.safe_math import logsumexp


class LogGaussHermiteQuadrature1D(GaussHermiteQuadrature1D):
    """the log version of the Gauss-Hermite quadrature for one-dimensional integration.
    It calculates the logarithm of the integrated probability using the Gauss-Hermite quadrature rule
    """

    def forward(self, func, gaussian_dists):
        """
        Runs Gauss-Hermite quadrature on the callable func, integrating against the Gaussian distributions specified
        by gaussian_dists.

        Args:
            - func (callable): Function to integrate
            - gaussian_dists (Distribution): Either a MultivariateNormal whose covariance is assumed to be diagonal
                or a :obj:`torch.distributions.Normal`.
        Returns:
            - Result of integrating func against each univariate Gaussian in gaussian_dists.
        """
        means = gaussian_dists.mean
        variances = gaussian_dists.variance

        locations = _pad_with_singletons(
            self.locations, num_singletons_before=0, num_singletons_after=means.dim()
        )

        shifted_locs = torch.sqrt(2.0 * variances) * locations + means
        log_probs = func(shifted_locs)
        weights = _pad_with_singletons(
            self.weights,
            num_singletons_before=0,
            num_singletons_after=log_probs.dim() - 1,
        )

        log_res = (
            torch.log(torch.tensor(1 / math.sqrt(math.pi), dtype=torch.float64))
            + log_probs
            + torch.log(weights)
        )
        log_res = logsumexp(log_res, dim=tuple(range(self.locations.dim())))

        return log_res
