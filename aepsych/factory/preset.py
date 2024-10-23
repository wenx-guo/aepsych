#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from configparser import NoOptionError
from typing import List, Optional, Tuple

import gpytorch
import torch
from aepsych.config import Config

from scipy.stats import norm

from .default import _get_default_mean_function, _get_default_cov_function


def preset_mean_covar_factory(
    config: Optional[Config] = None,
    dim: Optional[int] = None,
    stimuli_per_trial: int = 1,
) -> Tuple[gpytorch.means.ConstantMean, gpytorch.kernels.ScaleKernel]:
    """Default factory for generic GP models

    Args:
        config (Config, optional): Object containing bounds (and potentially other
            config details).
        dim (int, optional): Dimensionality of the parameter space. Must be provided
            if config is None.

    Returns:
        Tuple[gpytorch.means.Mean, gpytorch.kernels.Kernel]: Instantiated
            ConstantMean and ScaleKernel with priors based on bounds.
    """

    assert (config is not None) or (
        dim is not None
    ), "Either config or dim must be provided!"

    assert stimuli_per_trial in (1, 2), "stimuli_per_trial must be 1 or 2!"

    mean = _get_default_mean_function(config)

    if config is not None:
        lb = config.gettensor("default_mean_covar_factory", "lb")
        ub = config.gettensor("default_mean_covar_factory", "ub")
        assert lb.shape[0] == ub.shape[0], "bounds shape mismatch!"
        config_dim: int = lb.shape[0]

        if dim is not None:
            assert dim == config_dim, "Provided config does not match provided dim!"
        else:
            dim = config_dim

    # covar = _get_default_cov_function(config, dim, stimuli_per_trial)  # type: ignore
    fixed_kernel_amplitude = config.getboolean(
        "preset_mean_covar_factory",
        "fixed_kernel_amplitude",
        fallback=True,
    )

    if fixed_kernel_amplitude:
        covar = gpytorch.kernels.RBFKernel(
            ard_num_dims=dim, active_dims=None
        )
        preset_lengthscale = config.getlist("preset_mean_covar_factory", "lengthscale")
        covar.lengthscale = torch.tensor(preset_lengthscale).reshape(1, -1)
        covar.raw_lengthscale.requires_grad = False
    else:
        raise NotImplementedError("Only fixed kernel amplitude is supported for now")

    return mean, covar
