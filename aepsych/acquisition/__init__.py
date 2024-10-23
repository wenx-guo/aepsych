#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys

from .rt_aware import RTMeanAwareAcquisition

from ..config import Config
from .mutual_information import (
    BernoulliMCMutualInformation,
    MonotonicBernoulliMCMutualInformation,
)
from .mc_posterior_variance import (
    MCPosteriorVariance,
    MonotonicMCPosteriorVariance,
)
from .objective import (
    FloorGumbelObjective,
    FloorLogitObjective,
    FloorProbitObjective,
    ProbitObjective,
    semi_p,
)

from ..config import Config
from .lookahead import (
    ApproxGlobalSUR,
    EAVC,
    GlobalMI,
    GlobalSUR,
    LocalMI,
    LocalSUR,
    LogGlobalMI,
    LogGlobalSUR,
    MOCU,
    SMOCU,
    CoreMSE,
    LogCoreMSE
)
from .rt_aware import RTMeanAwareAcquisition

lse_acqfs = [
    GlobalMI,
    GlobalSUR,
    ApproxGlobalSUR,
    EAVC,
    LocalMI,
    LocalSUR,
    LogGlobalMI,
    LogGlobalSUR,
]
__all__ = [
    "RTMeanAwareAcquisition",
    "BernoulliMCMutualInformation",
    "MonotonicBernoulliMCMutualInformation",
    "MCPosteriorVariance",
    "MonotonicMCPosteriorVariance",
    "MCPosteriorVariance",
    "ProbitObjective",
    "FloorProbitObjective",
    "FloorLogitObjective",
    "FloorGumbelObjective",
    "SMOCU",
    "MOCU",
    "GlobalMI",
    "GlobalSUR",
    "ApproxGlobalSUR",
    "EAVC",
    "LocalMI",
    "LocalSUR",
    "semi_p",
    "LogGlobalMI",
    "LogGlobalSUR",
    "CoreMSE",
    "LogCoreMSE",
]

Config.register_module(sys.modules[__name__])
