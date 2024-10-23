#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from aepsych.acquisition.objective import ProbitObjective
from botorch.acquisition.monte_carlo import (
    AcquisitionFunction,
    MCAcquisitionFunction,
    MCAcquisitionObjective,
    MCSampler,
)
from botorch.acquisition.objective import PosteriorTransform

from botorch.models.model import Model
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform


class RTMeanAwareAcquisition(MCAcquisitionFunction):

    def __init__(
        self,
        model: Model,
        base_acqf_cls: MCAcquisitionFunction,
        objective: Optional[MCAcquisitionObjective] = None,
        sampler: Optional[MCSampler] = None,
        **kwargs,
    ) -> None:

        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
        if objective is None:
            objective = ProbitObjective()

        super().__init__(model=model, sampler=sampler, objective=None, X_pending=None)
        self.base_acqf = base_acqf_cls(
            model=model, sampler=sampler, objective=objective, **kwargs
        )

    @t_batch_mode_transform()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate the acquisition function

        Args:
            X (torch.Tensor): Points at which to evaluate.

        Returns:
            torch.Tensor: Value of the acquisition functiona at these points.
        """
        acqval = self.base_acqf.forward(X)  # the acquisition value based on f
        # now compute mean RT
        post = self.model.posterior(X)
        samples = self.sampler(post)  # same samples that the `forward` call uses
        mean_rt = self.model.likelihood(samples).mean.squeeze().mean(0)
        return acqval / mean_rt


class RTMeanAwareAnalyticAcquisition(AcquisitionFunction):

    def __init__(
        self,
        model: Model,
        base_acqf_cls: AcquisitionFunction,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs,
    ) -> None:

        super().__init__(model=model)
        self.base_acqf = base_acqf_cls(
            model=model, posterior_transform=posterior_transform, **kwargs
        )

    @t_batch_mode_transform()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate the acquisition function

        Args:
            X (torch.Tensor): Points at which to evaluate.

        Returns:
            torch.Tensor: Value of the acquisition functiona at these points.
        """
        acqval = self.base_acqf.forward(X)  # the acquisition value based on f
        # now compute mean RT
        post = self.model.posterior(X)
        mean_rt = self.model.likelihood(post.mean).mean.squeeze()
        return acqval / mean_rt
