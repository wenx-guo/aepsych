#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn.functional import softplus
from aepsych.utils import make_scaled_sobol
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.input_constructors import acqf_input_constructor
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.gpytorch import GPyTorchModel
from botorch.utils.multi_objective import is_non_dominated

from botorch.utils.safe_math import (
    fatmaximum,
    logdiffexp,
    logplusexp,
    logsumexp,
)
from botorch.utils.transforms import t_batch_mode_transform
from gpytorch.kernels.rbf_kernel import RBFKernel
from scipy.special import owens_t
from scipy.stats import norm
from torch import Tensor

from .lookahead_utils import (
    approximate_lookahead_levelset_at_xstar,
    log_lookahead_levelset_at_xstar,
    lookahead_levelset_at_xstar,
    log_lookahead_p_at_xstar,
    lookahead_p_at_xstar,
)

torch.set_default_dtype(torch.float64)


def Hb(p: Tensor):
    """
    Binary entropy.

    Args:
        p: Tensor of probabilities.

    Returns: Binary entropy for each probability.
    """
    epsilon = torch.tensor(np.finfo(float).eps)
    p = torch.clamp(p, min=epsilon, max=1 - epsilon)
    return -torch.nan_to_num(p * torch.log2(p) + (1 - p) * torch.log2(1 - p))


def MI_fn(Px: Tensor, P1: Tensor, P0: Tensor, py1: Tensor) -> Tensor:
    """
    Average mutual information.
    H(p) - E_y*[H(p | y*)]

    Args:
        Px: (b x m) Level-set posterior before observation
        P1: (b x m) Level-set posterior given observation of 1
        P0: (b x m) Level-set posterior given observation of 0
        py1: (b x 1) Probability of observing 1

    Returns: (b) tensor of mutual information averaged over Xq.
    """
    mi = Hb(Px) - py1 * Hb(P1) - (1 - py1) * Hb(P0)
    return mi


def ClassErr(p: Tensor) -> Tensor:
    """
    Expected classification error, min(p, 1-p).
    """
    return torch.min(p, 1 - p)


def SUR_fn(Px: Tensor, P1: Tensor, P0: Tensor, py1: Tensor) -> Tensor:
    """
    Stepwise uncertainty reduction.

    Expected reduction in expected classification error given observation at Xstar,
    averaged over Xq.

    Args:
        Px: (b x m) Level-set posterior before observation
        P1: (b x m) Level-set posterior given observation of 1
        P0: (b x m) Level-set posterior given observation of 0
        py1: (b x 1) Probability of observing 1

    Returns: (b) tensor of SUR values.
    """
    sur = ClassErr(Px) - py1 * ClassErr(P1) - (1 - py1) * ClassErr(P0)
    return sur


def EAVC_fn(Px: Tensor, P1: Tensor, P0: Tensor, py1: Tensor) -> Tensor:
    """
    Expected absolute value change.

    Expected absolute change in expected level-set volume given observation at Xstar.

    Args:
        Px: (b x m) Level-set posterior before observation
        P1: (b x m) Level-set posterior given observation of 1
        P0: (b x m) Level-set posterior given observation of 0
        py1: (b x 1) Probability of observing 1

    Returns: (b) tensor of EAVC values.
    """
    avc1 = torch.abs((Px - P1).sum(dim=-1))
    avc0 = torch.abs((Px - P0).sum(dim=-1))
    return py1.squeeze(-1) * avc1 + (1 - py1).squeeze(-1) * avc0


class LookaheadAcquisitionFunction(AcquisitionFunction):
    def __init__(
        self,
        model: GPyTorchModel,
        target: Optional[Union[List[float], float]],
        lookahead_type: str = "levelset",
        log_acqf: bool = False,
    ) -> None:
        """
        A localized look-ahead acquisition function.

        Args:
            model: The gpytorch model.
            target: Threshold value(s) to target in probability space.
        """
        super().__init__(model=model)
        self.lookahead_type = lookahead_type
        self.log_acqf = log_acqf
        if "levelset" in lookahead_type:
            assert target is not None, "Need a target for levelset lookahead!"

            if log_acqf:
                self.lookahead_fn = log_lookahead_levelset_at_xstar
            else:
                self.lookahead_fn = lookahead_levelset_at_xstar

            self.gamma = norm.ppf(target).astype(np.float64)
            # check data types
            if lookahead_type == "levelset":
                assert isinstance(
                    self.gamma, np.float64
                ), "supply a single target threshold for the 'levelset' lookahead type"
                self.gamma = np.array([self.gamma])
            elif lookahead_type in ["multi_levelset", "multi_levelset_pareto_front"]:
                assert self.gamma.size > 1 and isinstance(
                    self.gamma, np.ndarray
                ), "supply a list of target thresholds for the 'multi_levelset' lookahead type"
            self.gamma = torch.from_numpy(self.gamma).unsqueeze(1).unsqueeze(2)

        elif lookahead_type == "posterior":
            if log_acqf:
                self.lookahead_fn = log_lookahead_p_at_xstar
            else:
                self.lookahead_fn = lookahead_p_at_xstar
            self.gamma = None
        else:
            raise RuntimeError(f"Got unknown lookahead type {lookahead_type}!")


## Local look-ahead acquisitions
class LocalLookaheadAcquisitionFunction(LookaheadAcquisitionFunction):
    def __init__(
        self,
        model: GPyTorchModel,
        lookahead_type: str = "levelset",
        target: Optional[Union[List[float], float]] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
    ) -> None:
        """
        A localized look-ahead acquisition function.

        Args:
            model: The gpytorch model.
            target: Threshold value to target in p-space.
        """

        super().__init__(model=model, target=target, lookahead_type=lookahead_type)
        self.posterior_transform = posterior_transform

    @t_batch_mode_transform(assert_output_shape=True)
    def forward(self, X: Tensor) -> Tensor:
        """
        Evaluate acquisition function at X.

        Args:
            X: (b x 1 x d) point at which to evalaute acquisition function.

        Returns: (b) tensor of acquisition values.
        """

        lookahead_kwargs = self.lookahead_fn(
            model=self.model,
            Xstar=X,
            Xq=X,
            gamma=self.gamma,
            posterior_transform=self.posterior_transform,
        )  # Return shape here has m=1.
        acqf_vals = self._compute_acqf(**lookahead_kwargs)  # level sets x points
        if self.lookahead_type == "levelset":
            acqf_vals = acqf_vals.sum(0).squeeze()
        elif self.lookahead_type == "multi_levelset":
            acqf_vals = torch.log(acqf_vals + 1e-06).sum(
                0
            )  # use logsum to stabilize multiplication of small values
        elif self.lookahead_type == "multi_levelset_pareto_front":
            acqf_vals = is_non_dominated(acqf_vals.T)  # (b x # gammas) -> (b)
        return acqf_vals

    def _compute_acqf(self, **kwargs) -> Tensor:
        raise NotImplementedError


class LocalMI(LocalLookaheadAcquisitionFunction):
    def _compute_acqf(self, Px: Tensor, P1: Tensor, P0: Tensor, py1: Tensor) -> Tensor:
        return MI_fn(Px, P1, P0, py1).sum(dim=-1)


class LocalSUR(LocalLookaheadAcquisitionFunction):
    def _compute_acqf(self, Px: Tensor, P1: Tensor, P0: Tensor, py1: Tensor) -> Tensor:
        return SUR_fn(Px, P1, P0, py1).sum(dim=-1)


@acqf_input_constructor(LocalMI, LocalSUR)
def construct_inputs_local_lookahead(
    model: GPyTorchModel,
    training_data,
    lookahead_type="levelset",
    target: Optional[float] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    **kwargs,
):
    return {
        "model": model,
        "lookahead_type": lookahead_type,
        "target": target,
        "posterior_transform": posterior_transform,
    }


## Global look-ahead acquisitions
class GlobalLookaheadAcquisitionFunction(LookaheadAcquisitionFunction):
    def __init__(
        self,
        model: GPyTorchModel,
        lookahead_type: str = "levelset",
        target: Optional[Union[List[float], float]] = None,
        log_acqf: bool = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        query_set_size: Optional[int] = 256,
        Xq: Optional[Tensor] = None,
        sampling_method: Optional[str] = "sobol_sampling",
    ) -> None:
        """
        A global look-ahead acquisition function.

        Args:
            model: The gpytorch model.
            target: Threshold value(s) to target in p-space.
            Xq: (m x d) global reference set.
            sampling_method: method to sample the query set for global look-ahead. default is sobol_sampling.
                kernel_dist_sampling samples Xq with probability proportional to K(X, Xq). Current implementation only supports RBF kernel.
        """
        super().__init__(
            model=model, target=target, lookahead_type=lookahead_type, log_acqf=log_acqf
        )
        self.posterior_transform = posterior_transform
        self.sampling_method = sampling_method

        if Xq is not None:
            if query_set_size is not None:
                assert Xq.shape[0] == query_set_size, (
                    "If passing both Xq and query_set_size,"
                    + f"first dim of Xq should be query_set_size, got {Xq.shape[0]} != {query_set_size}"
                )
            self.register_buffer("Xq", Xq)
        elif sampling_method == "sobol_sampling":
            assert (
                query_set_size is not None
            ), "Must pass either query set size or a query set!"
            # cast to an int in case we got a float from Config, which
            # would raise on make_scaled_sobol
            query_set_size = cast(int, query_set_size)  # make mypy happy
            assert int(query_set_size) == query_set_size  # make sure casting is safe
            # if the asserts above pass and Xq is None, query_set_size is not None so this is safe
            query_set_size = int(query_set_size)  # cast
            Xq = make_scaled_sobol(model.lb, model.ub, query_set_size)
            self.register_buffer("Xq", Xq)
        else:
            raise NotImplementedError

    @t_batch_mode_transform(assert_output_shape=True)
    def forward(self, X: Tensor) -> Tensor:
        """
        Evaluate acquisition function at X.

        Args:
            X: (b x 1 x d) point at which to evalaute acquisition function.

        Returns: (b) tensor of acquisition values.
        """
        lookahead_kwargs = self._get_lookahead_posterior(X)
        acqf_vals = self._compute_acqf(**lookahead_kwargs)
        if self.log_acqf:
            acqf_vals = logsumexp(acqf_vals, dim=-1)
        elif (acqf_vals.shape[-1] == self.Xq.shape[0]) and not isinstance(
            self, EAVC
        ):  # For EAVC, the output is (num_level_sets x b)
            acqf_vals = acqf_vals.sum(dim=-1)
        acqf_vals = self._reweight_acqf_vals(acqf_vals)
        return acqf_vals

    def _get_lookahead_posterior(
        self, X: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        Xq_batch = self.Xq.expand(X.shape[0], *self.Xq.shape)

        return self.lookahead_fn(
            model=self.model,
            Xstar=X,
            Xq=Xq_batch,
            gamma=self.gamma,
            posterior_transform=self.posterior_transform,
        )

    def _compute_acqf(self, **kwargs) -> Tensor:
        raise NotImplementedError

    def _reweight_acqf_vals(self, acqf_vals):
        if self.lookahead_type == "levelset":
            acqf_vals = acqf_vals.sum(0).squeeze()
        elif self.lookahead_type == "multi_levelset":
            if torch.any(
                acqf_vals < 0
            ):  # bvn estimate isn't accurate for certain values, causing negative acqf vals
                eps = -acqf_vals.min() + 1e-6
            else:
                eps = 1e-6
            acqf_vals = torch.log(acqf_vals + eps).sum(
                0
            )  # use logsum to stabilize multiplication of small values
        elif self.lookahead_type == "multi_levelset_pareto_front":
            acqf_vals = is_non_dominated(acqf_vals.T)  # (b x # gammas) -> (b)
        return acqf_vals


class GlobalMI(GlobalLookaheadAcquisitionFunction):
    def _compute_acqf(self, Px: Tensor, P1: Tensor, P0: Tensor, py1: Tensor) -> Tensor:
        return MI_fn(Px, P1, P0, py1)


class GlobalSUR(GlobalLookaheadAcquisitionFunction):
    def _compute_acqf(self, Px: Tensor, P1: Tensor, P0: Tensor, py1: Tensor) -> Tensor:
        return SUR_fn(Px, P1, P0, py1)


class LogGlobalLookaheadAcquisitionFunction(GlobalLookaheadAcquisitionFunction):
    def __init__(
        self,
        model: GPyTorchModel,
        lookahead_type: str = "levelset",
        target: Optional[Union[List[float], float]] = None,
        log_acqf: bool = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        query_set_size: Optional[int] = 256,
        Xq: Optional[Tensor] = None,
        sampling_method: Optional[str] = "sobol_sampling",
    ) -> None:
        super().__init__(
            model=model,
            target=target,
            lookahead_type=lookahead_type,
            log_acqf=True,
            posterior_transform=posterior_transform,
            query_set_size=query_set_size,
            Xq=Xq,
            sampling_method=sampling_method,
        )

    def _reweight_acqf_vals(self, acqf_vals):
        if self.lookahead_type in ["levelset", "multi_levelset"]:
            acqf_vals = acqf_vals.sum(0).squeeze()
        elif self.lookahead_type == "multi_levelset_pareto_front":
            acqf_vals = is_non_dominated(acqf_vals.T)  # (b x # gammas) -> (b)
        return acqf_vals


class LogGlobalMI(LogGlobalLookaheadAcquisitionFunction):
    """GlobalMI computed in the log space.
    Note: The entropy used in this computation is based on the natural logarithm (ln) rather than the binary logarithm (log2),
    so the acquisition function does not equal to the log of the original GlobalMI.
    """

    def _compute_acqf(
        self, log_cdf_b_q, log_cdf_neg_b_q, log_cdf_a_s, log_cdf_neg_a_s, log_Z_qs
    ):
        tau = 1e-6  # temperature for fatmaximum
        beta = 1e6  # temperature for softplus
        log_one = torch.log(torch.tensor(1.0, dtype=torch.float64))
        cur_entropy = logplusexp(
            log_cdf_b_q + torch.log(softplus(-log_cdf_b_q, beta=beta)),
            log_cdf_neg_b_q + torch.log(softplus(-log_cdf_neg_b_q, beta=beta)),
        )

        lookahead_p1_entropy = log_cdf_a_s + logplusexp(
            log_Z_qs
            - log_cdf_a_s
            + torch.log(softplus(log_cdf_a_s - log_Z_qs, beta=beta)),
            logdiffexp(
                log_Z_qs - log_cdf_a_s,
                fatmaximum(log_one, log_Z_qs - log_cdf_a_s, tau=tau),
            )
            + torch.log(
                softplus(
                    log_cdf_a_s
                    - logdiffexp(log_Z_qs, fatmaximum(log_cdf_a_s, log_Z_qs, tau=tau)),
                    beta=beta,
                )
            ),
        )

        log_diff_1 = logdiffexp(log_Z_qs, fatmaximum(log_cdf_b_q, log_Z_qs, tau=tau))
        log_diff_2 = logdiffexp(
            log_diff_1, fatmaximum(log_cdf_neg_a_s, log_diff_1, tau=tau)
        )
        lookahead_p0_entropy = log_cdf_neg_a_s + logplusexp(
            log_diff_1
            - log_cdf_neg_a_s
            + torch.log(softplus(log_cdf_neg_a_s - log_diff_1, beta=beta)),
            log_diff_2
            - log_cdf_neg_a_s
            + torch.log(softplus(log_cdf_neg_a_s - log_diff_2, beta=beta)),
        )
        expected_lookahead_entropy = logplusexp(
            lookahead_p1_entropy, lookahead_p0_entropy
        )
        log_entropy_reduction = logdiffexp(
            expected_lookahead_entropy,
            fatmaximum(cur_entropy, expected_lookahead_entropy, tau=tau) + tau,
        )
        return log_entropy_reduction


class LogGlobalSUR(LogGlobalLookaheadAcquisitionFunction):
    """GlobalSUR computed in the log space."""

    def _compute_acqf(
        self, log_cdf_b_q, log_cdf_neg_b_q, log_cdf_a_s, log_cdf_neg_a_s, log_Z_qs
    ):
        tau = 1e-6  # temperature for fatmaximum
        cur_misclassification = torch.min(log_cdf_b_q, log_cdf_neg_b_q)

        lookahead_p1_misclassification = torch.min(
            log_Z_qs, logdiffexp(log_Z_qs, fatmaximum(log_cdf_a_s, log_Z_qs, tau=tau))
        )

        log_diff_1 = logdiffexp(log_Z_qs, fatmaximum(log_cdf_b_q, log_Z_qs, tau=tau))
        lookahead_p0_misclassification = torch.min(
            log_diff_1,
            logdiffexp(
                log_diff_1,
                fatmaximum(log_cdf_neg_a_s, log_diff_1, tau=tau),
            ),
        )

        lookahead_misclassification = logplusexp(
            lookahead_p1_misclassification, lookahead_p0_misclassification
        )

        misclassification_reduction = logdiffexp(
            lookahead_misclassification,
            fatmaximum(cur_misclassification, lookahead_misclassification, tau=tau)
            + tau,
        )

        return misclassification_reduction


class ApproxGlobalSUR(GlobalSUR):
    def __init__(
        self,
        model: GPyTorchModel,
        lookahead_type="levelset",
        target: Optional[float] = None,
        query_set_size: Optional[int] = 256,
        Xq: Optional[Tensor] = None,
    ) -> None:
        assert (
            "levelset" in lookahead_type
        ), f"ApproxGlobalSUR only supports lookahead on level set(s), got {lookahead_type}!"
        super().__init__(
            model=model,
            target=target,
            lookahead_type=lookahead_type,
            query_set_size=query_set_size,
            Xq=Xq,
        )

    def _get_lookahead_posterior(
        self, X: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        Xq_batch = self.Xq.expand(X.shape[0], *self.Xq.shape)

        return approximate_lookahead_levelset_at_xstar(
            model=self.model,
            Xstar=X,
            Xq=Xq_batch,
            gamma=self.gamma,
            posterior_transform=self.posterior_transform,
        )


class EAVC(GlobalLookaheadAcquisitionFunction):
    def _compute_acqf(self, Px: Tensor, P1: Tensor, P0: Tensor, py1: Tensor) -> Tensor:
        return EAVC_fn(Px, P1, P0, py1)


class MOCU(GlobalLookaheadAcquisitionFunction):
    """
    MOCU acquisition function given in expr. 4 of:

        Zhao, Guang, et al. "Uncertainty-aware active learning for optimal Bayesian classifier."
        International Conference on Learning Representations (ICLR) 2021.
    """

    def _compute_acqf(self, Px: Tensor, P1: Tensor, P0: Tensor, py1: Tensor) -> Tensor:
        current_max_query = torch.maximum(Px, 1 - Px)
        # expectation w.r.t. y* of the max of pq
        lookahead_pq1_max = torch.maximum(P1, 1 - P1)
        lookahead_pq0_max = torch.maximum(P0, 1 - P0)
        lookahead_max_query = lookahead_pq1_max * py1 + lookahead_pq0_max * (1 - py1)
        return lookahead_max_query - current_max_query


class SMOCU(GlobalLookaheadAcquisitionFunction):
    """
    SMOCU acquisition function given in expr. 11 of:

       Zhao, Guang, et al. "Bayesian active learning by soft mean objective cost of uncertainty."
       International Conference on Artificial Intelligence and Statistics (AISTATS) 2021.
    """

    # the init args are specified, or config would not recognize the arguments
    def __init__(
        self,
        model: GPyTorchModel,
        lookahead_type="posterior",
        target: Optional[float] = None,
        query_set_size: Optional[int] = 256,
        Xq: Optional[Tensor] = None,
        sampling_method: Optional[str] = "sobol_sampling",
        k: Optional[float] = 20.0,
    ):

        super().__init__(
            model=model,
            target=target,
            lookahead_type=lookahead_type,
            query_set_size=query_set_size,
            Xq=Xq,
            sampling_method=sampling_method,
        )
        self.k = k

    def _compute_acqf(self, Px: Tensor, P1: Tensor, P0: Tensor, py1: Tensor) -> Tensor:
        current_softmax_query = (
            torch.logsumexp(self.k * torch.stack((Px, 1 - Px), dim=-1), dim=-1) / self.k
        )
        # expectation w.r.t. y* of the max of pq
        lookahead_pq1_softmax = (
            torch.logsumexp(self.k * torch.stack((P1, 1 - P1), dim=-1), dim=-1) / self.k
        )
        lookahead_pq0_softmax = (
            torch.logsumexp(self.k * torch.stack((P0, 1 - P0), dim=-1), dim=-1) / self.k
        )
        lookahead_softmax_query = (
            lookahead_pq1_softmax * py1 + lookahead_pq0_softmax * (1 - py1)
        )
        return lookahead_softmax_query - current_softmax_query


class BEMPS(GlobalLookaheadAcquisitionFunction):
    """
    BEMPS acquisition function given in:

        Tan, Wei, et al. "Diversity Enhanced Active Learning with Strictly Proper Scoring Rules."
        Advances in Neural Information Processing Systems 34 (2021).
    """

    def __init__(self, scorefun, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scorefun = scorefun

    def _compute_acqf(self, Px: Tensor, P1: Tensor, P0: Tensor, py1: Tensor) -> Tensor:
        current_score = self.scorefun(Px)
        lookahead_pq1_score = self.scorefun(P1)
        lookahead_pq0_score = self.scorefun(P0)
        lookahead_expected_score = lookahead_pq1_score * py1 + lookahead_pq0_score * (
            1 - py1
        )
        return lookahead_expected_score - current_score


class CoreMSE(BEMPS):
    def __init__(
        self,
        model: GPyTorchModel,
        lookahead_type="posterior",
        target: Optional[float] = None,
        query_set_size: Optional[int] = 256,
        Xq: Optional[Tensor] = None,
        sampling_method: Optional[str] = "sobol_sampling",
    ):
        scorefun = lambda p: p**2 + (1 - p) ** 2 - 1
        super().__init__(
            scorefun=scorefun,
            model=model,
            target=target,
            lookahead_type=lookahead_type,
            query_set_size=query_set_size,
            Xq=Xq,
            sampling_method=sampling_method,
        )


class LogCoreMSE(LogGlobalLookaheadAcquisitionFunction):
    def _scorefun(
        self,
        log_p1: Tensor,
        log_p0: Tensor,
    ):
        log_sum_sq = logplusexp(2 * log_p1, 2 * log_p0)
        return log_sum_sq

    def _compute_acqf(
        self,
        log_pq_marginal_1: Tensor,
        log_pq_marginal_0: Tensor,
        log_lookahead_yq1_ystar1: Tensor,
        log_lookahead_yq0_ystar1: Tensor,
        log_lookahead_yq1_ystar0: Tensor,
        log_lookahead_yq0_ystar0: Tensor,
        log_pstar_marginal_1: Tensor,
        log_pstar_marginal_0: Tensor,
    ) -> Tensor:
        log_cur_score = self._scorefun(log_pq_marginal_1, log_pq_marginal_0)
        log_lookahead_ystar1_score = self._scorefun(
            log_lookahead_yq1_ystar1, log_lookahead_yq0_ystar1
        )
        log_lookahead_ystar0_score = self._scorefun(
            log_lookahead_yq1_ystar0, log_lookahead_yq0_ystar0
        )

        log_lookahead_expected_score = logplusexp(
            log_lookahead_ystar1_score + log_pstar_marginal_1,
            log_lookahead_ystar0_score + log_pstar_marginal_0,
        )
        mse_reduction = logdiffexp(
            log_cur_score,
            fatmaximum(log_cur_score, log_lookahead_expected_score, tau=1e-6),
        )
        return mse_reduction


@acqf_input_constructor(GlobalMI, GlobalSUR, ApproxGlobalSUR, EAVC, MOCU, SMOCU, BEMPS)
def construct_inputs_global_lookahead(
    model: GPyTorchModel,
    training_data,
    lookahead_type="levelset",
    target: Optional[float] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    query_set_size: Optional[int] = 256,
    Xq: Optional[Tensor] = None,
    **kwargs,
):
    lb = [bounds[0] for bounds in kwargs["bounds"]]
    ub = [bounds[1] for bounds in kwargs["bounds"]]
    Xq = Xq if Xq is not None else make_scaled_sobol(lb, ub, query_set_size)

    return {
        "model": model,
        "lookahead_type": lookahead_type,
        "target": target,
        "posterior_transform": posterior_transform,
        "query_set_size": query_set_size,
        "Xq": Xq,
    }
