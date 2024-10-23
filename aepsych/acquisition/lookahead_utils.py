#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, Optional, Tuple

import torch

from botorch.acquisition.objective import PosteriorTransform
from botorch.utils.probability.utils import log_ndtr
from gpytorch.models import GP
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D
from torch import Tensor
from torch.distributions import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from .log_gaussian_quadrature import LogGaussHermiteQuadrature1D
from .bvn import bvn_cdf, log_bvn_cdf


torch.set_default_dtype(torch.float64)


def posterior_at_xstar_xq(
    model: GP,
    Xstar: Tensor,
    Xq: Tensor,
    posterior_transform: Optional[PosteriorTransform] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Evaluate the posteriors of f at single point Xstar and set of points Xq.

    Args:
        model: The model to evaluate.
        Xstar: (b x 1 x d) tensor.
        Xq: (b x m x d) tensor.

    Returns:
        Mu_s: (b x 1) mean at Xstar.
        Sigma2_s: (b x 1) variance at Xstar.
        Mu_q: (b x m) mean at Xq.
        Sigma2_q: (b x m) variance at Xq.
        Sigma_sq: (b x m) covariance between Xstar and each point in Xq.
    """
    # Evaluate posterior and extract needed components
    Xext = torch.cat((Xstar, Xq), dim=-2)
    posterior = model.posterior(Xext, posterior_transform=posterior_transform)
    mu = posterior.mean[..., :, 0]
    Mu_s = mu[..., 0].unsqueeze(-1)
    Mu_q = mu[..., 1:]
    Cov = posterior.distribution.covariance_matrix
    Sigma2_s = Cov[..., 0, 0].unsqueeze(-1)
    Sigma2_q = torch.diagonal(Cov[..., 1:, 1:], dim1=-1, dim2=-2)
    Sigma_sq = Cov[..., 0, 1:]
    return Mu_s, Sigma2_s, Mu_q, Sigma2_q, Sigma_sq


def lookahead_levelset_at_xstar(
    model: GP,
    Xstar: Tensor,
    Xq: Tensor,
    posterior_transform: Optional[PosteriorTransform] = None,
    eps: float = 1e-8,
    **kwargs: Dict[str, Any],
):
    """
    Evaluate the look-ahead level-set posterior at Xq given observation at xstar.

    Args:
        model: The model to evaluate.
        Xstar: (b x 1 x d) observation point.
        Xq: (b x m x d) reference points.
        gamma: Threshold in f-space.

    Returns:
        Px: (b x m) Level-set posterior at Xq, before observation at xstar.
        P1: (b x m) Level-set posterior at Xq, given observation of 1 at xstar.
        P0: (b x m) Level-set posterior at Xq, given observation of 0 at xstar.
        py1: (b x 1) Probability of observing 1 at xstar.
    """
    Mu_s, Sigma2_s, Mu_q, Sigma2_q, Sigma_sq = posterior_at_xstar_xq(
        model=model, Xstar=Xstar, Xq=Xq, posterior_transform=posterior_transform
    )

    try:
        gamma = kwargs.get("gamma")
        assert gamma.ndim == 3 and gamma.shape[0] > 0
    except KeyError:
        raise RuntimeError("lookahead_levelset_at_xtar requires passing gamma!")

    # Compute look-ahead components
    Norm = torch.distributions.Normal(0, 1)
    Sigma_q = torch.sqrt(Sigma2_q)
    b_q = (gamma - Mu_q.unsqueeze(0)) / (Sigma_q + eps)
    Phi_bq = Norm.cdf(b_q)
    denom = torch.sqrt(1 + Sigma2_s)
    a_s = Mu_s / denom
    Phi_as = Norm.cdf(a_s)
    Z_rho = -Sigma_sq / (Sigma_q * denom + eps)
    Z_qs = bvn_cdf(a_s, b_q, Z_rho)

    Px = Phi_bq
    py1 = Phi_as
    P1 = Z_qs / (py1 + eps)
    P0 = (Phi_bq - Z_qs) / (1 - py1 + eps)
    return {"Px": Px, "P1": P1, "P0": P0, "py1": py1}


def log_lookahead_levelset_at_xstar(
    model: GP,
    Xstar: Tensor,
    Xq: Tensor,
    posterior_transform: Optional[PosteriorTransform] = None,
    eps: float = 1e-8,
    **kwargs: Dict[str, Any],
):
    """
    Evaluate the look-ahead level-set posterior at Xq given observation at xstar.

    Args:
        model: The model to evaluate.
        Xstar: (b x 1 x d) observation point.
        Xq: (b x m x d) reference points.
        gamma: Threshold in f-space.

    Returns:
        Px: (b x m) Level-set posterior at Xq, before observation at xstar.
        P1: (b x m) Level-set posterior at Xq, given observation of 1 at xstar.
        P0: (b x m) Level-set posterior at Xq, given observation of 0 at xstar.
        py1: (b x 1) Probability of observing 1 at xstar.
    """
    Mu_s, Sigma2_s, Mu_q, Sigma2_q, Sigma_sq = posterior_at_xstar_xq(
        model=model, Xstar=Xstar, Xq=Xq, posterior_transform=posterior_transform
    )

    try:
        gamma = kwargs.get("gamma")
        assert gamma.ndim == 3 and gamma.shape[0] > 0
    except KeyError:
        raise RuntimeError("lookahead_levelset_at_xtar requires passing gamma!")

    # Compute look-ahead components
    Sigma_q = torch.sqrt(Sigma2_q)
    b_q = (gamma - Mu_q.unsqueeze(0)) / (Sigma_q + eps)
    log_cdf_b_q = log_ndtr(b_q)
    log_cdf_neg_b_q = log_ndtr(-b_q)

    denom = torch.sqrt(1 + Sigma2_s)
    a_s = Mu_s / (denom + eps)
    log_cdf_a_s = log_ndtr(a_s)
    log_cdf_neg_a_s = log_ndtr(-a_s)

    Z_rho = -Sigma_sq / (Sigma_q * denom + eps)
    log_Z_qs = log_bvn_cdf(a_s, b_q, Z_rho)

    return {
        "log_cdf_b_q": log_cdf_b_q,
        "log_cdf_neg_b_q": log_cdf_neg_b_q,
        "log_cdf_a_s": log_cdf_a_s,
        "log_cdf_neg_a_s": log_cdf_neg_a_s,
        "log_Z_qs": log_Z_qs,
    }


def lookahead_p_at_xstar(
    model: GP,
    Xstar: Tensor,
    Xq: Tensor,
    posterior_transform: Optional[PosteriorTransform] = None,
    **kwargs: Dict[str, Any],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Evaluate the look-ahead response probability posterior at Xq given observation at xstar.

    Uses the approximation given in expr. 9 in:
    Zhao, Guang, et al. "Efficient active learning for Gaussian process classification by
    error reduction." Advances in Neural Information Processing Systems 34 (2021): 9734-9746.


    Args:
        model: The model to evaluate.
        Xstar: (b x 1 x d) observation point.
        Xq: (b x m x d) reference points.
        kwargs: ignored (here for compatibility with other kinds of lookahead)

    Returns:
        Px: (b x m) Response posterior at Xq, before observation at xstar.
        P1: (b x m) Response posterior at Xq, given observation of 1 at xstar.
        P0: (b x m) Response posterior at Xq, given observation of 0 at xstar.
        py1: (b x 1) Probability of observing 1 at xstar.
    """
    Mu_s, Sigma2_s, Mu_q, Sigma2_q, Sigma_sq = posterior_at_xstar_xq(
        model=model, Xstar=Xstar, Xq=Xq, posterior_transform=posterior_transform
    )

    probit = Normal(0, 1).cdf

    def lookahead_inner(f_q):
        mu_tilde_star = Mu_s + (f_q - Mu_q) * Sigma_sq / Sigma2_q
        sigma_tilde_star = Sigma2_s - (Sigma_sq**2) / Sigma2_q
        return probit(mu_tilde_star / torch.sqrt(sigma_tilde_star + 1)) * probit(f_q)

    pstar_marginal_1 = probit(Mu_s / torch.sqrt(1 + Sigma2_s))
    pstar_marginal_0 = 1 - pstar_marginal_1
    pq_marginal_1 = probit(Mu_q / torch.sqrt(1 + Sigma2_q))

    quad = GaussHermiteQuadrature1D()
    fq_mvn = Normal(Mu_q, torch.sqrt(Sigma2_q))
    joint_ystar1_yq1 = quad(lookahead_inner, fq_mvn)
    joint_ystar0_yq1 = pq_marginal_1 - joint_ystar1_yq1

    # now we need from the joint to the marginal on xq
    lookahead_pq1 = joint_ystar1_yq1 / pstar_marginal_1
    lookahead_pq0 = joint_ystar0_yq1 / pstar_marginal_0
    return {
        "Px": pq_marginal_1,
        "P1": lookahead_pq1,
        "P0": lookahead_pq0,
        "py1": pstar_marginal_1,
    }


def log_lookahead_p_at_xstar(
    model: GP,
    Xstar: Tensor,
    Xq: Tensor,
    posterior_transform: Optional[PosteriorTransform] = None,
    **kwargs: Dict[str, Any],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """log lookahead posterior probability at xstar, xq
    To avoid issues in the log space, e.g. when logdiffexp operates on two potentially equal probabilty,
    we calculate the integral for the joint probability (ystar, yq) separately
    """
    Mu_s, Sigma2_s, Mu_q, Sigma2_q, Sigma_sq = posterior_at_xstar_xq(
        model=model, Xstar=Xstar, Xq=Xq, posterior_transform=posterior_transform
    )

    sigma_tilde_star = Sigma2_s - (Sigma_sq**2) / Sigma2_q

    def log_lookahead_inner_s1q1(f_q):  # p(ystar=1, yq=1 | xstar, xq)
        mu_tilde_star = Mu_s + (f_q - Mu_q) * Sigma_sq / Sigma2_q
        return log_ndtr(mu_tilde_star / torch.sqrt(sigma_tilde_star + 1)) + log_ndtr(
            f_q
        )

    def log_lookahead_inner_s0q0(f_q):  # p(ystar=0, yq=0 | xstar, xq)
        mu_tilde_star = Mu_s + (f_q - Mu_q) * Sigma_sq / Sigma2_q
        return log_ndtr(-mu_tilde_star / torch.sqrt(sigma_tilde_star + 1)) + log_ndtr(
            -f_q
        )

    def log_lookahead_inner_s1q0(f_q):  # p(ystar=1, yq=0 | xstar, xq)
        mu_tilde_star = Mu_s + (f_q - Mu_q) * Sigma_sq / Sigma2_q
        return log_ndtr(mu_tilde_star / torch.sqrt(sigma_tilde_star + 1)) + log_ndtr(
            -f_q
        )

    def log_lookahead_inner_s0q1(f_q):  # p(ystar=1, yq=0 | xstar, xq)
        mu_tilde_star = Mu_s + (f_q - Mu_q) * Sigma_sq / Sigma2_q
        return log_ndtr(-mu_tilde_star / torch.sqrt(sigma_tilde_star + 1)) + log_ndtr(
            f_q
        )

    log_pstar_marginal_1 = log_ndtr(Mu_s / torch.sqrt(1 + Sigma2_s))
    log_pstar_marginal_0 = log_ndtr(-Mu_s / torch.sqrt(1 + Sigma2_s))
    log_pq_marginal_1 = log_ndtr(Mu_q / torch.sqrt(1 + Sigma2_q))
    log_pq_marginal_0 = log_ndtr(-Mu_q / torch.sqrt(1 + Sigma2_q))

    # we calculate the integral for the joint probability (ystar, yq) separately
    # rather than using the relations of joint and marginal probability because the latter involves
    # one to two logdiffexp operations, with potential numerical issues in the log space
    # when the marginal and joint probabilities are close to each other
    log_quad = LogGaussHermiteQuadrature1D()
    fq_mvn = Normal(Mu_q, torch.sqrt(Sigma2_q))
    log_joint_ystar1_yq1 = log_quad(log_lookahead_inner_s1q1, fq_mvn)
    log_joint_ystar1_yq0 = log_quad(log_lookahead_inner_s1q0, fq_mvn)
    log_joint_ystar0_yq1 = log_quad(log_lookahead_inner_s0q1, fq_mvn)
    log_joint_ystar0_yq0 = log_quad(log_lookahead_inner_s0q0, fq_mvn)

    # from the joint to the marginal on xq
    log_lookahead_yq1_ystar1 = log_joint_ystar1_yq1 - log_pstar_marginal_1
    log_lookahead_yq0_ystar1 = log_joint_ystar1_yq0 - log_pstar_marginal_1
    log_lookahead_yq1_ystar0 = log_joint_ystar0_yq1 - log_pstar_marginal_0
    log_lookahead_yq0_ystar0 = log_joint_ystar0_yq0 - log_pstar_marginal_0

    return {
        "log_pq_marginal_1": log_pq_marginal_1,
        "log_pq_marginal_0": log_pq_marginal_0,
        "log_lookahead_yq1_ystar1": log_lookahead_yq1_ystar1,
        "log_lookahead_yq0_ystar1": log_lookahead_yq0_ystar1,
        "log_lookahead_yq1_ystar0": log_lookahead_yq1_ystar0,
        "log_lookahead_yq0_ystar0": log_lookahead_yq0_ystar0,
        "log_pstar_marginal_1": log_pstar_marginal_1,
        "log_pstar_marginal_0": log_pstar_marginal_0,
    }


def approximate_lookahead_levelset_at_xstar(
    model: GP,
    Xstar: Tensor,
    Xq: Tensor,
    gamma: float,
    posterior_transform: Optional[PosteriorTransform] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    The look-ahead posterior approximation of Lyu et al.

    Args:
        model: The model to evaluate.
        Xstar: (b x 1 x d) observation point.
        Xq: (b x m x d) reference points.
        gamma: Threshold in f-space.

    Returns:
        Px: (b x m) Level-set posterior at Xq, before observation at xstar.
        P1: (b x m) Level-set posterior at Xq, given observation of 1 at xstar.
        P0: (b x m) Level-set posterior at Xq, given observation of 0 at xstar.
        py1: (b x 1) Probability of observing 1 at xstar.
    """
    assert gamma.ndim == 3 and gamma.shape[0] > 0
    Mu_s, Sigma2_s, Mu_q, Sigma2_q, Sigma_sq = posterior_at_xstar_xq(
        model=model, Xstar=Xstar, Xq=Xq, posterior_transform=posterior_transform
    )
    Mu_q = Mu_q.unsqueeze(0)

    Norm = torch.distributions.Normal(0, 1)
    Mu_s_pdf = torch.exp(Norm.log_prob(Mu_s))
    Mu_s_cdf = Norm.cdf(Mu_s)

    # Formulae from the supplement of the paper (Result 2)
    vnp1_p = Mu_s_pdf**2 / Mu_s_cdf**2 + Mu_s * Mu_s_pdf / Mu_s_cdf  # (C.4)
    p_p = Norm.cdf(Mu_s / torch.sqrt(1 + Sigma2_s))  # (C.5)

    vnp1_n = Mu_s_pdf**2 / (1 - Mu_s_cdf) ** 2 - Mu_s * Mu_s_pdf / (
        1 - Mu_s_cdf
    )  # (C.6)
    p_n = 1 - p_p  # (C.7)

    vtild = vnp1_p * p_p + vnp1_n * p_n

    Sigma2_q_np1 = Sigma2_q - Sigma_sq**2 / ((1 / vtild) + Sigma2_s)  # (C.8)

    Px = Norm.cdf((gamma - Mu_q) / torch.sqrt(Sigma2_q))
    P1 = Norm.cdf((gamma - Mu_q) / torch.sqrt(Sigma2_q_np1))
    P0 = P1  # Same because we ignore value of y in this approximation
    py1 = 0.5 * torch.ones(*Px.shape[:-1], 1)  # Value doesn't matter because P1 = P0
    return Px, P1, P0, py1
