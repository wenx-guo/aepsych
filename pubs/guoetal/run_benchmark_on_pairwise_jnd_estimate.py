# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import os
import logging

import argparse

# run each job single-threaded, paralellize using pathos
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# multi-socket friendly args
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
os.environ["KMP_BLOCKTIME"] = "1"
import torch

# force torch to 1 thread too just in case
torch.set_num_interop_threads(1)
torch.set_num_threads(1)

import numpy as np
from copy import deepcopy
from pathlib import Path
from scipy.stats import norm

import aepsych.utils_logging as utils_logging

logger = utils_logging.getLogger(logging.ERROR)

from aepsych_prerelease.benchmark import run_benchmarks_with_checkpoints
from aepsych.benchmark import DerivedValue
from aepsych_prerelease.benchmark import example_problems

problem_map = {
    # "pairwise_discrim_lowdim": "PairwiseDiscrimLowdim",
    # "pairwise_dipper_1d": "PairwiseDipper1d",
    # "pairwise_discrim_highdim": "PairwiseDiscrimHighdim",
    # "pairwise_hartmann6_binary": "PairwiseHartmann6Binary",
}


def make_argparser():
    str2bool = lambda s: str(s).lower() in ("yes", "true", "t", "y", "1")
    parser = argparse.ArgumentParser(description="Lookahead LSE Benchmarks")
    parser.add_argument("--nproc", type=int, default=60)
    parser.add_argument("--reps_per_chunk", type=int, default=20)
    parser.add_argument("--chunks", type=int, default=2)
    parser.add_argument("--total_trials", type=int, default=750)  # 490
    parser.add_argument("--init_trials", type=int, default=10)
    parser.add_argument("--global_seed", type=int, default=1000)
    parser.add_argument("--log_frequency", type=int, default=10)
    parser.add_argument("--output_path", type=str, default="data/pairwise_benchmark")
    parser.add_argument("--bench_name", type=str, required=True, default="")
    parser.add_argument("--serial_debug", type=str2bool, default=False)
    parser.add_argument(
        "--problem",
        type=str,
        default="all",
    )
    return parser


if __name__ == "__main__":

    parser = make_argparser()
    args = parser.parse_args()
    chunks = (
        args.chunks
    )  # The number of chunks to break the results into. Each chunk will contain at least 1 run of every
    # combination of problem and config.
    start_idx = 0
    reps_per_chunk = (
        args.reps_per_chunk
    )  # Number of repetitions to run each problem/config in each chunk.

    out_fname_base = args.output_path
    nproc = args.nproc  # how many processes to use
    global_seed = args.global_seed  # random seed for reproducibility
    log_every = args.log_frequency  # log to csv every this many trials
    checkpoint_every = 120  # save intermediate results every this many seconds
    serial_debug = (
        args.serial_debug
    )  # whether to run simulations serially for debugging
    bench_name = args.bench_name
    total_trials = int(args.total_trials)
    init_trials = args.init_trials

    problems = []
    for i in range(1, 4):
        problems.append(getattr(example_problems, "Energy2Intensity2d")(subject_id=i))
    for sub_experiment in ["local", "strength"]:
        for i in range(0, 3):
            problems.append(
                getattr(example_problems, "Wavelet2d")(
                    subject_id=i, sub_experiment=sub_experiment
                )
            )
    for problem_name in problem_map.keys():
        problems.append(
            getattr(example_problems, problem_map[problem_name])(thresholds=None)
        )

    bench_config = {
        "common": {
            "stimuli_per_trial": 2,
            "outcome_types": "binary",
            "strategy_names": "[init_strat, opt_strat]",
            "invalid_config": DerivedValue(
                [("opt_strat", "generator"), ("OptimizeAcqfGenerator", "acqf")],
                lambda generator, acqf: (
                    True
                    if (generator == "SobolGenerator" and acqf != "none")
                    or (generator == "OptimizeAcqfGenerator" and acqf == "none")
                    else False
                ),
            ),
        },
        "init_strat": {"n_trials": init_trials, "generator": "SobolGenerator"},
        "opt_strat": {
            "model": "GPClassificationModel",
            "generator": "OptimizeAcqfGenerator",
            "n_trials": total_trials - init_trials,
            "refit_every": log_every,
        },
        "GPClassificationModel": {
            "inducing_size": 100,
            "mean_covar_factory": "default_mean_covar_factory",
            "inducing_point_method": "auto",
        },
        "default_mean_covar_factory": {
            "fixed_mean": False,
            "fixed_kernel_amplitude": True,
            "lengthscale_prior": "lognormal",
            "kernel": "RBFKernel",
        },
        "OptimizeAcqfGenerator": {
            "acqf": [
                # "none",
                # "GlobalMI",
                # "BernoulliMCMutualInformation",
                # "MOCU",
                "LogGlobalMI",
                # "CoreMSE",
            ],
            "restarts": 2,
            "samps": 100,
        },
        # "BernoulliMCMutualInformation": {
        #     "objective": "ProbitObjective",
        # },
        # "GlobalMI": {
        #     "target": norm.cdf(1).item(),
        #     "lookahead_type": "levelset",
        #     "sampling_method": "sobol_sampling",
        # },
        # "MOCU": {"lookahead_type": "posterior"},
        "CoreMSE": {"lookahead_type": "posterior"},
        "LogGlobalMI": {
            "target": norm.cdf(1).item(),
            "lookahead_type": "levelset",
            "sampling_method": "sobol_sampling",
        },
    }
    print(bench_config)

    out_fname = os.path.join(out_fname_base, "logglobalmi_lognormal_prior_soft")
    Path(out_fname).mkdir(parents=True, exist_ok=True)

    for problem in problems:
        levelset_bench_name = (
            f"{bench_name}{problem.name}_novel_acqf_{np.round(norm.cdf(1).item(), 2)}"
        )
        run_benchmarks_with_checkpoints(
            out_fname,
            levelset_bench_name,
            [problem],
            bench_config,
            global_seed,
            start_idx,
            chunks,
            reps_per_chunk,
            log_every,
            checkpoint_every,
            nproc,
            serial_debug,
        )
