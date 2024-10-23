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

import aepsych.utils_logging as utils_logging

logger = utils_logging.getLogger(logging.ERROR)

from aepsych_prerelease.benchmark import run_benchmarks_with_checkpoints
from aepsych.benchmark import DerivedValue
from aepsych_prerelease.benchmark import example_problems

problem_map = {
    "contrast_sensitivity_6d": "ContrastSensitivity6d",
    # "hartmann6_binary": "Hartmann6Binary",
    # "discrim_lowdim": "DiscrimLowDim",
    # "discrim_highdim": "DiscrimHighDim",
    # "hartmann_binary_high_dim_10": "HartmannBinaryHighDimEmbed"
    # "ackley_10_dim": "Ackley"
}
# "remote_haptics_2d": "RemoteHapticsShortDuration2d",
# for subject_id in range(1, 8):
#     problem_map[f"remote_haptics_3d_subj_{subject_id}"] = ["RemoteHaptics3d", subject_id]

problem_probability_map = {
    "discrim_highdim": [0.5, 1.0],
    "hartmann6_binary": [0.0, 1.0],
    "contrast_sensitivity_6d": [0.5, 1.0],
    "remote_haptics_2d": [0.0, 1.0],
    "discrim_lowdim": [0.5, 1.0],
    "hartmann_binary_high_dim_10": [0.0, 1.0],
    "ackley_10_dim": [0.5, 1.0],
}

for subject_id in range(1, 8):
    problem_probability_map[f"remote_haptics_3d_subj_{subject_id}"] = [0.0, 1.0]


def make_argparser():
    str2bool = lambda s: str(s).lower() in ("yes", "true", "t", "y", "1")
    parser = argparse.ArgumentParser(description="Lookahead LSE Benchmarks")
    parser.add_argument("--nproc", type=int, default=72)
    parser.add_argument("--reps_per_chunk", type=int, default=12)
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument("--total_trials", type=int, default=750)
    parser.add_argument("--init_trials", type=int, default=10)
    parser.add_argument("--global_seed", type=int, default=1000)
    parser.add_argument("--log_frequency", type=int, default=10)
    parser.add_argument(
        "--output_path", type=str, default="data/benchmark_log_normal_ls_unit_variance"
    )
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
    start_idx = 11
    reps_per_chunk = (
        args.reps_per_chunk
    )  # Number of repetitions to run each problem/config in each chunk.

    out_fname_base = args.output_path
    Path(out_fname_base).mkdir(parents=True, exist_ok=True)
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

    # passing DerivedValue with total_trials as an argument into the opt_strat dict in parallel mode breaks
    def create_derived_value(total_trials):
        return DerivedValue([("init_strat", "n_trials")], lambda x: total_trials - x)

    # initialize problems
    problem_names = problem_map.keys() if args.problem == "all" else [args.problem]
    problems = []
    for problem_name in problem_names:
        low_p, high_p = problem_probability_map[problem_name]
        n = 4 if (high_p - low_p == 0.5) else 5
        epsilon = 0.1
        thresholds_to_record = (
            np.linspace(low_p + epsilon, high_p - epsilon, n)
            .astype(np.float32)
            .tolist()
        )
        thresholds_to_record = np.round(thresholds_to_record, 2).tolist()
        problem_value = problem_map[problem_name]
        if isinstance(problem_value, list):
            assert len(problem_value) == 2
            problems.append(
                getattr(example_problems, problem_value[0])(
                    problem_value[1], thresholds_to_record
                )
            )
        else:
            problems.append(
                getattr(example_problems, problem_value)(thresholds_to_record)
            )
    print([problem.thresholds for problem in problems])

    bench_config = {
        "common": {
            "stimuli_per_trial": 1,
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
            "fixed_lengthscale_amplitude": True,
            "lengthscale_prior": "lognormal",
            "kernel": "RBFKernel",
        },
        "OptimizeAcqfGenerator": {
            "acqf": ["LogCoreMSE", "GlobalMI", "LogGlobalMI"],
            "restarts": 2,
            "samps": 100,
        },
        "GlobalMI": {
            "lookahead_type": "multi_levelset",
            "sampling_method": "sobol_sampling",
            
        },
        "LogGlobalMI": {
            "lookahead_type": "multi_levelset",
            "sampling_method": "sobol_sampling",
        },
        # "CoreMSE": {"lookahead_type": "posterior"},
        "LogCoreMSE": {"lookahead_type": "posterior"},
        # "MOCU": {"lookahead_type": "posterior"},
        # "BernoulliMCMutualInformation": {
        #     "objective": "ProbitObjective",
        # },
    }

    for i_chunk in range(chunks):
        for problem in problems:
            print(problem.name, problem.thresholds)
            if start_idx + i_chunk == 11 and problem.name in [
                "contrast_sensitivity_6d",
                "hartmann6_binary",
            ]:
                continue
            out_fname = os.path.join(out_fname_base, "log_coremse")
            # out_fname = os.path.join(out_fname_base, "globalmi_multi")
            # out_fname = os.path.join(out_fname_base, "ackley_10_dim")
            Path(out_fname).mkdir(parents=True, exist_ok=True)
            print(out_fname)

            problem_config = deepcopy(bench_config)
            if problem.name in ["remote_haptics_2d", "discrim_lowdim"]:
                problem_config["opt_strat"]["n_trials"] = 500 - init_trials
            else:
                problem_config["opt_strat"]["n_trials"] = 750 - init_trials

            low_p, high_p = problem_probability_map[problem.name]
            epsilon = 0.1
            n = 4 if (high_p - low_p == 0.5) else 5
            target_list = np.linspace(low_p + epsilon, high_p - epsilon, n).astype(
                np.float64
            )
            for acqf in ["GlobalMI", "LogGlobalMI"]:
                problem_config[acqf]["target"] = target_list

            print(problem_config)

            config_bench_name = f"{bench_name}{problem.name}"
            run_benchmarks_with_checkpoints(
                out_fname,
                config_bench_name,
                [problem],
                problem_config,
                global_seed,
                start_idx + i_chunk,
                1,
                reps_per_chunk,
                log_every,
                checkpoint_every,
                nproc,
                serial_debug,
            )
