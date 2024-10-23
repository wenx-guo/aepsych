# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import os
import logging
import argparse
import copy
from tqdm import tqdm
import warnings
from botorch.exceptions.warnings import OptimizationWarning

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
import pandas as pd
from copy import deepcopy
from pathlib import Path

import aepsych.utils_logging as utils_logging

logger = utils_logging.getLogger(logging.ERROR)

from aepsych.benchmark import run_benchmarks_with_checkpoints, example_problems
from aepsych.benchmark import DerivedValue
from aepsych.models import GPClassificationModel

problem_map = {
    # "contrast_sensitivity_6d": "ContrastSensitivity6d",
    "hartmann6_binary": "Hartmann6Binary",
    "discrim_lowdim": "DiscrimLowDim",
    "discrim_highdim": "DiscrimHighDim",
}

problem_probability_map = {
    "discrim_highdim": [0.5, 1.0],
    "hartmann6_binary": [0.0, 1.0],
    "contrast_sensitivity_6d": [0.5, 1.0],
    "discrim_lowdim": [0.5, 1.0],
}


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
    parser.add_argument("--output_path", type=str, default="data/benchmark_teacher")
    parser.add_argument("--bench_name", type=str, required=True, default="")
    parser.add_argument("--serial_debug", type=str2bool, default=False)
    parser.add_argument("--hyperparam_fit", type=int, default=100)
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
    hyperparam_fit = args.hyperparam_fit

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
            "generator": ["OptimizeAcqfGenerator", "SobolGenerator"],
            "n_trials": total_trials - init_trials,
            "refit_every": log_every,
        },
        "GPClassificationModel": {
            "inducing_size": 100,
            "mean_covar_factory": "preset_mean_covar_factory",
            "inducing_point_method": "auto",
        },
        "preset_mean_covar_factory": {
            "fixed_mean": False,
            "fixed_kernel_amplitude": True,
            "kernel": "RBFKernel",
        },
        "OptimizeAcqfGenerator": {
            "acqf": ["GlobalMI", "LogGlobalMI", "CoreMSE", "LogCoreMSE", "none"],
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
        "CoreMSE": {"lookahead_type": "posterior"},
        "LogCoreMSE": {"lookahead_type": "posterior"},
    }

    # fit model hyperparameters
    for i_chunk in range(chunks):
        for problem in problems:
            # for some problems we need to artificially generate some data
            torch.manual_seed(global_seed + i_chunk)
            if problem.name == "contrast_sensitivity_6d":
                y = torch.LongTensor(problem.data[:, 0])
                x = torch.Tensor(problem.data[:, 1:])
                problem.m = GPClassificationModel(
                    lb=problem.bounds[0],
                    ub=problem.bounds[1],
                    inducing_size=100,
                    inducing_point_method="kmeans++",
                )
                success = False
                while not success:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", OptimizationWarning)
                        try:
                            problem.m.fit(x, y)
                            success = True
                        except OptimizationWarning:
                            continue

                problem.m.mean_module.constant.requires_grad_(False)
                problem.m.covar_module.raw_lengthscale.requires_grad_(False)
            else:
                torch.manual_seed(i_chunk)
                m = GPClassificationModel(
                    lb=problem.bounds[0],
                    ub=problem.bounds[1],
                    inducing_size=100,
                    inducing_point_method="kmeans++",
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("error", OptimizationWarning)
                    try:
                        m.fit(x, y)
                    except OptimizationWarning:
                        continue

                    mean = problem.m.mean_module.constant.detach().numpy().item()
                    lengthscale = problem.m.covar_module.lengthscale.detach().numpy()[0]
                    df = {f"d{i}": [lengthscale[i]] for i in range(len(problem.lb))}
                    df["mean"] = [mean]
                    df["seed"] = [i_fit]
                    dfs.append(
                        pd.DataFrame(df)
                    )
                df = pd.concat(dfs)
                df.to_csv(out_fname_base + f"/{problem.name}_fitted_lengthscale.csv", index=False)
                import pdb; pdb.set_trace()
                # define teacher model
                ground_truth_lengthscale = df.drop(columns="seed").median(axis=0)

            print(problem.name, problem.thresholds)
            out_fname = os.path.join(out_fname_base, "sota")
            Path(out_fname).mkdir(parents=True, exist_ok=True)
            print(out_fname)

            problem_config = deepcopy(bench_config)
            if problem.name == "discrim_lowdim":
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
                
            problem_config["preset_mean_covar_factory"][
                "lengthscale"
            ] = problem.m.covar_module.lengthscale.clone().detach().numpy()[0]
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
