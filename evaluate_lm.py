# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import torch

from common.utils import (
    bool_flag,
    init_distributed_mode,
    initialize_exp,
)
from src.model import load_model
from src.data import dataloading
from src.tokenized_data import tokenized_dataloading
from common.paths import DATA_PATH
from src.test import test_std


def print_params(logger, net):
    n_params = sum([p.numel() for p in net.parameters() if p.requires_grad])
    if n_params > 1e9:
        logger.info(f"{n_params // 1e9}B parameters")
    elif n_params > 1e6:
        logger.info(f"{n_params // 1e6}M parameters")
    elif n_params > 1e3:
        logger.info(f"{n_params // 1e3}K parameters")
    else:
        logger.info(f"{n_params} parameters")



def main(params):
    local_path = f"/checkpoint/asablayrolles/synlm/{params.snippet}/checkpoint.pt"
    ckpt = torch.load(local_path)

    # Update current params
    vparams = vars(params)
    for k, v in vars(ckpt["params"]).items():
        if k not in vparams.keys():
            vparams[k] = v

    init_distributed_mode(params)

    # initialize the experiment
    logger, log_stream = initialize_exp(params)

    for k, v in os.environ.items():
        logger.info(f"os.environ[{k}] = {v}")

    # Model
    model = load_model(params)
    model = model.cuda()
    print_params(logger, model)

    state_dict = {k.replace("_module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state_dict)

    # Data
    if params.tokenizer == "level":
        params.train_path = f"{DATA_PATH}/tokenized/{params.dataset}_train.jsonl"
        params.valid_path = f"{DATA_PATH}/tokenized/{params.dataset}_valid.jsonl"
        loaders = tokenized_dataloading(params, params.global_rank, params.world_size)
    else:
        params.train_path = f"{DATA_PATH}/discretized/{params.dataset}_train.jsonl"
        params.valid_path = f"{DATA_PATH}/discretized/{params.dataset}_valid.jsonl"
        loaders = dataloading(params, params.global_rank, params.world_size)

    trainloader, _, valloader, create_loader = loaders

    xe, std = test_std(model, valloader)
    print(f"Test XE {xe}±{std}")




def get_parser():
    parser = argparse.ArgumentParser()

    # main parameters
    parser.add_argument("--snippet", type=str, required=True)
    parser.add_argument("--shuffle", type=bool_flag, default=True)
    parser.add_argument("--n_data", type=int, default=None)
    parser.add_argument("--is_cluster", type=bool_flag, default=False)
    parser.add_argument("--force", type=bool_flag, default=False)

    # Bookkeeping
    parser.add_argument("--dump_path", type=str, default="")
    parser.add_argument("--exp_name", type=str, default="bypass")
    parser.add_argument("--save_periodic", type=int, default=0)
    parser.add_argument("--exp_id", type=str, default="")
    # multi-gpu / multi-node
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--master_port", type=int, default=-1)
    # debug
    parser.add_argument("--debug_train", type=bool_flag, default=False)
    parser.add_argument("--debug_slurm", type=bool_flag, default=False)
    parser.add_argument("--debug", help="Enable all debug flags", action="store_true")

    return parser


def validate_parameters(logger, params, n_data):
    if params.epochs is None:
        assert params.steps is not None
    else:
        assert params.steps is None
        params.steps = params.epochs * n_data // params.batch_size
        logger.info(f"Computed {params.steps} steps from {params.epochs} epochs")
    assert (
        params.warmup_steps < params.steps
    ), f"{params.warmup_steps} warmup steps > {params.steps} steps"


if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()

    main(params)
