# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import time

from math import ceil

import pandas as pd
import torch

from common.utils import (
    bool_flag,
    init_distributed_mode,
    initialize_exp,
)
from src.model import load_model
from src.trie import RowState, RowGuide, Trie, get_tries_tokenized, get_tries
from src.data import dataloading
from src.tokenized_data import tokenized_dataloading
from common.paths import DATA_PATH


@torch.no_grad()
def generate_batch(model, batch_size, field_begins, field_ends, field_tries, pad_token, field_order):
    start = time.time()

    model.eval()
    guides = [
        RowGuide(field_tries, field_begins, field_ends, pad_token, field_order)
        for _ in range(batch_size)
    ]
    t = 0
    while any([g.state != RowState.END_ROW for g in guides]):
        tokens = torch.zeros(batch_size, 1, dtype=torch.long).to(model.device)
        # Create a tensor of size (batch_size, 1)
        for b in range(batch_size):
            if t == 0:
                tokens[b, 0] = guides[b].next(None)
            else:
                tokens[b, 0] = guides[b].next(logits[b, t-1, :])
        if t == 0:
            inputs = tokens
        else:
            inputs = torch.cat([inputs, tokens], dim=1)

        t += 1
        logits = model(inputs)[0]

    return [g.values for g in guides]



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


def get_field_tries(df, tokenizer, verbose=True):
    field_tries = {}

    for column in df.columns:
        # print(column + '\t\t\t' + str(df[column].nunique()) + '(' + str(df_transformed.df[column].nunique()) + ')')
        # print(df[column].value_counts())
        values = df[column].unique()
        t = Trie()
        for value in values:
            t.add(tokenizer.encode(str(value)))

        if verbose:
            print(f"{column}: num:{len(values)}, depth:{t.depth()}")
        field_tries[column] = t

    return field_tries


def parse_to_df(generated, fields):
    dics = []
    for s in generated:
        dic = {}
        for field in fields:
            if not s.startswith(f"BEGIN_{field} "):
                break
            end_field = s.find(f"END_{field}")
            if end_field == -1:
                break
            dic[field] = s[len(f"BEGIN_{field} ") : end_field].strip()
            s = s[end_field + len(f"END_{field} ") :]
        dics.append(dic)

    return pd.DataFrame(dics)


# def generate_df(tokenizer, model, field_tries, n_data, shuffle=False):
def generate_df(model, n_data, field_begins, field_ends, field_tries, pad_token, field_order):
    batch_size = 32  # 64
    n_batches = ceil(n_data / batch_size)

    dics = []
    for _ in range(n_batches):
        generated = generate_batch(model, batch_size, field_begins, field_ends, field_tries, pad_token, field_order)
        dics.extend(generated)

    return pd.DataFrame(dics)


def main(params):
    local_path = f"/checkpoint/asablayrolles/synlm/{params.snippet}/checkpoint.pt"
    ckpt = torch.load(local_path)

    # Update current params
    vparams = vars(params)
    for k, v in vars(ckpt["params"]).items():
        if k not in vparams.keys():
            vparams[k] = v

    vparams["trie_guided"] = False # we guide the generation ourselves

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
    trainset = trainloader.dataset
    validset = valloader.dataset

    if params.n_data is None:
        params.n_data = len(trainset)

    # Generate synthetic data
    # field_tries = get_field_tries(trainset.original_df, tokenizer)
    # synthetic_data = generate_df(tokenizer, model, field_tries, params.n_data, shuffle=params.shuffle)

    if params.tokenizer == "level":
        field_begins, field_ends, field_tries = get_tries_tokenized(trainset.data, validset.data)
        field_order = None
        if not params.permute_fields:
            field_order = list(trainset.data[0].keys())
            print("FIELD ORDER", field_order)
    else:
        field_begins, field_ends, field_tries = get_tries(trainset.df, validset.df, trainset.tokenizer)


    # Synthetic data

    synthetic_df = generate_df(model, params.n_data, field_begins, field_ends, field_tries, pad_token=0, field_order=field_order)
    data_path = local_path.replace(".pt", ".csv")
    synthetic_df.to_csv(data_path, index=False)
    print(f"Model saved to {data_path}")

    return log_stream.getvalue()


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
