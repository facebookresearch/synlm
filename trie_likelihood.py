# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import time

import numpy as np
import torch
from common.utils import (
    bool_flag,
    download_snippet,
    garbage_collection_cuda,
    init_distributed_mode,
    initialize_exp
)
from src.model import load_model
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from src.trie import RowGuide, get_tries, get_tries_tokenized
from src.data import dataloading
from src.tokenized_data import tokenized_dataloading
from src.test import test
from common.paths import DATA_PATH



def cross_entropy_eval(lm_logits, labels, reduction="mean"):
    """
    Routine from Huggingface's GPT-2 implementation (v 4.7.0)
    """
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    XH = nn.CrossEntropyLoss(reduction=reduction)
    return XH(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def cross_entropy_train(lm_logits, labels):
    """
    Routine from Huggingface's GPT-2 implementation (v 4.7.0)
    """
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    num_labels = torch.sum(shift_labels != -100)

    # Trick to replace -100 with 0 and make vmap happy
    is_negative = shift_labels == -100
    shift_labels += 100 * is_negative.long()

    # Flatten the tokens
    XH = nn.CrossEntropyLoss(reduction="none")
    output = XH(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    return torch.sum(output * torch.logical_not(is_negative.view(-1))) / num_labels



@torch.no_grad()
def test(model, test_loader, physical_batch_size):
    model.eval()
    xe_means, xe_sums = [], []
    n_data = 0
    for data in test_loader:
        if type(data) is list:
            # For some reason, some datasets will have a list of one element instead of a regular batch
            assert len(data) == 1
            data = data[0]

        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        logits = model(inputs)[0]
        xe_mean = cross_entropy_eval(logits, labels, reduction="mean")
        xe_sum = cross_entropy_eval(logits, labels, reduction="sum")

        xe_means.append(xe_mean.item())
        xe_sums.append(xe_sum.item())
        n_data += inputs.size(0)

    garbage_collection_cuda()
    return np.mean(xe_means), np.sum(xe_sums) / n_data


def reload_if_exists(params, model, optimizer, scheduler, privacy_engine):
    if params.snippet is None:
        return 0

    local_path = download_snippet(params.snippet)
    if local_path == 0:
        return 0
    ckpt = torch.load(local_path)

    # Rename stuff because of Opacus
    state_dict = ckpt["model_state_dict"]
    if params.multi_gpu:
        if "_module.module.positional_embedding.emb.weight" in state_dict:
            state_dict["_module.module.positional_embedding.weight"] = state_dict[
                "_module.module.positional_embedding.emb.weight"
            ]
    else:
        if "_module.positional_embedding.emb.weight" in state_dict:
            state_dict["_module.positional_embedding.weight"] = state_dict[
                "_module.positional_embedding.emb.weight"
            ]
    model.load_state_dict(state_dict, strict=False)

    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if privacy_engine is not None:
        privacy_engine.accountant.load_state_dict(ckpt["engine_state_dict"])
    print(f"SUCCESSFULLY LOADED MODEL TO STEP {ckpt['steps']}")

    return ckpt["steps"] + 1




@torch.no_grad()
def generate_batch(model, batch, field_begins, field_ends, field_tries, pad_token):
    start = time.time()
    inputs, _ = batch
    inputs = inputs.long().cuda()

    model.eval()
    logits = model(inputs)[0]
    batch_size = inputs.size(0)
    guides = [
        RowGuide(field_tries, field_begins, field_ends, pad_token)
        for _ in range(batch_size)
    ]
    lls = torch.zeros(batch_size).cuda()
    for t in range(inputs.size(1)):
        for b in range(batch_size):
            if t == 0:
                lls[b] += guides[b].next_loglikelihood(None, inputs[b, t].item())
            else:
                lls[b] += guides[b].next_loglikelihood(logits[b, t-1, :], inputs[b, t].item())

    return -lls



def main(params):
    # download_hf_gpt2("gpt2", do_wait=params.is_cluster)  # to get the tokenizer
    # download_hf_gpt2("toksyn", do_wait=params.is_cluster)  # to get the tokenizer
    # if params.architecture != "custom":
    #     download_hf_gpt2(params.architecture, do_wait=params.is_cluster)
    init_distributed_mode(params)

    # initialize the experiment
    logger, log_stream = initialize_exp(params)

    for k, v in os.environ.items():
        logger.info(f"os.environ[{k}] = {v}")

    # Update current params
    ckpt = torch.load(params.ckpt)
    vparams = vars(params)
    for k, v in vars(ckpt["params"]).items():
        vparams[k] = v

    vparams["augmentations"] = None
    vparams["batch_size"] = 128
    vparams["trie_guided"] = False

    # Model
    model = load_model(params)
    model = model.cuda()
    state_dict = {k.replace("_module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state_dict)

    # Multi-gpu
    is_main_worker = params.global_rank == 0
    if params.multi_gpu:
        if params.disable_dp:
            model = DDP(model)
        else:
            model = DPDDP(model)

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

    if params.tokenizer == "level":
        field_begins, field_ends, field_tries = get_tries_tokenized(trainset.data, validset.data)
    else:
        field_begins, field_ends, field_tries = get_tries(trainset.df, validset.df, trainset.tokenizer)

    loglikelihoods = []
    for batch in valloader:
        lls = generate_batch(model, batch, field_begins, field_ends, field_tries, pad_token=0)
        loglikelihoods.extend(lls.tolist())

    print("Trie guided likelihood: ", np.mean(loglikelihoods))
    results = test(model, valloader, None)
    print("Regular eval", results)


def get_parser():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("--random_seed", type=int, default=0)

    # Model
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--architecture", type=str, default="gpt2")
    parser.add_argument("--pretrained", type=bool_flag, default=True)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--embedding_size", type=int, default=512)
    parser.add_argument("--lstm_hidden_size", type=int, default=1024)
    parser.add_argument("--lstm_num_layers", type=int, default=2)
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=50257,  # tokenizer.from_pretrained("/tmp/gpt2").vocab_size,
    )
    parser.add_argument("--num_embed", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--resid_pdrop", type=float, default=0.1)
    parser.add_argument("--embd_pdrop", type=float, default=0.1)
    parser.add_argument("--attn_pdrop", type=float, default=0.1)

    # Data
    parser.add_argument("--augmentations", type=int, default=None)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--dataset", type=str, default="shopping")
    parser.add_argument("--data_size", type=int, default=None)
    parser.add_argument("--permute_fields", type=bool_flag, default=True)
    parser.add_argument("--split_on_nectar", type=bool_flag, default=True)
    parser.add_argument("--num_workers", type=int, default=0)

    # Optimization
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--batch_size_test", type=int, default=16)
    parser.add_argument("--physical_batch_size", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=4096)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--freeze_emb", type=bool_flag, default=False)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)

    # Privacy parameters
    parser.add_argument("--target_epsilon", type=float, default=5)
    parser.add_argument("--gradclip", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)

    # Bookkeeping
    parser.add_argument("--snippet", type=str, default=None)
    parser.add_argument("--is_cluster", type=bool_flag, default=False)
    parser.add_argument("--print_freq", type=int, default=1000)
    parser.add_argument("--val_freq", type=int, default=1000)
    parser.add_argument("--disable_dp", type=bool_flag, default=True)

    # main parameters
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
