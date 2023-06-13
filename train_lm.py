# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import math
import os
import time

from common.paths import DATA_PATH
import numpy as np
import torch
import torch.nn.functional as F
from functorch import grad_and_value, make_functional_with_buffers, vmap
from common.utils import (
    bool_flag,
    download_snippet,
    garbage_collection_cuda,
    init_distributed_mode,
    initialize_exp,
)

# from src.data import download_hf_gpt2
from src.model import load_model

from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant

from opacus.accountants.utils import get_noise_multiplier
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP

# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from src.tokenized_data import tokenized_dataloading
from src.data import dataloading
from src.test import test, cross_entropy_train


def get_sigma_tan(q, steps, epsilon, delta):
    sigma_tan = get_noise_multiplier(
        target_epsilon=epsilon, target_delta=delta, sample_rate=1.0, steps=1, accountant='rdp'
    )
    sigma = sigma_tan * q * np.sqrt(steps)

    return sigma


def get_epsilon_tan(q, sigma, steps, delta):
    # Using RDPAccountant as PRVAccountant only needed if q<1
    accountant = RDPAccountant()
    sigma_tan = sigma / (q * np.sqrt(steps))
    accountant.history = [(sigma_tan, 1.0, 1)]

    return accountant.get_epsilon(delta)


def predict_next(model: nn.Module, current_batch: torch.Tensor):
    outputs = model(current_batch)  # (B, T, V)

    return outputs[:, -1]


def sample(model, test_loader, logger):
    batch_size = 1
    max_field_len = 5

    model.eval()
    try:
        tokenizer = test_loader.dataset.tokenizer
        sample = test_loader.dataset.df[0]
    except Exception:
        # if test_loader.dataset is a Subset
        tokenizer = test_loader.dataset.dataset.tokenizer
        sample = test_loader.dataset.dataset.df[0]

    fields = list(sample.keys())
    current_sample = torch.zeros(batch_size, 0, dtype=int)
    for i_field in range(len(fields)):
        begin_field = torch.LongTensor(
            tokenizer.encode("BEGIN_" + fields[i_field])
        ).unsqueeze(0)
        end_field = torch.LongTensor(
            tokenizer.encode("END_" + fields[i_field])
        ).unsqueeze(0)
        current_sample = torch.cat([current_sample, begin_field], dim=1)
        field_len = 0

        while field_len < max_field_len:
            preds = predict_next(model, current_sample.cuda())
            preds = F.softmax(preds, dim=-1)
            next_token = torch.multinomial(preds, num_samples=1).cpu()
            current_sample = torch.cat([current_sample, next_token], dim=1)

            if next_token == end_field[0, 0]:
                # Model predicts next field: we switch to it
                break
            else:
                # Otherwise we continue business as usual
                field_len += 1

            if field_len == max_field_len:
                current_sample = torch.cat([current_sample, end_field], dim=1)

    for generated_sample in current_sample:
        logger.info(tokenizer.decode(generated_sample.tolist()))

    garbage_collection_cuda()


def train(
    params,
    model,
    optimizer,
    scheduler,
    trainloader,
    valloader,
    step_init,
    is_main_worker,
    privacy_engine,
    logger,
):
    losses = []
    # Buffers here are attention masks
    fmodel, _fparams, buffers = make_functional_with_buffers(model)
    del _fparams

    if params.augmentations is None:

        if params.trie_guided:

            def compute_loss_stateless_model(params, buffers, sample, target, mask):
                batch = sample.unsqueeze(0)
                targets = target.unsqueeze(0)
                masks = mask.unsqueeze(0)

                predictions = fmodel(params, buffers, batch)[0]
                loss = cross_entropy_train(predictions, targets, masks)

                return loss

        else:

            def compute_loss_stateless_model(params, buffers, sample, target):
                batch = sample.unsqueeze(0)
                targets = target.unsqueeze(0)

                predictions = fmodel(params, buffers, batch)[0]
                # loss = criterion(predictions.view(-1, predictions.shape[-1]), targets.view(-1))
                loss = cross_entropy_train(predictions, targets)

                return loss

    else:
        if params.trie_guided:
                def compute_loss_stateless_model(params, buffers, sample, target, mask):
                    predictions = fmodel(params, buffers, sample)[0]
                    loss = cross_entropy_train(predictions, target, mask)

                    return loss
        else:
            def compute_loss_stateless_model(params, buffers, sample, target):
                predictions = fmodel(params, buffers, sample)[0]
                loss = cross_entropy_train(predictions, target)

                return loss

    ft_compute_grad = grad_and_value(compute_loss_stateless_model)
    if params.trie_guided:
        ft_compute_sample_grad = vmap(
            ft_compute_grad, in_dims=(None, None, 0, 0, 0), randomness="same"
        )
    else:
        ft_compute_sample_grad = vmap(
            ft_compute_grad, in_dims=(None, None, 0, 0), randomness="different"
        )
    parameters = list(model.parameters())

    # B = 256
    # T = 64
    # inputs = torch.randint(0, V, (B, T)).cuda()
    # targets = torch.randint(0, V, (B, T)).cuda()
    # grads, losses = ft_compute_sample_grad(params, buffers, inputs, targets)

    if is_main_worker and len(trainloader) > 1000 and not params.is_cluster:
        it = tqdm(trainloader)
    else:
        it = trainloader

    virtual_step = 0

    data_time, step_time, remain_time = 0, 0, 0
    t = time.time()
    for i, data in enumerate(it):
        data_time += time.time() - t
        t = time.time()
        step = i + step_init
        (
            norm_before_clip,
            norm_after_clip,
            norm_after_noise,
            losses_step,
            batch_virtual_steps,
        ) = train_one_step(
            physical_batch_size=params.physical_batch_size,
            params=params,
            parameters=parameters,
            fmodel=fmodel,
            ft_compute_sample_grad=ft_compute_sample_grad,
            buffers=buffers,
            optimizer=optimizer,
            scheduler=scheduler,
            data=data,
        )
        step_time += time.time() - t
        t = time.time()
        virtual_step += batch_virtual_steps
        losses.extend(losses_step)

        # Validation logging
        if step % params.val_freq == 0:
            xe_val_mean, xe_val_sum = test(model, valloader)

            if is_main_worker and params.is_cluster and params.snippet is not None:
                save_model(params, model, optimizer, scheduler, privacy_engine, step)

            if is_main_worker:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"XE_val: {xe_val_mean}\tPerplexity_val: {np.exp(xe_val_mean)}\tLR: {lr}\tsteps: {step}"
                )
                logger.info(
                    "__log:"
                    + json.dumps(
                        {
                            "xe_val_mean": xe_val_mean,
                            "xe_val_sum": xe_val_sum,
                            "perp_val_mean": np.exp(xe_val_mean),
                            "perp_val_sum": np.exp(xe_val_sum),
                            "lr": lr,
                            "steps": step,
                        }
                    )
                )

        # Logging train stuff
        if is_main_worker and step % params.print_freq == 0 and step >= 1:
            train_loss = np.mean(losses)
            pp = np.exp(train_loss)
            losses = []

            logger.info(
                f"XE_train: {train_loss}\tPerplexity_train: {pp}\tnorm_before_clip:{norm_before_clip}\tsteps: {step} (logical), {virtual_step}(physical)"
            )
            logger.info(
                f"data_time: {data_time / step}\tstep_time: {step_time / step}\tremain_time: {remain_time / step}"
            )

            stats = {"xe_train": train_loss, "perp_train": pp, "step": step}
            if not params.disable_dp:
                if params.method == "real":
                    try:
                        stats["epsilon"] = privacy_engine.get_epsilon(delta=params.delta)
                    except:
                        # Sometimes PRV accountant is not happy with small values of noise
                        stats["epsilon"] = -1
                stats["epsilon_TAN"] = get_epsilon_tan(
                    q=params.batch_size / len(trainloader.dataset),
                    sigma=params.sigma,
                    steps=step,
                    delta=params.delta,
                )
                logger.info(
                    f"Norms: clipped {norm_after_clip}\tnoised {norm_after_noise}"
                )

            logger.info("__log:" + json.dumps(stats))
        remain_time += time.time() - t
        t = time.time()


def train_one_step(
    physical_batch_size,
    params,
    parameters,
    fmodel,
    ft_compute_sample_grad,
    buffers,
    optimizer,
    scheduler,
    data,
):
    assert not params.disable_dp, "Code only works for DP"
    fmodel.train()

    # Splitting logical batch into physical batches
    # batch_inputs, batch_labels = data
    batch_size = data[0].shape[0]
    if batch_size >= physical_batch_size:
        chunks = math.ceil(batch_size / physical_batch_size)
        # batch_inputs = torch.chunk(batch_inputs, chunks=chunks)
        # batch_labels = torch.chunk(batch_labels, chunks=chunks)

        batch_data = [torch.chunk(d, chunks=chunks) for d in data]
        # Converting tuple of list to list of tuples
        batch_data = zip(*batch_data)
    else:
        chunks = 1
        batch_data = [data]

    all_losses = []
    for i_chunk, (sample) in enumerate(batch_data):
        inputs = sample[0].cuda()
        labels = sample[1].cuda()

        # Forward/backward
        optimizer.zero_grad(set_to_none=True)
        if params.trie_guided:
            mask = sample[2].cuda()
            grads, losses = ft_compute_sample_grad(parameters, buffers, inputs, labels, mask)
        else:
            grads, losses = ft_compute_sample_grad(parameters, buffers, inputs, labels)
        grads = [g.detach() for g in grads]
        for (p, g) in zip(parameters, grads):
            p.grad_sample = g

        # Logging
        with torch.no_grad():
            all_losses.append(losses.mean().item())
            norm_before_clip = torch.stack(
                [p.grad_sample.mean().norm(2) for p in parameters if p.requires_grad]
            ).norm(2)

        # Step
        if params.disable_dp:
            if i_chunk == chunks - 1:
                optimizer.step()
        else:
            optimizer.signal_skip_step(i_chunk < chunks - 1)
            optimizer.step()

    scheduler.step()

    # Logging
    norm_after_clip, norm_after_noise = 0, 0
    if not params.disable_dp:
        with torch.no_grad():
            norm_after_clip = torch.stack(
                [
                    p.summed_grad.norm(2) / optimizer.expected_batch_size
                    for p in parameters
                    if p.requires_grad
                ]
            ).norm(2)

            norm_after_noise = torch.stack(
                [p.grad.norm(2) for p in parameters if p.requires_grad]
            ).norm(2)

    garbage_collection_cuda()

    return norm_before_clip, norm_after_clip, norm_after_noise, all_losses, chunks


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


def save_model(params, model, optimizer, scheduler, privacy_engine, steps):
    print(f"SAVED MODEL AT STEP {steps}")

    ckpt = {
        "params": params,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "steps": steps,
    }
    if privacy_engine is not None:
        ckpt["engine_state_dict"] = privacy_engine.accountant.state_dict()

    local_path = f"/checkpoint/asablayrolles/synlm/{params.snippet}/checkpoint.pt"
    torch.save(ckpt, local_path)


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

    # Model
    model = load_model(params)
    model = model.cuda()
    print_params(logger, model)

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

    trainset = loaders[0].dataset
    logger.info(
        f"dataset_size={len(trainset)}, data_loader_size={len(loaders[0])}, rank={params.global_rank}, world_size={params.world_size}, steps={params.steps}"
    )
    trainloader, testloader, valloader, create_loader = loaders
    validate_parameters(logger, params, len(trainset))

    if params.multi_gpu:
        weights = model.module.parameters()
    else:
        weights = model.parameters()

    # Optimizer, lr scheduler
    optimizer = optim.AdamW(weights, lr=params.lr, weight_decay=params.weight_decay, betas=(params.beta1, params.beta2))

    # LR scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=params.warmup_steps,
        num_training_steps=params.steps,
    )

    privacy_engine = None
    if not params.disable_dp:
        privacy_engine = PrivacyEngine()
        model.train()  # to satisfy validator

        # Compute sigma_tan, scale up batch size and recompute sigma
        sample_rate = params.batch_size / len(trainset)
        sigma_tan = get_sigma_tan(
            q=sample_rate,
            steps=params.steps,
            epsilon=params.target_epsilon,
            delta=params.delta,
        )

        if params.method == "real":
            params.batch_size = int(max(2.0 / sigma_tan, 1) * params.batch_size)
            sample_rate = params.batch_size / len(trainset)
            params.sigma = get_noise_multiplier(
                target_epsilon=params.target_epsilon, target_delta=params.delta, sample_rate=sample_rate, steps=params.steps, accountant='prv'
            )
        else:
            assert params.method == "tan"
            params.sigma = sigma_tan

        logger.info(f"New batch size: {params.batch_size} (sigma: {params.sigma})")

        # Manually doing make_private work because the batch size has changed since creating trainloader
        model = privacy_engine._prepare_model(model, grad_sample_mode="no_op")

        optimizer = privacy_engine._prepare_optimizer(
            optimizer,
            noise_multiplier=params.sigma,
            max_grad_norm=params.gradclip,
            expected_batch_size=params.batch_size,
            grad_sample_mode="no_op",
        )
        optimizer.attach_step_hook(
            privacy_engine.accountant.get_optimizer_hook_fn(sample_rate=sample_rate)
        )



    # Reload checkpoint and optionnally data
    step_init = reload_if_exists(params, model, optimizer, scheduler, privacy_engine)
    trainloader = create_loader(
        dataset=trainset,
        batch_size=params.batch_size,
        steps=params.steps - step_init,
        augmentations=params.augmentations,
        distributed=params.multi_gpu,
    )

    train(
        params=params,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        trainloader=trainloader,
        valloader=valloader,
        step_init=step_init,
        is_main_worker=is_main_worker,
        privacy_engine=privacy_engine,
        logger=logger,
    )

    return log_stream.getvalue()


def get_parser():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("--random_seed", type=int, default=0)

    # Model
    parser.add_argument("--architecture", type=str, default="gpt2")
    parser.add_argument("--pretrained", type=bool_flag, default=False)
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
    parser.add_argument("--trie_guided", type=bool_flag, default=False)

    # Data
    parser.add_argument("--augmentations", type=int, default=None)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--dataset", type=str, default="shopping")
    parser.add_argument("--data_size", type=int, default=None)
    parser.add_argument("--permute_fields", type=bool_flag, default=True)
    parser.add_argument("--split_on_nectar", type=bool_flag, default=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--tokens_beginend", type=bool_flag, default=True)

    # Optimization
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--batch_size_test", type=int, default=16)
    parser.add_argument("--physical_batch_size", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=4096)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--freeze_emb", type=bool_flag, default=False)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)

    # Privacy parameters
    parser.add_argument("--method", type=str, default="real")
    parser.add_argument("--target_epsilon", type=float, default=5)
    parser.add_argument("--gradclip", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-6)

    # Bookkeeping
    parser.add_argument("--snippet", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--is_cluster", type=bool_flag, default=False)
    parser.add_argument("--print_freq", type=int, default=1000)
    parser.add_argument("--val_freq", type=int, default=1000)
    parser.add_argument("--disable_dp", type=bool_flag, default=False)

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
    if params.delta is None:
        params.delta = 1 / n_data
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

    if params.checkpoint_dir is not None and params.snippet is None:
        print(f"CHECKPOINT DIR {params.checkpoint_dir}")
        params.snippet = "/".join(params.checkpoint_dir.split("/")[-2:])

    main(params)
