# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
from common.utils import (
    garbage_collection_cuda,
)
from src.utils import timeit


def cross_entropy_eval(lm_logits, labels, reduction="mean"):
    """
    Routine from Huggingface's GPT-2 implementation (v 4.7.0)
    """
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    XH = nn.CrossEntropyLoss(reduction=reduction)
    return XH(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def cross_entropy_train(lm_logits, labels, mask=None, reduction="mean"):
    """
    Cross entropy loss
    Includes trick to make vmap happy with -100
    """
    shift_logits = lm_logits[..., :-1, :].contiguous()  # (B, T - 1, V)
    shift_labels = labels[..., 1:].contiguous()         # (B, T - 1)

    num_labels = torch.sum(shift_labels != -100)

    # Trick to replace -100 with 0 and make vmap happy
    is_negative = shift_labels == -100
    shift_labels += 100 * is_negative.long()

    # Flatten the tokens
    XH = nn.CrossEntropyLoss(reduction="none")
    if mask is None:
        output = XH(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.size())
    else:
        shift_mask = mask[..., 1:, :].contiguous()          # (B, T - 1, V)
        logits_masked = torch.where(shift_mask, shift_logits, -300 * torch.ones_like(shift_logits))

        val = shift_logits.view(-1, shift_logits.size(-1)).gather(1, shift_labels.view(-1, 1)).view(shift_labels.size())
        lse = torch.logsumexp(logits_masked, dim=-1)
        output = lse - val

    if reduction == "mean":
        return torch.sum(output * torch.logical_not(is_negative)) / num_labels
    elif reduction == "sum":
        return torch.sum(output * torch.logical_not(is_negative))
    else:
        return output * torch.logical_not(is_negative)


@timeit
@torch.no_grad()
def test(model, test_loader):
    model.eval()
    xe_means, xe_sums = [], []
    n_data = 0
    for data in test_loader:
        if type(data) is list:
            # For some reason, some datasets will have a list of one element instead of a regular batch
            assert len(data) == 1
            data = data[0]

        inputs = data[0].cuda()
        labels = data[1].cuda()

        logits = model(inputs)[0]
        if len(data) == 3:
            mask = data[2].cuda()
            xe_mean = cross_entropy_train(logits, labels, mask=mask, reduction="mean")
            xe_sum = cross_entropy_train(logits, labels, mask=mask, reduction="sum")
        else:
            xe_mean = cross_entropy_eval(logits, labels, reduction="mean")
            xe_sum = cross_entropy_eval(logits, labels, reduction="sum")

        xe_means.append(xe_mean.item())
        xe_sums.append(xe_sum.item())
        n_data += inputs.size(0)

    garbage_collection_cuda()
    return np.mean(xe_means), np.sum(xe_sums) / n_data

@timeit
@torch.no_grad()
def test_std(model, test_loader):
    model.eval()
    xe_sums = []
    n_data = 0
    for data in test_loader:
        if type(data) is list:
            # For some reason, some datasets will have a list of one element instead of a regular batch
            assert len(data) == 1
            data = data[0]

        inputs = data[0].cuda()
        labels = data[1].cuda()

        logits = model(inputs)[0]
        if len(data) == 3:
            mask = data[2].cuda()
            xe = cross_entropy_train(logits, labels, mask=mask, reduction="none")

        xe_sums += xe.sum(dim=1).tolist()
        n_data += inputs.size(0)

    return np.mean(xe_sums), np.std(xe_sums) / np.sqrt(n_data)
