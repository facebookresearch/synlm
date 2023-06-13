# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from opacus.utils.uniform_sampler import (
    DistributedUniformWithReplacementSampler,
    UniformWithReplacementSampler,
)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


def make_collate_and_pad(pad_token, augmentations=None, guided=False):
    if augmentations is None:

        def collate_and_pad(batch):
            batch_size = len(batch)
            max_len = max([len(s[0]) for s in batch])

            collated_input = pad_token * torch.ones(batch_size, max_len, dtype=int)
            collated_label = -100 * torch.ones(batch_size, max_len, dtype=int)

            for i, s in enumerate(batch):
                sentence_length = len(s[0])
                collated_input[i, :sentence_length] = torch.Tensor(s[0])
                collated_label[i, :sentence_length] = torch.Tensor(s[1])

            if guided:
                vocab_size = batch[0][2].shape[-1]
                collated_valid_choices = torch.zeros(batch_size, max_len, vocab_size, dtype=bool)
                for i, s in enumerate(batch):
                    collated_valid_choices[i, :s[2].shape[0]] = s[2]
                return collated_input, collated_label, collated_valid_choices
            else:
                return collated_input, collated_label

    else:

        def collate_and_pad(batch):
            batch_size = len(batch)
            max_len = max([s[0].shape[-1] for s in batch])

            collated_input = pad_token * torch.ones(
                batch_size, augmentations, max_len, dtype=int
            )
            collated_label = -100 * torch.ones(
                batch_size, augmentations, max_len, dtype=int
            )
            for i, s in enumerate(batch):
                sentence_length = s[0].shape[-1]
                collated_input[i, :, :sentence_length] = s[0]
                collated_label[i, :, :sentence_length] = s[1]

            if guided:
                vocab_size = batch[0][2].shape[-1]
                collated_valid_choices = torch.zeros(batch_size, augmentations, max_len, vocab_size, dtype=bool)
                for i, s in enumerate(batch):
                    collated_valid_choices[i, :, :s[2].shape[1]] = s[2] # (A, T, V)

                return collated_input, collated_label, collated_valid_choices
            else:
                return collated_input, collated_label

    return collate_and_pad


def create_dataloader(
    *,
    dataset,
    batch_size,
    shuffle,
    rank,
    world_size,
    pad_token,
    num_workers=0,
    augmentations=None,
    guided=False,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(
            dataset, rank=rank, num_replicas=world_size, shuffle=shuffle
        ),
        num_workers=num_workers,
        collate_fn=make_collate_and_pad(pad_token, augmentations, guided),
    )


def create_private_dataloader(
    *,
    dataset,
    batch_size,
    pad_token,
    distributed,
    steps=None,
    num_workers=0,
    augmentations=None,
    guided=False,
):
    if distributed:
        batch_sampler = DistributedUniformWithReplacementSampler(
            total_size=len(dataset),
            sample_rate=batch_size / len(dataset),
            steps=steps,
        )
    else:
        batch_sampler = UniformWithReplacementSampler(
            num_samples=len(dataset),
            sample_rate=batch_size / len(dataset),
            steps=steps,
        )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=make_collate_and_pad(pad_token, augmentations, guided),
    )
