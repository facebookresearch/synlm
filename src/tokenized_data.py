# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
import json
import random
import torch
import time

from src.loaders import create_dataloader, create_private_dataloader
from src.trie import RowGuide, get_tries_tokenized

class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, permute=False, augmentations=None, guide=None, vocab_size=None):
        self.data = []
        self.permute = permute
        self.augmentations = augmentations
        self.guide = guide
        self.vocab_size = vocab_size

        with open(data_path) as f:
            start_time = time.time()
            for i_line, line in enumerate(f):
                if i_line % 1000 == 999:
                    speed = (i_line + 1) / (time.time() - start_time)
                    print(f"Loading line {i_line + 1} at {speed:.2f} lines/s")
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __get_one_item(self, index, tensor=False):
        data = self.data[index]
        tokenized_input, tokenized_label = [], []
        keys = list(data.keys())
        if self.permute:
            random.shuffle(keys)

        for key in keys:
            tokenized_input += data[key]["field_prompt"]
            tokenized_label += [-100]

            tokenized_input += data[key]["field"]
            tokenized_label += data[key]["field"]

            tokenized_input += data[key]["end_of_field"]
            tokenized_label += data[key]["end_of_field"]

        if tensor:
            tokenized_input = torch.LongTensor(tokenized_input).view(1, -1)
            tokenized_label = torch.LongTensor(tokenized_label).view(1, -1)

        if self.guide is not None:
            guide = RowGuide(self.guide.field_tries, self.guide.field_begins, self.guide.field_ends, self.guide.pad_token)
            if tensor:
                valid_choices = guide.valid_choices(tokenized_input.view(-1).tolist(), vocab_size=self.vocab_size).unsqueeze(0)
            else:
                valid_choices = guide.valid_choices(tokenized_input, vocab_size=self.vocab_size)
            return tokenized_input, tokenized_label, valid_choices
        else:
            return tokenized_input, tokenized_label


    def __getitem__(self, index):
        if self.augmentations is None:
            return self.__get_one_item(index)
        else:
            augmented_tuples = [
                self.__get_one_item(index, tensor=True)
                for _ in range(self.augmentations)
            ]
            # Some zip magic going on here to convert list of tuples to tuple of lists
            # augmented_input, augmented_label = tuple(zip(*augmented_tuples))
            augmented_data = tuple(zip(*augmented_tuples))
            return tuple([torch.cat(data, dim=0) for data in augmented_data])




def tokenized_dataloading(args, rank=0, world_size=1):
    pad_token = 0
    valid_augmentations = None
    trainset = TokenizedDataset(args.train_path, permute=args.permute_fields, augmentations=args.augmentations, vocab_size=args.vocab_size)
    validset = TokenizedDataset(args.valid_path, permute=args.permute_fields, augmentations=valid_augmentations, vocab_size=args.vocab_size)

    if args.trie_guided:
        field_begins, field_ends, field_tries = get_tries_tokenized(trainset.data, validset.data)
        guide = RowGuide(field_tries, field_begins, field_ends, pad_token)
        trainset.guide = guide
        validset.guide = guide

    create_loader = functools.partial(
        create_dataloader,
        rank=rank,
        world_size=world_size,
        pad_token=pad_token,
        guided=args.trie_guided,
        num_workers=args.num_workers,
    )
    create_private_loader = functools.partial(
        create_private_dataloader,
        pad_token=pad_token,
        guided=args.trie_guided,
        num_workers=args.num_workers,
    )
    trainloader = create_loader(
        dataset=trainset, batch_size=args.batch_size, augmentations=args.augmentations, shuffle=True
    )
    validloader = create_loader(
        dataset=validset, batch_size=args.batch_size_test, augmentations=valid_augmentations, shuffle=False
    )

    return trainloader, None, validloader, create_private_loader