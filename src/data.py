# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
import random

import json
import torch
from src.loaders import create_dataloader, create_private_dataloader
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from src.trie import RowGuide, get_tries


class DataFrameDataset(Dataset):
    def __init__(self, df, tokenizer, permute=True, augmentations=None, guide=None):
        self.df = df
        self.tokenizer = tokenizer
        self.permute = permute
        self.augmentations = augmentations
        self.guide = guide

    def __len__(self):
        return len(self.df)

    def __get_one_item(self, index, tensor=False):
        dic = self.df[index]
        tokenized_input, tokenized_label = [], []
        keys = list(dic.keys())
        if self.permute:
            random.shuffle(keys)

        for key in keys:
            tokenized_key = self.tokenizer.encode("BEGIN_" + key)
            tokenized_input += tokenized_key
            tokenized_label += [-100] * len(tokenized_key)

            tokenized_value = self.tokenizer.encode(str(dic[key]))
            tokenized_input += tokenized_value
            tokenized_label += tokenized_value

            tokenized_key = self.tokenizer.encode("END_" + key)
            tokenized_input += tokenized_key
            tokenized_label += tokenized_key

        if tensor:
            tokenized_input = torch.LongTensor(tokenized_input).view(1, -1)
            tokenized_label = torch.LongTensor(tokenized_label).view(1, -1)

        if self.guide is not None:
            guide = RowGuide(self.guide.field_tries, self.guide.field_begins, self.guide.field_ends, self.guide.pad_token)
            if tensor:
                valid_choices = guide.valid_choices(tokenized_input.view(-1).tolist(), vocab_size=len(self.tokenizer)).unsqueeze(0)
            else:
                valid_choices = guide.valid_choices(tokenized_input, vocab_size=len(self.tokenizer))
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


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.read().splitlines()]


def dataloading(args, rank, world_size):
    # Load tokenizer
    if args.tokenizer == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    else:
        tokenizer = GPT2Tokenizer.from_pretrained("/tmp/toksyn")

    # Load df
    train_df = load_jsonl(args.train_path)
    valid_df = load_jsonl(args.valid_path)

    # Add field tokens
    fields = list(train_df[0].keys())

    if args.tokens_beginend:
        begin_tokens = [f"BEGIN_{f}" for f in fields]
        tokenizer.add_tokens(begin_tokens)
        end_tokens = [f"END_{f}" for f in fields]
        tokenizer.add_tokens(end_tokens)

    # Pad tokenizer to match vocab_size
    if len(tokenizer) < args.vocab_size:
        print("Padding tokenizer to match vocab size")
        tokenizer.add_tokens(
            [f"NEW_TOKEN_{i}" for i in range(args.vocab_size - len(tokenizer))]
        )
    pad_token = tokenizer.encode(tokenizer.eos_token)[0]

    if args.trie_guided:
        field_begins, field_ends, field_tries = get_tries(train_df, valid_df, tokenizer)
        guide = RowGuide(field_tries, field_begins, field_ends, pad_token)
    else:
        guide = None

    trainset = DataFrameDataset(
        train_df,
        tokenizer,
        permute=args.permute_fields,
        augmentations=args.augmentations,
        guide=guide,
    )

    validset = DataFrameDataset(
        valid_df,
        tokenizer,
        permute=args.permute_fields,
        augmentations=None,
        guide=guide,
    )

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


    valid_augmentations = None
    trainloader = create_loader(
        dataset=trainset, batch_size=args.batch_size, augmentations=args.augmentations, shuffle=True
    )
    validloader = create_loader(
        dataset=validset, batch_size=args.batch_size_test, augmentations=valid_augmentations, shuffle=False
    )

    return trainloader, None, validloader, create_private_loader
