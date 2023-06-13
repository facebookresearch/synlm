# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from enum import Enum

import torch
import torch.nn.functional as F
from src.utils import timeit


class Trie:
    def __init__(self):
        self.children = {}
        self.is_terminal = False

    def add(self, sequence):
        if len(sequence) >= 1:
            c = sequence[0]
            if c not in self.children:
                self.children[c] = Trie()
            self.children[c].add(sequence[1:])
        else:
            self.is_terminal = True

    def print(self, prefix=""):
        if self.is_terminal:
            print(prefix)
        for c, t in self.children.items():
            t.print(prefix + "\t" + str(c))

    def depth(self):
        max_depth = 0
        for t in self.children.values():
            max_depth = max(max_depth, 1 + t.depth())

        return max_depth

    def __eq__(self, other):
        if self.is_terminal != other.is_terminal:
            return False
        c1 = self.children.keys()
        c2 = other.children.keys()
        if set(c1) != set(c2):
            return False
        else:
            return all([self.children[c] == other.children[c] for c in c1])

    def __repr__(self):
        return "\n".join(self.__repr())

    def __repr(self, prefix=""):
        returns = []
        if self.is_terminal:
            returns.append(prefix)
        for c, t in self.children.items():
            returns.extend(t.__repr(prefix + "\t" + str(c)))

        return returns

    # for field, trie in field_tries.items():
    #     print(field)
    #     with open(f"/tmp/tries/{field}.txt", "w") as f:
    #         for s in print_str(trie):
    #             f.write(s + '\n')

    # Load tries
    # from os import listdir
    # from os.path import isfile, join

    # trie_path = "/tmp/tries"
    # tries2 = {}
    # for fname in listdir(trie_path):
    #     if isfile(join(trie_path, fname)):
    #         with open(join(trie_path, fname), "r") as f:
    #             field = fname.replace('.txt', '')
    #             tries2[field] = Trie()
    #             for s in f.readlines():
    #                 tries2[field].add([int(c) for c in s.strip().split('\t') if len(c) > 0])


class RowState(Enum):
    BEGIN_FIELD = 1
    END_FIELD = 2
    MID_FIELD = 3
    END_ROW = 4


class RowGuide:
    def __init__(self, field_tries, field_begins, field_ends, pad_token, field_order=None):
        self.field_begins = field_begins
        self.field_ends = field_ends
        self.field_tries = field_tries
        self.pad_token = pad_token
        self._field_order = field_order
        self.reset()

    def reset(self):
        self.values = {}

        self.fields = list(self.field_tries.keys()).copy()
        self.state = RowState.BEGIN_FIELD
        self.current_field = None
        if self._field_order is not None:
            self.field_order = self._field_order.copy()
        else:
            random.shuffle(self.fields)
            self.field_order = self.fields


    def next_loglikelihood(self, distribution, token):
        assert distribution is None or distribution.ndim == 1

        if self.state == RowState.BEGIN_FIELD:
            if self.current_field == None:
                potential_fields = [k for k, v in self.field_begins.items() if v[0] == token]
                assert len(potential_fields) == 1 # fine for now but should be improved with a trie over field prompts
                self.current_field = potential_fields[0]
                self.fields.remove(self.current_field)
                self.current_seq = self.field_begins[self.current_field].copy()

            next_token = self.current_seq.pop(0)
            assert token == next_token

            if len(self.current_seq) == 0:
                self.state = RowState.MID_FIELD
                self.current_seq = self.field_tries[self.current_field]

            return 0  # 0 perplexity from predicting field prompt
        elif self.state == RowState.END_FIELD:
            end_token = self.current_seq.pop(0)
            assert token == end_token

            if len(self.current_seq) == 0:
                if len(self.fields) == 0:
                    self.state = RowState.END_ROW
                else:
                    self.state = RowState.BEGIN_FIELD
            return 0  # 0 perplexity from predicting end field
        elif self.state == RowState.END_ROW:
            return 0
        else:
            end_token = self.field_ends[self.current_field][0]
            possible_values = list(self.current_seq.children.keys())
            if self.current_seq.is_terminal:
                possible_values += [end_token]

            assert token in possible_values
            loglikelihood = distribution[token] - torch.logsumexp(distribution[possible_values], dim=0).item()

            if token == end_token:
                self.state = RowState.END_FIELD
                self.current_seq = self.field_ends[self.current_field].copy()
                self.current_field = None
                self.current_seq.pop(0)

                if len(self.current_seq) == 0:
                    if len(self.fields) == 0:
                        self.state = RowState.END_ROW
                    else:
                        self.state = RowState.BEGIN_FIELD
                return loglikelihood

            self.current_seq = self.current_seq.children[token]
            if len(self.current_seq.children) == 0:
                self.current_seq = self.field_ends[self.current_field].copy()
                self.current_field = None
                self.state = RowState.END_FIELD

            return loglikelihood

    def next(self, distribution):
        assert distribution is None or distribution.ndim == 1

        if self.state == RowState.BEGIN_FIELD:
            if self.current_field == None:
                self.current_field = self.field_order.pop(0)
                self.current_seq = self.field_begins[self.current_field].copy()

            next_token = self.current_seq.pop(0)

            if len(self.current_seq) == 0:
                self.state = RowState.MID_FIELD
                self.current_seq = self.field_tries[self.current_field]

            return next_token
        elif self.state == RowState.END_FIELD:
            end_token = self.current_seq.pop(0)

            if len(self.current_seq) == 0:
                if len(self.field_order) == 0:
                    self.state = RowState.END_ROW
                else:
                    self.state = RowState.BEGIN_FIELD
            return end_token
        elif self.state == RowState.END_ROW:
            return self.pad_token
        else:
            end_token = self.field_ends[self.current_field][0]
            possible_values = list(self.current_seq.children.keys())
            if self.current_seq.is_terminal:
                possible_values += [end_token]

            next_token = torch.multinomial(F.softmax(distribution[possible_values]), num_samples=1).item()
            next_token = possible_values[next_token]

            if self.current_field not in self.values:
                self.values[self.current_field] = [next_token]
            else:
                self.values[self.current_field].append(next_token)

            if next_token == end_token:
                self.state = RowState.END_FIELD
                self.current_seq = self.field_ends[self.current_field]
                self.current_field = None
                self.current_seq.pop(0)

                if len(self.current_seq) == 0:
                    if len(self.field_order) == 0:
                        self.state = RowState.END_ROW
                    else:
                        self.state = RowState.BEGIN_FIELD
                return next_token

            self.current_seq = self.current_seq.children[next_token]

            if len(self.current_seq.children) == 0:
                self.current_seq = self.field_ends[self.current_field].copy()
                self.current_field = None
                self.state = RowState.END_FIELD

            return next_token

    def valid_choices(self, tokens: list, vocab_size: int):
        assert isinstance(tokens, list)
        valid_choices = torch.zeros(len(tokens), vocab_size, dtype=torch.bool)
        for i_token, token in enumerate(tokens):

            if self.state == RowState.BEGIN_FIELD:
                # First time we enter this field
                if self.current_field == None:
                    # Find potential fields that match
                    potential_fields = {k: len(v) for k, v in self.field_begins.items() if i_token + len(v) <= len(tokens) and tokens[i_token:i_token+len(v)] == v}
                    max_match = max(potential_fields.values())
                    # If several matches, take the longest one (e.g. 'education' and 'education_num' in adult)
                    potential_fields = [k for k, v in potential_fields.items() if v == max_match]
                    assert len(potential_fields) == 1

                    self.current_field = potential_fields[0]
                    self.fields.remove(self.current_field)
                    self.current_seq = self.field_begins[self.current_field].copy()

                # In general we just keep popping from current_seq
                next_token = self.current_seq.pop(0)
                assert token == next_token
                valid_choices[i_token][token] = 1  # Only one valid choice

                if len(self.current_seq) == 0:
                    self.state = RowState.MID_FIELD
                    self.current_seq = self.field_tries[self.current_field]

            elif self.state == RowState.END_FIELD:
                end_token = self.current_seq.pop(0)
                assert token == end_token

                if len(self.current_seq) == 0:
                    if len(self.fields) == 0:
                        self.state = RowState.END_ROW
                    else:
                        self.state = RowState.BEGIN_FIELD

                valid_choices[i_token][token] = 1  # Only one valid choice
            elif self.state == RowState.END_ROW:
                valid_choices[i_token][self.pad_token] = 1  # Only one valid choice
            else:
                end_token = self.field_ends[self.current_field][0]
                possible_values = list(self.current_seq.children.keys())
                if self.current_seq.is_terminal:
                    possible_values += [end_token]

                assert token in possible_values
                valid_choices[i_token][possible_values] = 1

                if token == end_token:
                    self.state = RowState.END_FIELD
                    self.current_seq = self.field_ends[self.current_field].copy()
                    self.current_field = None
                    self.current_seq.pop(0)

                    if len(self.current_seq) == 0:
                        if len(self.fields) == 0:
                            self.state = RowState.END_ROW
                        else:
                            self.state = RowState.BEGIN_FIELD
                    continue

                self.current_seq = self.current_seq.children[token]
                if len(self.current_seq.children) == 0:
                    self.current_seq = self.field_ends[self.current_field].copy()
                    self.current_field = None
                    self.state = RowState.END_FIELD

        return valid_choices


@timeit
def get_tries(train_df, valid_df, tokenizer):
    keys = train_df[0].keys()
    field_tries = {}
    field_begins = {k: tokenizer.encode("BEGIN_" + k) for k in train_df[0].keys()}
    field_ends = {k: tokenizer.encode("END_" + k) for k in train_df[0].keys()}

    for k in keys:
        values = set([tuple(tokenizer.encode(str(d[k]))) for d in train_df] + [tuple(tokenizer.encode(str(d[k]))) for d in valid_df])

        t = Trie()
        for value in values:
            t.add(value)
        field_tries[k] = t

    return field_begins, field_ends, field_tries


@timeit
def get_tries_tokenized(train_dics, valid_dics):
    keys = train_dics[0].keys()
    field_tries = {}
    for k in keys:
        values = set([tuple(d[k]['field']) for d in train_dics] + [tuple(d[k]['field']) for d in valid_dics])

        t = Trie()
        for value in values:
            t.add(value)
        field_tries[k] = t

    field_begins = {k: v["field_prompt"] for k, v in train_dics[0].items()}
    field_ends = {k: v["end_of_field"] for k, v in train_dics[0].items()}

    return field_begins, field_ends, field_tries
