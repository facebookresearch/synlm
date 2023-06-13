# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from opacus.layers import DPLSTM

from transformers import AutoConfig, AutoModelForCausalLM, GPT2Config


class LSTMModel(nn.Module):
    def __init__(self, embedding, lstm, linear):
        super().__init__()
        self.embedding = embedding
        self.lstm = lstm
        self.linear = linear

    def forward(self, x):
        embs = self.embedding(x)
        output, _ = self.lstm(embs)
        bs, seq_len = output.shape[:2]
        # output = output.reshape(-1, output.size(-1))
        preds = self.linear(output)
        # preds = preds.view(bs, seq_len, -1)

        return preds


def load_model(args):
    if args.architecture == "gpt2-large":
        size = "L"
        # lm_head_rank = 1024
    elif args.architecture == "gpt2-medium":
        size = "M"
        # lm_head_rank = 1024
    elif args.architecture == "gpt2":
        size = "S"
        # lm_head_rank = 768
    elif args.architecture == "gpt2-xl":
        size = "XL"
        # lm_head_rank = 1024
    elif args.architecture == "distilgpt2":
        size = "D"
    elif args.architecture == "lstm":
        if args.disable_dp:
            lstm = nn.LSTM(
                input_size=args.embedding_size,
                hidden_size=args.lstm_hidden_size,
                num_layers=args.lstm_num_layers,
                batch_first=True,
            )
        else:
            lstm = DPLSTM(
                input_size=args.embedding_size,
                hidden_size=args.lstm_hidden_size,
                num_layers=args.lstm_num_layers,
                batch_first=True,
            )

        model = LSTMModel(
            nn.Embedding(args.vocab_size, args.embedding_size),
            lstm,
            nn.Linear(args.lstm_hidden_size, args.vocab_size),
        )
        return model
    else:
        assert not args.pretrained, "Pretrained model not supported for this architecture"
        config = GPT2Config(
            vocab_size=args.vocab_size,
            return_dict=False,
            n_embd=args.num_embed,
            n_layer=args.num_layers,
            n_head=args.num_heads,
            attn_pdrop=args.attn_pdrop,
            embd_pdrop=args.embd_pdrop,
            resid_pdrop=args.resid_pdrop,
        )
        print(config)
        model = AutoModelForCausalLM.from_config(config)

        return model

    config = AutoConfig.from_pretrained(args.architecture)
    config.vocab_size = args.vocab_size
    model = AutoModelForCausalLM.from_config(config)

    if args.pretrained:
        pretrained_model = AutoModelForCausalLM.from_pretrained(args.architecture)

        for ((name, p), (pretrained_name, pretrained_p)) in zip(model.named_parameters(), pretrained_model.named_parameters()):
            assert name == pretrained_name
            if name in ["lm_head.weight", "transformer.wte.weight"]:
                V = pretrained_p.data.shape[0]
                p.data[:V] = pretrained_p.data
            else:
                p.data = pretrained_p.data

        del pretrained_model

    return model