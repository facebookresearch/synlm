# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from mbi import Dataset, Domain

import numpy as np
import itertools
import argparse
from pgm.aim import AIM, compute_cross_entropy
from src.utils import df_from_jsonl
from common.paths import DATA_PATH


def dataset_from_jsonl(train_path, valid_path):
    train_df = df_from_jsonl(train_path)
    valid_df = df_from_jsonl(valid_path)

    columns = list(train_df.columns)
    sizes = []
    remappings = {}
    for column in columns:
        unique_values = set(train_df[column].unique()) | set(valid_df[column].unique())
        sizes.append(len(unique_values))
        remap_dic = {val: i for i, val in enumerate(unique_values)}
        invmap = {i: val for i, val in enumerate(unique_values)}
        remap_this = lambda val: remap_dic[val]

        train_df[column] = train_df[column].apply(remap_this)
        valid_df[column] = valid_df[column].apply(remap_this)
        remappings[column] = invmap

    # Create dataset and domain
    domain = Domain(columns, sizes)
    train_dataset = Dataset(train_df, domain)
    valid_dataset = Dataset(valid_df, domain)

    return train_dataset, valid_dataset, domain, remappings



parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, required=True)
parser.add_argument('--epsilon', type=float, default=5, help='privacy parameter')
parser.add_argument('--delta', type=float, default=1e-6, help='privacy parameter')
parser.add_argument('--max_model_size', type=float, default=80, help='maximum size (in megabytes) of model')
parser.add_argument('--degree', type=int, default=3, help='degree of marginals in workload')
parser.add_argument('--num_marginals', type=int, default=None, help='number of marginals in workload')
parser.add_argument('--max_cells', type=int, default=100000, help='maximum number of cells for marginals in workload')
parser.add_argument("--save_data", type=str, default=None)

args = parser.parse_args()
print(args)

data_path = f"{DATA_PATH}/tokenized/{args.dataset}_{{split}}.jsonl"
trainset, validset, domain, remappings = dataset_from_jsonl(data_path.format(split="train"), data_path.format(split="valid"))

workload = []
for d in range(args.degree):
    workload += list(itertools.combinations(domain, d+1))
workload = [cl for cl in workload if domain.size(cl) <= args.max_cells]
if args.num_marginals is not None:
    workload = [workload[i] for i in np.random.choice(len(workload), args.num_marginals, replace=False)]

workload = [(cl, 1.0) for cl in workload]
mech = AIM(args.epsilon, args.delta, max_model_size=args.max_model_size)
model = mech.run(trainset, workload)

train_xe = compute_cross_entropy(model, trainset)
valid_xe = compute_cross_entropy(model, validset)

print("Train XE", np.mean(train_xe))
print("Valid XE", np.mean(valid_xe))

if args.save_data != None:
    data = model.synthetic_data(rows=len(trainset.df) + len(validset.df)).df
    for column in data.columns:
        data[column] = data[column].apply(lambda val: [remappings[column][val]])

    data.to_csv(args.save_data, index=False)