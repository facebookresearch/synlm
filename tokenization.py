# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pandas as pd
import time
import json
from sklearn.preprocessing import KBinsDiscretizer
from common.paths import DATA_PATH


def df_from_jsonl(data_path):
    data = []
    # Load data
    with open(data_path) as f:
        start_time = time.time()
        for i_line, line in enumerate(f):
            if i_line % 1000 == 999:
                speed = (i_line + 1) / (time.time() - start_time)
                print(f"Loading line {i_line + 1} at {speed:.2f} lines/s")
            dic = json.loads(line)
            data.append(dic)

    df = pd.DataFrame.from_records(data)

    return df

def jsonl_from_df(df, data_path, field_prompts=None):
    with open(data_path, 'w') as f:
        for _, row in df.iterrows():
            if field_prompts is not None:
                dic = row.to_dict()
                for field in dic.keys():
                    dic[field] = {
                        "field": [int(dic[field])],
                        "field_prompt": [int(field_prompts[field]["field_prompt"])],
                        "end_of_field": [int(field_prompts[field]["end_of_field"])]
                    }
                f.write(json.dumps(dic) + "\n")
            else:
                f.write(json.dumps(row.to_dict()) + "\n")



def tokenize_number(column: pd.Series, nbins=100):
    discretizer = KBinsDiscretizer(n_bins=nbins, encode='ordinal', strategy='uniform')
    discretizer.fit(column.values.reshape(-1, 1))

    tokenized = discretizer.transform(column.values.reshape(-1, 1))
    inversed = discretizer.inverse_transform(tokenized).reshape(-1)

    if column.dtype == int:
        inversed = inversed.round().astype(int)

    return tokenized.reshape(-1), inversed



def tokenize_categorical(column: pd.Series, ncat=100):
    # If more than ncat categories, keep only the ncat most frequent
    if len(column.unique()) > ncat:
        top_categories = column.value_counts().index[:ncat-1]
        column = column.apply(lambda x: x if x in top_categories else 'UNK')

    # Create a mapping from categories to integers
    category_to_int = {category: i for i, category in enumerate(column.unique())}

    return column.apply(lambda x: category_to_int[x]).values, column




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="scooter")
    parser.add_argument("--nbins", type=int, default=100)
    args = parser.parse_args()

    data_path = f"{DATA_PATH}/untokenized/{args.dataset}_{{split}}.jsonl"
    train_df = df_from_jsonl(data_path.format(split="train"))
    valid_df = df_from_jsonl(data_path.format(split="valid"))

    # Concatenate train and valid
    df = pd.concat([train_df, valid_df])

    tokenized_df = {}
    discretized_df = {}
    offset = 0
    for column in df.columns:
        if df[column].dtype in [int, float]:
            tokenized, discretized = tokenize_number(df[column], nbins=args.nbins)
        else:
            tokenized, discretized = tokenize_categorical(df[column], ncat=args.nbins)
        tokenized_df[column] = tokenized + offset
        discretized_df[column] = discretized

        # Update offset from tokenized
        offset += tokenized.max() + 1

    field_prompts = {}
    for column in df.columns:
        field_prompts[column] = {
            "field_prompt": offset,
            "end_of_field": offset + 1,
        }
        offset += 2

    tokenized_df = pd.DataFrame.from_dict(tokenized_df)
    discretized_df = pd.DataFrame.from_dict(discretized_df)

    # Resplit into train and valid
    tokenized_train_df = tokenized_df.iloc[:len(train_df)]
    tokenized_valid_df = tokenized_df.iloc[len(train_df):]

    discretized_train_df = discretized_df.iloc[:len(train_df)]
    discretized_valid_df = discretized_df.iloc[len(train_df):]

    # Save files
    jsonl_from_df(tokenized_train_df, f"{DATA_PATH}/tokenized/{args.dataset}_train.jsonl", field_prompts)
    jsonl_from_df(tokenized_valid_df, f"{DATA_PATH}/tokenized/{args.dataset}_valid.jsonl", field_prompts)

    jsonl_from_df(discretized_train_df, f"{DATA_PATH}/discretized/{args.dataset}_train.jsonl")
    jsonl_from_df(discretized_valid_df, f"{DATA_PATH}/discretized/{args.dataset}_valid.jsonl")