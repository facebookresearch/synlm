# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
from functools import wraps
import pandas as pd
import json


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
            for k in dic.keys():
                assert len(dic[k]['field']) == 1
                dic[k] = dic[k]['field'][0]
            data.append(dic)

    df = pd.DataFrame.from_records(data)

    return df

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start} seconds")
        return result

    return wrapper