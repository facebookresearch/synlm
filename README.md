
# Installation

## Dependencies

```bash
pip install -r requirements.txt
```

## Datasets

Download the datasets from their source and put them in a jsonl format. 
Each line of the dataset should look like this:
```json
{"age": 25, "height": "1m78", "city": "Castelnaudary"}
```
Put the dataset in 
```bash
$DATA_PATH/untokenized/${DATASET}_$SPLIT.jsonl
```
where $DATA_PATH is a chosen root, $DATASET is the name of the dataset and $SPLIT is 'train' and 'valid'. 

Tokenize the data. The script will produce data under $DATA_PATH/tokenized/ and $DATA_PATH/discretized/.
```bash
python tokenization.py --dataset $DATASET
```

Modify the paths in `common/paths.py`
```python
# common/paths.py
DATA_PATH = "/my/data/path"
```


# Training a model

The following command trains a language model on the scooter dataset.

```bash
python -train_lm.py \
--steps 10001 \
--lr 0.0005 \
--warmup_steps 100 \
--num_heads 4 \
--num_layers 4 \
--num_embed 256 \
--method real \
--trie_guided true \
--augmentations 2 \
--physical_batch_size 16 \
--dataset $DATASET \
--permute_fields false \
--tokenizer level \
--vocab_size 10000 \
--num_workers 10 \
--print_freq 10 \
--val_freq 1000 \
--architecture custom \
--batch_size 128
```

# License

The majority of SynLM is licensed under CC-BY-NC.  However portions of the project are 
available under separate license terms:  https://github.com/ryan112358/private-pgm/ 
is licensed under the Apache-2 license.

# Citation 

If you use this code in your research please cite
```latex
@article{sablayrolles2023privately,
title={Privately generating tabular data using language models},
author={Sablayrolles, Alexandre and Wang, Yue and Karrer, Brian},
journal={arXiv},
year={2023}
}
```
