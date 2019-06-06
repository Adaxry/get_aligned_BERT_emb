# get_aligned_BERT_emb
Get the aligned BERT embedding for sequence labeling tasks 

## Why this repo?

In the origin script [extract_features.py](https://github.com/google-research/bert/blob/master/extract_features.py) in BERT, tokens may be splited into pieces as follows:

```python
orig_tokens = ["John", "Johanson", "'s",  "house"]
bert_tokens = ["[CLS]", "john", "johan", "##son", "'", "s", "house", "[SEP]"]
orig_to_tok_map = [1, 2, 4, 6]
```
We investigate 3 align strategies (`first`, `mean` and `max`) to maintain an original-to-tokenized alignment. Take the "`Johanson` -> `johan`, `##son`" as example:

+ `first`: take the representation of `johan` as the whole word `Johanson`
+ `mean`: take the reduce_mean value of representations of `johan` and `##son` as the whole word `Johanson`
+ `max`: take the reduce_max value of representations of `johan` and `##son` as the whole word `Johanson`


## How to use this repo?

```shell
sh run.sh input_file BERT_BASE_DIR
# For example:
sh run.sh your_data path/to/bert/uncased_L-12_H-768_A-12 outout_file
```
You can modify `layers` and `align_strategies` in the `run.sh`.


## How to load the output embeddings?

After the above procedure, you are expected to get a output file of contextual embeddings (e.g., your_data_6_mean). Then you can load this file like conventional word embeddings. For example in a python script:
```python
with open("your_data_6_mean", "r", encoding="utf-8") as bert_f"
    for line in bert_f:
        bert_vec = [[float(value) for value in token.split()] for token in line.strip().split("|||")] 
```


