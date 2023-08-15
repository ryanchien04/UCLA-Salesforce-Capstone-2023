import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

# Load the dataset as an iterable so that it can be streamed
ds = load_dataset("bigcode/the-stack-dedup", streaming=True, data_dir="data/python", split="train")
# Only relevant column is content since this has the code snips (not sure if this makes it faster)
ds = ds.select_columns(["content"])
# iterate over the dataset, every row is stored as a dict
item = next(iter(ds))
# output row
print(item["content"])

# This loop iterates over the whole datset
for row in iter(ds):
    # perturb row and check against ground truth
    print(row["content"])
    break

# tokenizer example this is where chris puts his code
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

def encode(examples):
    return tokenizer(examples['content'], truncation=True, padding='max_length')

# use the encode function as a map over the dataset rows, perform operation in batches
# ds = ds.map(encode, batched=True)
# print(next(iter(ds)))
