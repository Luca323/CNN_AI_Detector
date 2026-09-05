import torch
from datasets import load_dataset, load_from_disk
import pandas as pd
pd.set_option('display.max_colwidth', None)

# Instantly maps the 16.7 GB dataset into Python with zero download wait time
print("Loading dataset from local disk...")
dataset = load_from_disk("./my_local_raid_dataset")

# Target specific splits just like before
train_data = dataset['train'].to_pandas()
print(train_data.head(5))

train_data['label'] = (train_data['model'] != 'human').astype(int)

print(train_data.value_counts('label'))
