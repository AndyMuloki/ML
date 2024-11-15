import pandas as pd

from datasets import load_dataset

ds = load_dataset("scikit-learn/adult-census-income")

df = ds['train'].to_pandas()

print(df['income'].value_counts())

# print(df.columns)