import numpy as np
import pandas as pd

df = pd.read_csv("../input/train.csv")

# ids = df.groupby(
#     [
#         "ord_1",
#         "ord_2"
#     ]
# )["id"].count().reset_index(name="count")

# print(f"{ids}")

# ## Create new features from categorical variables

df["new_feature"] = (
    df.ord_1.astype(str)
    + "_"
    + df.ord_2.astype(str)
    + "_"
    + df.ord_3.astype(str)
)

print(f"{df.new_feature}")