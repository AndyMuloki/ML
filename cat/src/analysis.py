import numpy as np
import pandas as pd

df = pd.read_csv("../input/train.csv")


# # Number of ids in dataframe where the value of ord_2 is 'Boiling Hot' 
# ids = df[df.ord_2 == "Boiling Hot"].shape

# # Number of ids in df for each value of ord_2
ids = df.groupby(["ord_2"])["id"].count()

print(f"{ids}\n")

tr = df.groupby(["ord_2"])["id"].transform("count")
print(tr)