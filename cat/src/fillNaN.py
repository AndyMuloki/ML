import pandas as pd
from sklearn import preprocessing

df = pd.read_csv("../input/train.csv")

# original = df.ord_2.value_counts()

# print(f"{original}" \n)

# fill NaN values in ord_2 column
# filled = df.ord_2.fillna("NONE").value_counts()

# filled = df.ord_4.fillna("NONE").value_counts()
# print(f"{filled}")

df.ord_4 = df.ord_4.fillna("NONE")


df.loc[
    df["ord_4"].value_counts()[df["ord_4"]].values < 2000,
    "ord_4"
] = "RARE"

print(f"{df.ord_4.value_counts()}")




