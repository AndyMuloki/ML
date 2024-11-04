import pandas as pd
from sklearn import preprocessing

# create dictionary 
"""
 mapping = {
    "Freezing": 0,
    "Warm":1,
    "Cold":2,
    "Boiling Hot":3,
    "Hot":4,
    "Lava Hot":5
}
"""

# df = pd.read_csv("../input/train.csv")
# df.loc[:, "ord_2"] = df.ord_2.map(mapping)

# df.ord_2.value_counts()

# print(df.ord_2.value_counts())
# ------------------------------------------------------

# ## Using LabelEncoder from scikit-learn

df = pd.read_csv("../input/train.csv")

# fill NaN values in ord_2 column
df.loc[:, "ord_2"] = df.ord_2.fillna("NONE")

# initialize LabelEncoder
lbl_enc = preprocessing.LabelEncoder()

df.loc[:, "ord_2"] = lbl_enc.fit_transform(df.ord_2.values)

print(df.ord_2.value_counts()) 