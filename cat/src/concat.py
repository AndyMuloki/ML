import pandas as pd
from sklearn import preprocessing

# read training data
train = pd.read_csv("../input/train.csv")

# read test data
test = pd.read_csv("../input/test.csv")

# create a fake target column for test data
test.loc[:, "target"] = -1

# concat both training and test data
data = pd.concat([train, test]).reset_index(drop=True)

# make a list of features
# do not encode id and target
features = [x for x in train.columns if x not in ["id", "target"]]

# loop over the features list
for feat in features:
    # create a new instance of LabelEncoder for each feature
    lbl_enc = preprocessing.LabelEncoder()

    temp_col = data[feat].fillna("NONE").astype(str).values

    data.loc[:, feat] = lbl_enc.fit_transform(temp_col)

train = data[data.target != -1].reset_index(drop=True)
test = data[data.target == -1].reset_index(drop=True)

