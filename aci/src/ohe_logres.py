import pandas as pd

from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

# one hot encoder with logistic regression

def run(fold):
    # load the training data with folds
    df = pd.read_csv("../input/train_folds.csv")

    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",        
        "hours.per.week"
    ]

    # drop numerical columns
    df = df.drop(num_cols, axis=1)

    # Map targets to 0s and 1s
    target_mapping = {
        "<=50K": 0,
        ">50K": 1
    }
    df['income'] = df['income'].map(target_mapping)

    # Check for NaN values after mapping
    if df['income'].isna().sum() > 0:
        print("NaN values in income column after mapping:", df['income'].isna().sum())
        df = df.dropna(subset=['income'])  # Drop rows where income is NaN

    # Ensure income is an integer type
    df['income'] = df['income'].astype(int)

    # all columns == features except kfold and income
    features = [
        f for f in df.columns if f not in ("kfold", "income")
    ]

    # fill all NaN values with NONE
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # initialize OnehotEncoder
    ohe = preprocessing.OneHotEncoder()

    # fit ohe on train plus validate data
    full_data = pd.concat(
        [df_train[features], df_valid[features]],
        axis=0
    )
    ohe.fit(full_data[features])
   
    # transform training data
    x_train = ohe.transform(df_train[features])

    # transform validation data
    x_valid = ohe.transform(df_valid[features])

    # initialize logistic regression model
    model = linear_model.LogisticRegression()

    # fit model on training data
    model.fit(x_train, df_train.income.values)

    # predict on validation data
    valid_preds = model.predict_proba(x_valid)[:, 1]

    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)

    # print auc
    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)


