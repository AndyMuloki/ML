import pandas as pd

import xgboost as xgb

from sklearn import metrics
from sklearn import preprocessing


def run(fold):
    # load full training data with folds
    df = pd.read_csv("../input/train_folds.csv")

    features = [
        f for f in df.columns if f not in ("id", "target", "kfold")
    ]

    # fill all NaN values with NONE
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # label encode the features
    for col in features:
        lbl = preprocessing.LabelEncoder()

        # fit label encoder on all data
        lbl.fit(df[col])

        # transform all the dat
        df.loc[:, col] = lbl.transform(df[col])

    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # get training data
    x_train = df_train[features].values

    # validation data
    x_valid = df_valid[features].values

    # initialize xgboost model
    model = xgb.XGBClassifier(
        n_jobs=-1,
        max_depth=7,
        n_estimators=200
    )

    #fit model on training data (ohe)
    model.fit(x_train, df_train.target.values)

    # predict on validation data
    valid_preds = model.predict_proba(x_valid)[:, 1]

    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    # print auc
    print(f"Fold = {fold}, AUC = {auc}")


if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)