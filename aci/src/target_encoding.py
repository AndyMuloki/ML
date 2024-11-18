import copy
import pandas as pd

from sklearn import metrics
from sklearn import preprocessing
import xgboost as xgb


def mean_target_encoding(data):
    # make a copy of df
    df = copy.deepcopy(data)

    # list of numerical columns
    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]

    # map targets
    target_mapping = {
        "<=50K": 0,
        ">50K": 1
    }
    df['income'] = df['income'].map(target_mapping).astype(int)

     # all columns are features except kfold and income
    features = [
        f for f in df.columns if f not in ("kfold", "income")
        and f not in num_cols
    ]

    # label encoding non-numerical features
    for col in features:
        if col not in num_cols:  # Skip label encoding for numerical columns
            df[col] = df[col].fillna("NONE").astype(str)
            lbl = preprocessing.LabelEncoder()
            df[col] = lbl.fit_transform(df[col])

    encoded_dfs = []

    # go over all folds
    for fold in range(5):
        # fetch training and validation data
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        # for all feature cols
        for column in features:
            # create dictionary of category:mean target
            mapping_dict = dict(
                df_train.groupby(column)["income"].mean()
            )
            # column_enc is the new column we have with mean encoding
            df_valid.loc[
                :, column + "_enc"
            ] = df_valid[column].map(mapping_dict)
        # append to list of encoded validation dataframes
        encoded_dfs.append(df_valid)
    # create full data frame again and return
    encoded_df = pd.concat(encoded_dfs, axis=0)
    return encoded_df


def run(df, fold):
    # folds are the same
    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # get validation data
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # all columns are features except kfold and income
    features = [
        f for f in df.columns if f not in ("kfold", "income")
    ]

    # scale training data
    x_train = df_train[features].values

    # scale validation data
    x_valid = df_valid[features].values

    # initialize xgboost model
    # initialize xgboost model
    model = xgb.XGBClassifier(
        n_jobs=-1,
        max_depth=7
    )

    # fit model on training data
    model.fit(x_train, df_train.income.values)

    # predict on validation data
    valid_preds = model.predict_proba(x_valid)[:, 1]

    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)

    # print auc
    print(f"Fold = {fold}, AUC = {auc}")


if __name__ == "__main__":
    # read data
    df = pd.read_csv("../input/train_folds.csv")

    # create mean target encoded categories & munge data
    df = mean_target_encoding(df)

    for fold_ in range(5):
        run(df, fold_)


