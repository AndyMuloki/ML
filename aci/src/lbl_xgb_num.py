import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn import preprocessing


def run(fold):
    df = pd.read_csv("../input/train_folds.csv")

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
    ]

    # label encoding non-numerical features
    for col in features:
        if col not in num_cols:  # Skip label encoding for numerical columns
            df[col] = df[col].fillna("NONE").astype(str)
            lbl = preprocessing.LabelEncoder()
            df[col] = lbl.fit_transform(df[col])
    
    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    
    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # get training data
    x_train = df_train[features].values

    # get validation data
    x_valid = df_valid[features].values

    # initialize xgboost model
    model = xgb.XGBClassifier(
        n_jobs=-1
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
    for fold_ in range(5):
        run(fold_)
