import pandas as pd

from scipy import sparse
from sklearn import decomposition
from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing

# one hot encoding - singular value decomposition - random forest

def run(fold):
    df = pd.read_csv("../input/train_folds.csv")

    features = [
        f for f in df.columns if f not in ("id", "target", "kfold")
    ]

    # filling all NaN values with NONE

    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # initialize ohe from scikit-learn
    ohe = preprocessing.OneHotEncoder()

    # fit ohe on training + validation features
    full_data = pd.concat(
        [df_train[features], df_valid[features]],
        axis = 0 
    )
    ohe.fit(full_data[features])

    # transform trainig data
    x_train = ohe.transform(df_train[features])

    # transform validation data
    x_valid = ohe.transform(df_valid[features])

    # initialize truncated SVD
    svd = decomposition.TruncatedSVD(n_components=120)

    # fit svd on full sparse training data
    full_sparse = sparse.vstack((x_train, x_valid))
    svd.fit(full_sparse)

    # transforom sparse training data
    x_train = svd.transform(x_train)

    # transform sparse validation data
    x_valid = svd.transform(x_valid)

    # initialize random forest model
    model = ensemble.RandomForestClassifier(n_jobs=-1)

    # fit model on training data (ohe)
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