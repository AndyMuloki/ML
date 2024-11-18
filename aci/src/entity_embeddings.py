import os
import gc
import joblib

import pandas as pd
import numpy as np

from sklearn import metrics, preprocessing

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras import utils




def create_model(data, catcols):
    """
    Returns a compiled tf.keras model for entity embeddings
    :param data: pandas dataframe
    :param catcols: list of categorical column names
    :return: compiled tf.keras model
    """
    # Initialize list of inputs for embeddings
    inputs = []

    # Initialize list of outputs for embeddings
    outputs = []

    # Loop over all categorical columns
    for c in catcols:
        # Find the number of unique values in the column
        num_unique_values = int(data[c].nunique())
        embed_dim = int(min(np.ceil((num_unique_values) / 2), 50))

        # Simple keras input layer with size 1
        inp = layers.Input(shape=(1,), name=f"input_{c}")

        # Add embedding layer to raw input
        # Embedding size is always 1 more than unique values in input
        out = layers.Embedding(
            num_unique_values + 1, embed_dim, name=f"embedding_{c}"
        )(inp)

        out = layers.SpatialDropout1D(0.3)(out)
        out = layers.Reshape(target_shape=(embed_dim,))(out)

        # Add input and output to respective lists
        inputs.append(inp)
        outputs.append(out)

    # Concatenate all embedding outputs
    x = layers.Concatenate()(outputs)

    # Add a batch normalization layer
    x = layers.BatchNormalization()(x)

    # Dense layers with dropout
    x = layers.Dense(300, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(300, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    # Output layer using softmax
    y = layers.Dense(2, activation="softmax")(x)

    # Create and compile the final model
    model = Model(inputs=inputs, outputs=y)
    model.compile(loss="binary_crossentropy", optimizer="adam")
    return model


def run(fold):
    df = pd.read_csv("../input/train_folds.csv")

    # map targets
    target_mapping = {"<=50K": 0, ">50K": 1}
    df["income"] = df["income"].map(target_mapping)


    # all columns are features except kfold and income
    features = [
        f for f in df.columns if f not in ("kfold", "income", "id")
    ]

    for col in features:
        df[col] = df[col].fillna("NONE").astype(str)

    # encode all features with label encoder individually
    for feat in features:
        lbl = preprocessing.LabelEncoder()
        df[feat] = lbl.fit_transform(df[feat].values)

    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    
    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # create tf.keras model
    model = create_model(df, features)    

    xtrain = [df_train[feat].values for feat in features]
    xvalid = [df_valid[feat].values for feat in features]

    # fetch target columns
    ytrain = df_train['income'].values
    yvalid = df_valid['income'].values

    # convert target columns to categories
    ytrain_cat = utils.to_categorical(ytrain)
    yvalid_cat = utils.to_categorical(yvalid)

    # fit model
    model.fit(
        xtrain,
        ytrain_cat,
        validation_data=(xvalid, yvalid_cat),
        verbose=1,
        batch_size=1024,
        epochs=3
    )

    # predict on validation data
    valid_preds = model.predict(xvalid)[:, 1]

    # print roc auc score
    # auc = metrics.roc_auc_score(yvalid, valid_preds)
    print(metrics.roc_auc_score(yvalid, valid_preds))

    # print auc
    # print(f"Fold = {fold}, AUC = {auc}")


    # clear session (free GPU Memory)
    K.clear_session()


if __name__ == "__main__":
    run(0)
    run(1)
    run(2)
    run(3)
    run(4)


    