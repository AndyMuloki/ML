import pandas as pd

from sklearn import model_selection
from datasets import load_dataset

if __name__ == "__main__":
    # Load dataset from Hugging Face
    ds = load_dataset("scikit-learn/adult-census-income", split="train")

    # Convert to pandas DataFrame
    df = ds.to_pandas()

    # Create a new column called kfold and fill it with -1
    df["kfold"] = -1  

    # Randomize the rows of the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Fetch labels
    y = df['income'].values 

    # Initiate the StratifiedKFold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Fill the new kfold column
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=y)):
        df.loc[val_idx, 'kfold'] = fold

    # Save the new CSV with fold information
    df.to_csv("../input/train_folds.csv", index=False)
