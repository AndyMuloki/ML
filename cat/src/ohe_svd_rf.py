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

    ]