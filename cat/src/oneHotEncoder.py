import numpy as np
from sklearn import preprocessing
from scipy import sparse

# create random 1-d array with 1001 diff categories
example = np.random.randint(1000, size=1000000)

# initialize OneHotEncoder (ohe) from scikit-learn
# keep sparse = False to get dense array
ohe = preprocessing.OneHotEncoder(sparse_output=False)

# fit and transform with dense ohe
ohe_example = ohe.fit_transform(example.reshape(-1, 1))

# print size in bytes for dense array
print(f"Size of dense array: {ohe_example.nbytes}")

# initialize ohe from scikit-learn
# keep sparse = True tp get sparse array
ohe = preprocessing.OneHotEncoder(sparse_output=True)

# fit and transform data with sparse ohe
ohe_example = ohe.fit_transform(example.reshape(-1, 1))

# print size of this sparse matrix
print(f"Size of sparse array: {ohe_example.data.nbytes}")

full_size = (
    ohe_example.data.nbytes + 
    ohe_example.indptr.nbytes + ohe_example.indices.nbytes
)

print(f"Full size of sparse array: {full_size}")
