import numpy as np
def train_test_split(X, y, split_ratio=0.2, seed=None):
    """split the given X and given y data into train dataset and test data """
    assert X.shape[0] == y.shape[0], \
    "the size of X must be equal to the size of y"
    assert 0.0 <= split_ratio <= 1.0, \
    "split_ratio must be valid"

    if seed:
        np.random.seed(seed)

    shuffled_indexes = np.random.permutation(len(X))
    test_size = int(len(X) * split_ratio)
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return  X_train, y_train, X_test, y_test