import numpy as np
from math import sqrt
from collections import Counter
from .metrics import accuracy_score


class KNNClassifier:
    def __init__(self, k):
        """initialize KNN Classifer """
        assert k >= 1, "k must be larger than 0"
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """Utilize X_train and y_train data to train the model"""
        assert X_train.shape[0] == y_train.shape[0], \
        "THe size of X_train data should match the size of y_train data"
        assert self.k <= X_train.shape[0], \
        "the size of X_train should always be larger than the defined k"
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        """Given X_predict and prompt users with the predict labels"""
        assert self._X_train is not None and self._y_train is not None, \
        "must fit before predict!"
        assert self._X_train.shape[1] == X_predict.shape[1], \
        "the feature number of X_predict must be equal to X_train"

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """Given single x return predicted value"""
        assert x.shape[0] == self._X_train.shape[1], \
        "the feature number of X must be equal to X_train"
        distances = []
        for x_train in self._X_train:
            distances.append(sqrt(sum((x_train - x)**2)))
        nearest = np.argsort(distances)
        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]

    def score(self, X_test, y_test):
        """test the accuracy of this model using the X_test and the y_test"""
        y_predict = self.predict(X_predict=X_test)

        return accuracy_score(y_test, y_predict)
    
    def __repr__(self):
        return "KNN(k=%d)" % self.k
