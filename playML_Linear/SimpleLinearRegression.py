import numpy as np
from .metrics import r2_score

class SimpleLinearRegression:

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """Use x_train and u_train data to train the simple linear regression model"""
        assert x_train.ndim == 1, \
        'Simple Linear Regression can only solve single feature training data '
        assert len(x_train) == len(y_train), \
        'the size of x_train should match the size of y_train'

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0
        d = 0.0

        for x, y in zip(x_train, y_train):
            num += (x_train - x_mean)* (y_train - y_mean)
            d += (x_train - x_mean) ** 2

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """Given accurate x_predict and returns the calcuated y_predict"""
        assert x_predict.nmin == 1, \
        'simple Linear Regression can only solve single feature training data'
        assert self.a_ is not None and self.b_ is not None, \
        'must fit before predict'
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x):
        return self.a_ * x + self.b_


class SimpleLinearRegression2:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """Use x_train and u_train data to train the simple linear regression model"""
        assert x_train.ndim == 1, \
            'Simple Linear Regression can only solve single feature training data '
        assert len(x_train) == len(y_train), \
            'the size of x_train should match the size of y_train'

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        self.a_ = (x_train - x_mean).dot(y_train - y_mean) /  (x_train - x_mean).dot(x_train - x_mean)
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """Given accurate x_predict and returns the calcuated y_predict"""
        assert x_predict.nmin == 1, \
            'simple Linear Regression can only solve single feature training data'
        assert self.a_ is not None and self.b_ is not None, \
            'must fit before predict'
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x):
        return self.a_ * x + self.b_

    def __repr__(self):
        return "SimpleLinearRegression()"
    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)




