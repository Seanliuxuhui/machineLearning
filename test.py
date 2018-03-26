from playML_kNN import model_selection
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from playML_kNN import kNN
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, y_train, X_test, y_test = model_selection.train_test_split(X, y)
my_knn_classifier = kNN.KNNClassifier(k=3)
my_knn_classifier.fit(X_train, y_train)
print(my_knn_classifier.predict(X_test))