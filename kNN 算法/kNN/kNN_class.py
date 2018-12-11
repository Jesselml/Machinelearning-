# This Python file uses the following encoding: utf-8
import numpy as np
from math import sqrt
from collections import Counter

""" 仿sklearn里的kNN """
class KNNClassifier:
    def __init__(self,k):
        self.k = k
        self._X_train = None
        self._y_train = None
    
    def fit(self,X_train,y_train):
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self,X_predict):
        y_predict = []
        for x_test in X_predict:
            y_predict.append(self._predict_each(x_test))   

        return np.array(y_predict)

    def _predict_each(self,x_test):
        distances = [sqrt(np.sum((x_train - x_test)**2)) for x_train in self._X_train]
        topK_index = np.argsort(distances)[:self.k]
        topK_result = [self._y_train[i] for i in topK_index]
        votes = Counter(topK_result)
        result = votes.most_common(1)[0][0]

        return result

    def __repr__(self):
        return "KNN(k=%d)" % self.k

