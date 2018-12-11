# This Python file uses the following encoding: utf-8
import numpy as np
from math import sqrt
from collections import Counter

""" 自己创建的kNN算法 """

#X_train 2*2 matrix
#y_train vector
def kNN_classify(k,X_train,y_train,x):
    distances = [sqrt(np.sum((x_train-x)**2)) for x_train in X_train]
    topK_index = np.argsort(distances)[:k]
    topK_resut = [y_train[i] for i in topK_index]

    votes = Counter(topK_resut)

    return votes.most_common(1)[0][0]
