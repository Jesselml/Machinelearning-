import numpy as np
#导入数据
from sklearn import datasets
iris = datasets.load_iris()
X = iris["data"]
y = iris["target"]
X = X[y<2,:2]
y = y[y<2]

print (X.shape)
print (y.shape)

#数据标准化
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
standard_scaler.fit(X)
X = standard_scaler.transform(X)

#使用sklearn的线性SVM构建模型
from sklearn.svm import LinearSVC
svm = LinearSVC(C = 1000000)
svm.fit(X,y)
print (svm.coef_)
print (svm.intercept_)