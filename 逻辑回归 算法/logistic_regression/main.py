# -*- coding=utf-8 -*-

# 逻辑回归步骤
# 1 获取数据
# 2 训练测试数据集分离
# 3 拟合逻辑回归模型
# 4 对测试集进行预测/对需要预测的数据进行预测
# 5 计算结果准确度

# 数据解释
# X_train - 用于训练模型的0.8的真实数据，0.8m*n
# y_train - 训练数据的结果，行向量
# X_test - 用于测试的0.2的真实数据，0.2m*n
# y_test - 测试数据的结果，行向量
# y_predict - 对测试数据的预测结果，行向量

import numpy as np

# step1 - 导入真实数据 （以下数据使用梯度下降会产生错误结果）
from sklearn import datasets
iris = datasets.load_iris()
X = iris['data']
y = iris['target']
""" 
数据集 150 * 4
Attribute(iris['data']): [sepal length,sepal width,petal length,petal width]     
Class(iris['target']): [Setosa,Versicolour,Virginica]
"""
#因为只考虑二分类问题，只保留前两个类的数据
X = X[y<2,:2]
y = y[y<2]
""" 
数据集 100 * 4
Attribute(iris['data']): [sepal length,sepal width,petal length,petal width]     
Class(iris['target']): [Setosa,Versicolour]
"""

# step2 - 训练测试数据集分离
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 66)

# """ 一 sklearn类逻辑回归模型(梯度下降拟合) """
# # step3 - 数据进行归一化预处理
# from sklearn.preprocessing import StandardScaler
# standard_scaler = StandardScaler()
# standard_scaler.fit(X_train)
# X_train = standard_scaler.transform(X_train)

# # step4 - 拟合多元线性回归模型
# from sklearn.logistic_model import SGDRegressor
# logistic_regression = SGDRegressor()
# logistic_regression.fit(X_train,y_train)

# # step5 - 进行预测
# y_predict = logistic_regression.predict(X_test)
# print ("1 sklearn类逻辑回归模型(梯度下降拟合)预测，结果为：",y_predict)

# # step6 - 求准确度
# accuracy = logistic_regression.score(X_test,y_test)
# print ("1 sklearn类逻辑回归模型(梯度下降拟合)预测，精确度为：",accuracy)

""" 二 仿sklearn类逻辑回归模型(梯度下降拟合) """
# # step3 - 数据进行归一化预处理
# from sklearn.preprocessing import StandardScaler
# standard_scaler = StandardScaler()
# standard_scaler.fit(X_train)
# X_train = standard_scaler.transform(X_train)

# step4 - 拟合多元线性回归模型
from logistic_regression_class import LogisticRegression
logistic_regression = LogisticRegression()
logistic_regression.fit_gd(X_train,y_train)

# step5 - 进行预测
y_predict = logistic_regression.predict(X_test)
print ("2 数据实际结果为：",y_predict)
print ("2 仿sklearn类逻辑回归模型(梯度下降拟合)预测，结果为：",y_predict)

# step6 - 求准确度
accuracy = logistic_regression.score(X_test,y_test)
print ("2 仿sklearn类逻辑回归模型(梯度下降拟合)预测，精确度为：",accuracy)




