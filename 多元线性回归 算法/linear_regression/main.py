# -*- coding=utf-8 -*-

# 数据解释
# X_train - 用于训练模型的0.8的真实数据，0.8m*n
# y_train - 训练数据的结果，行向量
# X_test - 用于测试的0.2的真实数据，0.2m*n
# y_test - 测试数据的结果，行向量
# y_predict - 对测试数据的预测结果，行向量

import numpy as np

# step1 - 导入数据
from sklearn import datasets
house = datasets.load_boston()
# 506 * 13
X = house['data']
y = house['target']
# print (X.shape)
# print (y.shape)

# step2 - 训练测试数据集分离
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 66)

""" 一 sklearn多元线性回归模型(正规方程拟合) """
# step3 - 拟合多元线性回归模型
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X_train,y_train)

# print (linear_regression.coef_)  行向量
# print (linear_regression.intercept_)

# step4 - 进行预测
y_predict = linear_regression.predict(X_test)
print ("1 sklearn多元线性回归模型(正规方程拟合)预测，结果前5个值为：",y_predict[:5])

# step5 - 求准确度
R2 = linear_regression.score(X_test,y_test)
print ("1 sklearn多元线性回归模型(正规方程拟合)预测，R Square值为：",R2)

""" 二 仿sklearn类多元线性回归模型(正规方程拟合) """
# step3 - 拟合多元线性回归模型
from linear_regression_class import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X_train,y_train)

# step4 - 进行预测
y_predict = linear_regression.predict(X_test)
print ("2 仿sklearn类多元线性回归模型(正规方程拟合)预测，结果前5个值为：",y_predict[:5])

# step5 - 求准确度
R2 = linear_regression.score(X_test,y_test)
print ("2 仿sklearn类多元线性回归模型(正规方程拟合)预测，R Square值为：",R2)

""" 三 仿sklearn类多元线性回归模型(梯度下降拟合) """
# step3 - 数据进行归一化预处理
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
standard_scaler.fit(X_train)
X_train = standard_scaler.transform(X_train)

# step4 - 拟合多元线性回归模型
from linear_regression_class import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit_gd(X_train,y_train)

# step5 - 进行预测
y_predict = linear_regression.predict(X_test)
print ("3 仿sklearn类多元线性回归模型(梯度下降拟合)预测，结果前5个值为：",y_predict[:5])

# step6 - 求准确度
R2 = linear_regression.score(X_test,y_test)
print ("3 仿sklearn类多元线性回归模型(梯度下降拟合)预测，R Square值为：",R2)




