# -*- coding=utf-8 -*-

# 多元线性回归步骤
# 1 获取数据
# 2 训练测试数据集分离
# （数据归一化处理  若使用梯度下降法拟合）
# 3 拟合线性回归模型
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
house = datasets.load_boston()
# 506 * 13
X = house['data']
y = house['target']
# print (X.shape)
# print (y.shape)

""" # step1 - 导入虚拟数据 （梯度下降和正规方程均可产生正确结果）
m = 100000
x = np.random.normal(size=m)
X = x.reshape(-1,1)
y = 4.*x + 3. + np.random.normal(0, 3, size=m) """

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

""" 三 sklearn类多元线性回归模型(梯度下降拟合) """
# step3 - 数据进行归一化预处理
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
standard_scaler.fit(X_train)
X_train = standard_scaler.transform(X_train)

# step4 - 拟合多元线性回归模型
from sklearn.linear_model import SGDRegressor
linear_regression = SGDRegressor()
linear_regression.fit(X_train,y_train)

# step5 - 进行预测
y_predict = linear_regression.predict(X_test)
print ("3 sklearn类多元线性回归模型(梯度下降拟合)预测，结果前5个值为：",y_predict[:5])

# step6 - 求准确度
R2 = linear_regression.score(X_test,y_test)
print ("3 sklearn类多元线性回归模型(梯度下降拟合)预测，R Square值为：",R2)

""" 四 仿sklearn类多元线性回归模型(梯度下降拟合) """
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
print ("4 仿sklearn类多元线性回归模型(梯度下降拟合)预测，结果前5个值为：",y_predict[:5])

# step6 - 求准确度
R2 = linear_regression.score(X_test,y_test)
print ("4 仿sklearn类多元线性回归模型(梯度下降拟合)预测，R Square值为：",R2)




