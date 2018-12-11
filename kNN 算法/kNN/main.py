# -*- coding=utf-8 -*-

# 数据解释
# X_train - 用于训练模型的0.8的真实数据
# y_train - 训练数据的结果，纵向向量
# X_test - 用于测试的0.2的真实数据
# y_test - 测试数据的结果，纵向向量
# x - 希望预测的某个数据

import numpy as np
from sklearn import datasets

# step-1 读取数据
digits = datasets.load_digits()
""" 
数据集 1797 * 64
Attribute: [0 1 2 ........ 16]     digits['data']
Target: [0 1 2 3 4 5 6 7 8 9]      digits['target']
"""
X = digits['data']
y = digits['target']
k = 5

#print (X[:10])
#print (y)
""" 
# 初始化训练集数据
raw_data_X = [[3.393533211, 2.331273381],
              [3.110073483, 1.781539638],
              [1.343808831, 3.368360954],
              [3.582294042, 4.679179110],
              [2.280362439, 2.866990263],
              [7.423436942, 4.696522875],
              [5.745051997, 3.533989803],
              [9.172168622, 2.511101045],
              [7.792783481, 3.424088941],
              [7.939820817, 0.791637231]
             ]

raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
X = np.array(raw_data_X)
y = np.array(raw_data_y)
k = 5 """

# step-2 训练测试数据集分离
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 66)

# step-3 数据集归一化处理
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
standard_scaler.fit(X_train)
X_train = standard_scaler.transform(X_train)
X_test = standard_scaler.transform(X_test)

""" 1 使用sklearn里的kNN """
# step-4 构建模型
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

kNN_classifier = KNeighborsClassifier(k)
kNN_classifier.fit(X_train,y_train)

# step-5 模型精确度计算
y_predict = kNN_classifier.predict(X_test)

# 精确度计算方法二 使用sklearn.metrics下的accuracy_score函数
# accuracy = accuracy_score(y_test,y_predict)
# 精确度计算方法三 直接使用sklearn.neighbors下的KNeighborsClassifier类下的score方法（无需预测测试值）
accuracy = kNN_classifier.score(X_test,y_test)
print ("3.使用sklearn里的kNN,传入测试数据集，预测的结果是 ",accuracy)

# step-6 进行预测
# x = np.array([2.093607318, 1.365731514])
# y_predict = kNN_classifier.predict(x.reshape(1,-1))
# print "3.使用sklearn里的kNN,传入",x,"预测的结果是: ", y_predict 


""" 2 使用仿sklearn的kNN类 """
from kNN_class import KNNClassifier

kNN_classifier = KNNClassifier(k)
kNN_classifier.fit(X_train,y_train)

#预测测试数据集
y_predict = kNN_classifier.predict(X_test)

# 精确度计算方法一 直接计算两个列表的相似量/总量
accuracy =  sum(y_predict == y_test)/len(y_test)
print ("1.使用仿sklearn里的kNN类,传入测试数据集，预测的结果的准确度是 ",accuracy )

# #传入不同的需要判断的值
# x = np.array([9.093607318, 2.365731514])
# y_predict = kNN_classifier.predict(x.reshape(1,-1))
# print "1.使用仿sklearn的kNN类,传入",x,"预测的结果是: ", y_predict 

""" 3 使用kNN函数 """
from kNN_function import kNN_classify

# #传入不同的需要判断的值
# x = np.array([2.093607318, 1.365731514])
# y_predict = kNN_classify(k,X_train,y_train,x)
# y_predict
# print "2.使用kNN函数,传入",x,"预测的结果是: ", y_predict 

