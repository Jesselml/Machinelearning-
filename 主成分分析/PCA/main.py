import time
# knn分类步骤
# 1 获取数据
# 2 训练测试数据集分离
# （数据归一化处理  若使用梯度下降法拟合，此处用pca故不归一化处理，而需要用PCA降维）
# 3 拟合knn分类模型
# 4 对测试集进行预测/对需要预测的数据进行预测
# 5 计算结果准确度

import numpy as np
# step1 - 获取数据
from sklearn import datasets
digits = datasets.load_digits()
X = digits['data']
y = digits['target']

# step2 -  训练测试数据集分离
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=666)

""" 一 不进行降维的数据训练的knn模型 """

# step3 - 用降维后的训练数据拟合knn分类模型
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
start1 = time.time()
knn.fit(X_train,y_train)
end1 = time.time()
# step4 - 用降维后的测试数据计算分类精确度
accuracy = knn.score(X_test,y_test)
print ("一 不进行降维的数据训练的knn模型,预测的精确度为:",accuracy,",拟合模型所需要的时间为:",end1-start1)


""" 二 用自己实现的PCA降维过的数据训练的knn模型(梯度上升) """
# step3 - 用PCA对数据进行降维
from PCA_class import PCA
pca = PCA(5)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)

# step4 - 用降维后的训练数据拟合knn分类模型
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
start2 = time.time()
knn.fit(X_train_reduction,y_train)
end2 = time.time()
# step5 - 用降维后的测试数据计算分类精确度
X_test_reduction = pca.transform(X_test)
accuracy = knn.score(X_test_reduction,y_test)

print ("二 用自己实现的PCA降维过的数据训练的knn模型,预测的精确度为:",accuracy,",拟合模型所需要的时间为:",end2-start2)


""" 三 用sklearn类的PCA实现的降维数据训练出的的knn模型(非梯度上升) """
# step3 - 用PCA对数据进行降维
from sklearn.decomposition import PCA
#这里参数可以传pca的成分/维度数  也可以传新生成的n个维度所能覆盖原始数据的百分比
pca2 = PCA(0.9)
pca2.fit(X_train)
print ("~当覆盖95%的数据的时候，生成的维度数:",pca2.n_components_)
print ("~生成这么多维度时，各维度覆盖数据比例:",pca2.explained_variance_ratio_)
X_train_reduction = pca2.transform(X_train)

# step4 - 用降维后的训练数据拟合knn分类模型
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
start3 = time.time()
knn.fit(X_train_reduction,y_train)
end3 = time.time()
# step5 - 用降维后的测试数据计算分类精确度
X_test = pca2.transform(X_test)
accuracy = knn.score(X_test,y_test)
print ("三 用sklearn类的PCA实现的降维数据训练出的的knn模型,预测的精确度为:",accuracy,",拟合模型所需要的时间为:",end3-start3)
