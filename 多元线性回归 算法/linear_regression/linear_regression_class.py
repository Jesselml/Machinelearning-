import numpy as np

class LinearRegression():
    def __init__(self):
        self.coef_ = None   #系数
        self.intercept_ = None  #截距
        self.theta_ = None  #一组theta值 = 截距+系数

    def fit(self,X_train,y_train):
        X_b = np.hstack([np.ones(shape = (len(X_train),1)),X_train])
        self.theta_ = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        #使用正规方程求theta
        self.coef_ = self.theta_[1:]
        self.intercept_ = self.theta_[0]

        return self

    def fit_gd(self,X_train,y_train,eta = 0.01,n_iters = 1e4):
        def gradient_descent(X_b,y_train,theta,eta,n_iters=1e4,epsilon=1e-8):
            cur_iter = 0
            theta = initial_theta
            while (cur_iter < 5):
                gradient = dJ(X_b,y_train,theta)
                last_theta = theta
                theta = theta - eta * gradient
                if(abs(J(X_b,y_train,last_theta) - J(X_b,y_train,theta))<epsilon):
                    break
                cur_iter += 1
                
            return theta

        def dJ(X_b,y_train,theta):
            res = np.empty(len(theta))
            res[0] = np.sum(X_b.dot(theta) - y_train)  
            for g in range(1,len(theta)):
                res[g] = (X_b.dot(theta) - y_train).dot(X_b[:,g])

            return res*2/len(X_b) 

        def J(X_b,y_train,theta):
            return np.sum((X_b.dot(theta) - y_train)**2) / len(y_train) 

        X_b = np.hstack([np.ones(shape=(len(X_train),1)),X_train])
        initial_theta = np.zeros(X_b.shape[1])
        #初始化最初的theta为0
        self.theta_ = gradient_descent(X_b,y_train,initial_theta,eta,n_iters)
        #使用梯度下降方法拟合求得最终的theta
        self.intercept_ = self.theta_[0]
        self.coef_ = self.theta_[1:]

        return self

    def predict(self,X):
        return np.hstack([np.ones(shape = (X.shape[0],1)),X]).dot(self.theta_)

    def score(self,X_test,y_test):
        y_predict = self.predict(X_test)
        mean_squared_error = np.sum((y_test - y_predict)**2) / len(y_test)
        return 1 - mean_squared_error/np.var(y_test)

    def __repr__(self):
        return "LinearRegression()"

# X_train = np.array([[1,1],[2,2],[3,3],[4,4],[5,5],[6,6]])
# y_train = np.array([6,10,14,18,22,26])

# linear_regression = LinearRegression()
# linear_regression.fit_gd(X_train,y_train)

# y_predict = linear_regression.predict(np.array([[1,2],[2,3]]))
# print ("3 使用梯度下降法构建的模型预测，结果前5个值为：",y_predict)
