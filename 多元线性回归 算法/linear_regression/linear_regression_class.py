import numpy as np

class LinearRegression():
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.theta_ = None

    def fit(self,X_train,y_train):
        X_b = np.hstack([np.ones(shape = (len(X_train),1)),X_train])
        self.theta_ = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
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
            for n in range(1,len(theta)):
                res[n] = (X_b.dot(theta) - y_train).dot(X_b[:,n])

            return res*2/len(X_b) 

        def J(X_b,y_train,theta):
            try:
                return np.sum((X_b.dot(theta) - y_train)**2) / len(y_train) 
            except:
                return float('inf')

        X_b = np.hstack([np.ones(shape=(len(X_train),1)),X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self.theta_ = gradient_descent(X_b,y_train,initial_theta,eta,n_iters)

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



# import numpy as np
# np.random.seed(666)
# x = 2 * np.random.random(size=100)
# y = x * 3. + 4. + np.random.normal(size=100)
# X = x.reshape(-1, 1)
# print (X[:20])
# print (y[:20])
# lin_reg = LinearRegression()
# lin_reg.fit_gd(X, y)

# print ("coef_",lin_reg.coef_)
# print ("intercept",lin_reg.intercept_)