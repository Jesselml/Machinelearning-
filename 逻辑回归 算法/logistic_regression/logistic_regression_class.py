import numpy as np

class LogisticRegression():
    def __init__(self):
        self.coef_ = None   #系数
        self.intercept_ = None  #截距
        self.theta_ = None  #一组theta值 = 截距+系数

    def sigmod(self,z):
        return 1/(1+np.exp(-z))

    def fit_gd(self,X_train,y_train,eta = 0.01,n_iters = 1e4):
        def gradient_descent(X_b,y_train,initial_theta,eta,n_iters=1e4,epsilon=1e-8):
            cur_iter = 0
            theta = initial_theta
            while (cur_iter < n_iters):
                gradient = dJ(X_b,y_train,theta)
                last_theta = theta
                theta = theta - eta * gradient
                if(abs(J(X_b,y_train,last_theta) - J(X_b,y_train,theta))<epsilon):
                    break
                cur_iter += 1
                
            return theta

        def dJ(X_b,y_train,theta):
            return X_b.T.dot(self.sigmod(X_b.dot(theta))-y_train) / len(y_train)

        def J(X_b,y_train,theta):
            h_theta = self.sigmod(X_b.dot(theta))
            return -np.sum(y_train*np.log(h_theta)+(1-y_train)*np.log(1-h_theta)) / (len(y_train))

        X_b = np.hstack([np.ones(shape=(len(X_train),1)),X_train])
        initial_theta = np.zeros(X_b.shape[1])
        #初始化最初的theta为0
        self.theta_ = gradient_descent(X_b,y_train,initial_theta,eta,n_iters)
        #使用梯度下降方法拟合求得最终的theta
        self.intercept_ = self.theta_[0]
        self.coef_ = self.theta_[1:]

        return self

    def predict_prob(self,X):
        return self.sigmod(np.hstack([np.ones(shape = (X.shape[0],1)),X]).dot(self.theta_))

    def predict(self,X):
        proba = self.predict_prob(X)
        return np.array(proba >= 0.5, dtype='int')

    def score(self,X_test,y_test):
        y_predict = self.predict(X_test)
        return np.sum(y_predict==y_test)/len(y_test)

    def __repr__(self):
        return "LogisticRegression()"



