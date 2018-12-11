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

    def predict(self,X):
        return np.hstack([np.ones(shape = (X.shape[0],1)),X]).dot(self.theta_)

    def score(self,X_test,y_test):
        y_predict = self.predict(X_test)
        mean_squared_error = np.sum((y_test - y_predict)**2) / len(y_test)
        return 1 - mean_squared_error/np.var(y_test)

    def __repr__(self):
        return "LinearRegression()"