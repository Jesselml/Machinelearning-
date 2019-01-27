import numpy as np

class PCA:
    def __init__(self, n_components):
        self.__n_components = n_components
        # components用来存放（n_components）行新生成的维度  W
        self.__components = None

    #1 拟合 - 寻找k个维度
    def fit(self, X, eta=0.01, n_iters=1e4):
        def demean(X):
            return X - np.mean(X, axis=0)

        def direction(w):
            return w/np.linalg.norm(w)

        def df(w, X):
            return X.T.dot(X.dot(w))*2/len(X)

        def f(w, X):
            return np.sum(X.dot(w)**2)/len(X)

        def gradient_ascent(X, w, eta, n_iters, epsilon=1e-8):
            #w = w + eta * df/dw
            w = direction(w)
            cur_iter = 1
            while cur_iter < n_iters:
                last_w = w
                w = w + eta * df(w, X)
                w = direction(w)
                if abs(f(w, X)-f(last_w, X)) < epsilon:
                    break
                cur_iter += 1
            return w

        X_pca = demean(X)
        self.__components = np.empty(
            (self.__n_components, X_pca.shape[1]))
        # 生成行向量w
        for i in range(self.__n_components):
            initial_w = np.random.random(X.shape[1])
            w = gradient_ascent(X_pca, initial_w, eta, n_iters)
            self.__components[i] = w
            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1)*w

        return self

    #2 转换 - 降维过程，将每个样本投射到找到的k个维度上去
    def transform(self,X):
        return X.dot(self.__components.T)
