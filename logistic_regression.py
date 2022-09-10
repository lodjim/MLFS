import numpy as np

class LogisticRegression:
    def __init__(self,lr=0.001,n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
    def fit(self,X,y):
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iter):
            linear_model_output = np.dot(self.weights,X) + self.bias
            y_hat = self._sigmoid(linear_model_output)
            dw = (1/n_samples)*np.dot(X.T,(y_hat-y))
            db = (1/n_samples)*np.sum(y_hat-y)
            self.weights -=  self.lr*dw
            self.bias -= self.lr*db 
    def predict(self,X):
        linear_model_output = np.dot(self.weights,X) + self.bias
        y_predict = self._sigmoid(linear_model_output)
        return y_predict
    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))


    


    