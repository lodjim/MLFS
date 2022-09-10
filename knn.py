import numpy as np
from collections import Counter
import sklearn

def euclidian_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))
class Knn:
    def __init__(self,k=5):
        self.k  = k

    def fit(self,X,y):
        self.X_train = X
        self.Y_train = y 

    def predict(self,X):
        predict_labels = self._predict(X)
        return np.array(predict_labels)

    def _predict(self,x):
        distance = [euclidian_distance(x,x_train) for x_train in self.X_train]
        k_indices = np.argsort(distance)[:self.k]
        k_nearest_labels = [self.Y_train[index] for index in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
