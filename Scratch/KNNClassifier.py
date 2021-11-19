from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score

class KNN_classifier:
    def __init__(self, X, y, k, dist_metric = 'euclidean'):
        self.X = X
        self.y = y
        self.k = k
        if dist_metric == 'euclidean':
            self.d = 2
        elif dist_metric == 'minkowski':
            self.d = 1
            
    def predict(self, X_test):
        y_pred = []
        for x_current in X_test:
            distances = []
            for i in range(len(self.X)):
                distances.append((np.linalg.norm(self.X[i] - x_current, ord = self.d), self.y[i]))
            distances.sort(key = lambda x: x[0])
            k_nearest_neighbours = [x[1] for x in distances[:self.k]]
            y_pred.append(max(k_nearest_neighbours))
        return y_pred

X = load_iris().data
y = load_iris().target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 10)

clf = KNN_classifier(X_train, y_train , 3, 'minkowski')
y_pred = clf.predict(X_test)
print(accuracy_score(y_test,y_pred)) # Accuracy score = 0.9736842105263158