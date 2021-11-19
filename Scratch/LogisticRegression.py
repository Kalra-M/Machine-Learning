import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class LogisticRegression:

    def sigmoid(self, z): 
        return 1 / (1 + np.e**(-z))
    
    def J(self, yh, y):
        return -1 * np.sum(y * np.log(yh) + (1 - y) * np.log(1 - yh)) / len(X)
        
    def fit(self, X, y, n_iterations, alpha = 0.09):
        costs = []

        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))
        w = np.random.rand(X.shape[1])
        for _ in range(n_iterations):
            yh = self.sigmoid(np.dot(X, w))
            costs.append(self.J(yh, y))
            w -= alpha * np.dot(X.T,  yh - y) / len(X)
        
        self.w = w
        self.costs = costs
    
    def predict(self, X):
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))
        z = np.dot(X, self.w)
        return [1 if i > 0.5 else 0 for i in self.sigmoid(z)]


data = make_blobs(n_samples=500, n_features=2, centers=2, random_state = 10)

X = data[0]
y = data[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 10)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

clf = LogisticRegression()
clf.fit(X_train, y_train, 1000)

y_pred = np.array(clf.predict(X_test))
print(f'Accuracy score = {accuracy_score(y_pred, y_test)}') # Accuracy score = 1.0