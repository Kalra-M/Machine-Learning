import numpy as np

class LinearRegression:
    
    def J(self, yh, y):
        return 0.5 / len(X) * np.sum((yh - y)**2)
    
    def fit(self, X, y, n_iterations, alpha = 0.09):
        costs = []
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))
        w = np.random.rand(X.shape[1])
        for _ in range(n_iterations):
            yh = np.dot(X, w)
            costs.append(self.J(yh, y))
            w -= alpha * np.dot(X.T,  yh - y) / len(X)
        self.w = w
        self.costs = costs
        
    def predict(self, X):
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))
        return np.dot(X, self.w)

X = np.linspace(0,5,200)
y = 2 * X + 8 + np.random.normal(0,3.5,np.size(X))
X = X.reshape((200,1))

reg = LinearRegression()
reg.fit(X, y, 1000)

print(f'Weights = {reg.w}') # Weights = [8.17642798 1.83361399]
test = np.array([10])
test = test.reshape((1,1))
print(f'y(10) = {reg.predict(test)}') # y(10) = [26.51256783]