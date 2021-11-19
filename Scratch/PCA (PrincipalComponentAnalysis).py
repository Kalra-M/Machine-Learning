import numpy as np

class PCA:
    def fit(self, X, n_components = 2):
        X_new = X - np.mean(X , axis = 0) # Can divide by std. However, scikit-learn does not divide by std.
        cov_matrix = np.cov(X_new , rowvar = False)
        eigen_values , eigen_vectors = np.linalg.eigh(cov_matrix)
        sorted_index = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[sorted_index][:n_components]
        eigen_vectors = eigen_vectors[:,sorted_index][:, :n_components]
        self.eigen_vectors = eigen_vectors.T
        
    def transform(self, X):
        X_new = X - np.mean(X , axis = 0)
        return np.dot(self.eigen_vectors , X_new.T).T

X = np.array([[3,2,3], [0, 3, 2], [3, 0, 3]])
pca = PCA()
pca.fit(X)
transformed = pca.transform(X)
print(f'Transformed = {transformed}')
print(f'Components = {pca.eigen_vectors}')

# Transformed = [[-0.62017367  0.91520863]
#  [ 2.48069469 -0.26148818]
#  [-1.86052102 -0.65372045]]
# Components = [[-0.74420841  0.62017367 -0.24806947]
#  [ 0.58834841  0.78446454  0.19611614]]
