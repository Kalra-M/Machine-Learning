# from sklearn.datasets import make_blobs
import numpy as np

class KMeans_clustering:
    def __init__(self, k = 2):
        self.k = k
    
    def run(self, X, iterations):
        self.centroids = []
        
        for _ in range(self.k):
            centroid = X[np.random.choice(range(len(X)))]
            self.centroids.append(centroid)
            
        for _ in range(iterations):
            self.classes = []
            for x_current in X:
                distances = [np.linalg.norm(x_current - centroid) for centroid in self.centroids]
                self.classes.append(distances.index(min(distances)))
                            
            for label in range(self.k):
                self.centroids[label] = np.mean([x_current for x_current, y in zip(X, self.classes) if y == label], axis=0)
                
        return self.classes

# n_clusters = 4
# X = make_blobs(n_samples = 400, n_features = 2, centers = n_clusters)
# ob = KMeans_clustering(4)
# labels = ob.run(X[0], 500)
