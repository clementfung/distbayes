from __future__ import division
import numpy as np


class leastSquaresClassifier:
    # Q3 - One-vs-all Least Squares for multi-class classification
    def __init__(self):
        pass

    def fit(self, X, y):
        n, d = X.shape    
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((d, self.n_classes))
        
        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            self.W[:, i] = np.linalg.lstsq(np.dot(X.T, X), np.dot(X.T, ytmp))[0]

    def predict(self, X):
        yhat = np.dot(X, self.W)

        return np.argmax(yhat, axis=1)

