from __future__ import division
import numpy as np
import minimizers
import utils
import pdb


class linReg:

    # Q3 - One-vs-all Least Squares for multi-class classification
    def __init__(self, X, y, verbose=1, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.X = X
        self.y = y
        self.alpha = 1
        
        n, d = self.X.shape
        self.w = np.zeros(d)        
        #utils.check_gradient(self, self.X, self.y)

    def funObj(self, w, X, y):
        
        xwy = (X.dot(w) - y)
        f = 0.5 * xwy.T.dot(xwy)
        g = X.T.dot(xwy)

        return f, g

    def fit(self):

        (self.w, self.alpha, f, _) = minimizers.findMin(self.funObj, self.w, self.alpha,
                                         self.maxEvals,
                                         self.verbose,
                                         self.X,
                                         self.y)

    def oneGradientStep(self):

        (self.w, self.alpha, f, optTol) = minimizers.findMin(self.funObj, self.w, self.alpha,
                                         1,  # Max one eval
                                         self.verbose,
                                         self.X,
                                         self.y)
        return (self.w, f, optTol)

    def getParameters(self):
        return self.w

    def predict(self, X):
        w = self.w
        yhat = np.dot(X, w)
        return yhat


class linRegL2(linReg):

    def __init__(self, X, y, lammy, verbose=1, maxEvals=100):
        self.lammy = lammy
        self.verbose = verbose
        self.maxEvals = maxEvals

        self.X = X
        self.y = y
        self.alpha = 1
        
        n, d = self.X.shape
        self.w = np.zeros(d)        
        #utils.check_gradient(self, self.X, self.y)

    def funObj(self, w, X, y):
        
        xwy = (X.dot(w) - y)
        f = 0.5 * xwy.T.dot(xwy) + 0.5 * self.lammy * w.T.dot(w)
        g = X.T.dot(xwy) + self.lammy * w

        return f, g
