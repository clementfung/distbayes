from __future__ import division
import numpy as np
import minimizers
import utils
import pdb


class globalModel:

    # Logistic Regression
    def __init__(self, verbose=1, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.models = []
        self.modelX = np.empty(0)
        self.weights = np.empty(0)

    def add_model(self, model):
        self.models.append(model)

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self, X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = minimizers.findMin(self.funObj, self.w,
                                         self.maxEvals,
                                         self.verbose,
                                         X, y)

    def predictAverage(self, X):
        n, d = X.shape
        yhats = {}
        yhat_total = np.zeros(n)

        # Aggregation function
        for i in xrange(len(self.models)):
            yhats[i] = self.models[i].predict(X)
            yhat_total = yhat_total + yhats[i]

        return yhat_total / len(self.models)

    def fitWeightedAverage(self, X, y):

        n, d = X.shape
        k = len(self.models)

        modelX = np.zeros(shape=(n, k))
        
        for i in xrange(k):
            modelX[:, i] = self.models[i].predict(X)

        A = np.dot(modelX.T, modelX)
        B = np.dot(modelX.T, y)
        
        self.modelX = modelX
        self.weights = np.linalg.solve(A, B)

    def predictWeightedAverage(self, X):
        
        n, d = X.shape
        k = len(self.models)

        modelX = np.zeros(shape=(n, k))
        
        for i in xrange(k):
            modelX[:, i] = self.models[i].predict(X)

        yhat = np.sign(np.dot(modelX, self.weights.T))

        return yhat

