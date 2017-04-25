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

    def predict(self, X):

        n, d = X.shape

        yhats = {}
        yhat_total = np.zeros(n)

        for i in xrange(len(self.models)):
            yhats[i] = self.models[i].predict(X)
            yhat_total = yhat_total + yhats[i]

        return yhat_total / len(self.models)
