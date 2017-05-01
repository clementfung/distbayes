from __future__ import division
import numpy as np
import minimizers
import utils
import pdb
from numpy.linalg import norm


class globalModel:

    # Logistic Regression
    def __init__(self, logistic=False, verbose=1, maxEvals=400):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.models = []
        self.modelX = np.empty(0)
        self.weights = np.empty(0)
        self.logistic = logistic

    def add_model(self, model):
        self.models.append(model)

    def fit(self, theta, *args):

        print "Training global model."

        # Parameters of the Optimization
        optTol = 1e-2
        i = 0
        n, d = self.models[0].X.shape

        # Initial guess
        self.w = np.zeros(d)
        funEvals = 1

        while True:

            (delta, f_new, g) = self.models[i % len(self.models)].privateFun(theta, self.w, *args)
            funEvals += 1
            i += 1

            # Print progress
            if self.verbose > 0:
                print("%d - loss: %.3f" % (funEvals, f_new))
                print("%d - g_norm: %.3f" % (funEvals, norm(g)))

            # Update parameters
            self.w = self.w + delta

            # Test termination conditions
            optCond = norm(g, float('inf'))

            if optCond < optTol:
                if self.verbose:
                    print("Problem solved up to optimality tolerance %.3f" % optTol)
                break

            if funEvals >= self.maxEvals:
                if self.verbose:
                    print("Reached maximum number of function evaluations %d" % self.maxEvals)
                break

        print "Done fitting."
       
    def selectFeatures(self, minVotes):

        n, d = self.models[0].X.shape
        votes = np.zeros(d)

        for i in xrange(len(self.models)):
            votes += (self.models[i].w == 0).astype(int)

        return np.where(votes < minVotes)[0]

    def predict(self, X):
        w = self.w
        
        if self.logistic:
            yhat = np.sign(np.dot(X, w))
        else:
            yhat = np.dot(X, w)

        return yhat

    def predictAverage(self, X):
        n, d = X.shape
        yhats = {}
        yhat_total = np.zeros(n)

        # Aggregation function
        for i in xrange(len(self.models)):
            yhats[i] = self.models[i].predict(X)
            yhat_total = yhat_total + yhats[i]

        if self.logistic:
            yhat = np.sign(yhat_total)
        else:
            yhat = yhat_total / float(len(self.models))

        return yhat

    def fitWeightedAverage(self, X, y, epsilon=1):

        n, d = X.shape
        k = len(self.models)

        modelX = np.zeros(shape=(n, k))
        
        for i in xrange(k):
            modelX[:, i] = self.models[i].privatePredict(X, epsilon)

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

        if self.logistic:
            yhat = np.sign(np.dot(modelX, self.weights.T))
        else:
            yhat = np.dot(modelX, self.weights.T)

        return yhat

