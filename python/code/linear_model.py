from __future__ import division
import numpy as np
import minimizers
import utils
import pdb

class logReg:
    # Logistic Regression
    def __init__(self, verbose=1, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape    

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = minimizers.findMin(self.funObj, self.w, 
                                         self.maxEvals, 
                                         self.verbose,
                                         X, y)
    def predict(self, X):
        w = self.w
        yhat = np.dot(X, w)

        return np.sign(yhat)

class logRegL2(logReg):

    def __init__(self, lammy, verbose=1, maxEvals=100):
        self.lammy = lammy
        self.verbose = verbose
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw))) + 0.5 * self.lammy * w.T.dot(w)

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res) + self.lammy * w

        return f, g

class logRegL1(logReg):

    def __init__(self, lammy, verbose=1, maxEvals=100):
        self.lammy = lammy
        self.verbose = verbose
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape    

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = minimizers.findMinL1(self.funObj, 
                                        self.w, 
                                        self.lammy,
                                        self.maxEvals, 
                                        self.verbose,
                                        X, y)


# L0 Regularized Logistic Regression
class logRegL0(logReg): 
    # this is class inheritance:
    # we "inherit" the funObj and predict methods from logReg
    # and we overwrite the __init__ and fit methods below.
    # Doing it this way avoids copy/pasting code. 
    # You can get rid of it and copy/paste
    # the code from logReg if that makes you feel more at ease.
    def __init__(self, lammy=1.0, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.lammy = lammy
        self.maxEvals = maxEvals

    def fit(self, X, y):
        n, d = X.shape  
        w0 = np.zeros(d)
        minimize = lambda ind: minimizers.findMin(self.funObj, 
                                                  w0[ind], 
                                                  self.maxEvals, 0, 
                                                  X[:, ind], y)
        selected = set()
        selected.add(0) # always include the bias variable 
        minLoss = np.inf
        oldLoss = 0
        bestFeature = -1

        while minLoss != oldLoss:
            oldLoss = minLoss
            if self.verbose > 1:
                print("Epoch %d " % len(selected))
                print("Selected feature: %d" % (bestFeature))
                print("Min Loss: %.3f\n" % minLoss)

            for i in range(d):
                if i in selected:
                    continue
                
                selected_new = selected | {i} # add "i" to the set
                # TODO: Fit the model with 'i' added to the features,
                # then compute the score and update the minScore/minInd

                # TODO: Fit the model with 'i' added to the features,
                # then compute the score and update the minScore/minInd
                sl = list(selected_new)
                temp_w, _ = minimize(sl)

                #pdb.set_trace()

                loss, _ = self.funObj(temp_w, X[:, sl], y)

                if loss < minLoss:
                    minLoss = loss
                    bestFeature = i

            selected.add(bestFeature)
        
        # re-train the model one last time using the selected features
        self.w = w0
        self.w[list(selected)], _ = minimize(list(selected))       


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

