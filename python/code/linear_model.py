from __future__ import division
import numpy as np
import minimizers
import utils
import pdb


class linReg:

    # Q3 - One-vs-all Least Squares for multi-class classification
    def __init__(self, X, y, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.X = X
        self.y = y
        self.alpha = 1

        n, self.d = self.X.shape
        self.w = np.zeros(self.d)

    # Reports the direct change to w, based on the given one.
    def privateFun(self, theta, ww):

        f, g = self.funObj(ww, self.X, self.y)

        alpha = 1
        gamma = 1e-4
        threshold = int(self.d * theta)

        # Line-search using quadratic interpolation to find an acceptable value of alpha
        gg = g.T.dot(g)

        while True:
            delta = - alpha * g
            w_new = ww + delta
            f_new, g_new = self.funObj(w_new, self.X, self.y)

            if f_new <= f - gamma * alpha * gg:
                break

            if self.verbose > 1:
                print("f_new: %.3f - f: %.3f - Backtracking..." % (f_new, f))
         
            # Update step size alpha
            alpha = (alpha**2) * gg/(2.*(f_new - f + alpha*gg))

        # Weird way to get NON top k values
        param_filter = np.argpartition(abs(delta), -threshold)[:self.d - threshold]
        delta[param_filter] = 0

        return (delta, f_new, g_new)

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

    def predict(self, X):
        w = self.w
        yhat = np.dot(X, w)
        return yhat

    def privatePredict(self, X, epsilon):
        n, _ = X.shape
        w = self.w
        yhat = np.dot(X, w)

        # TODO: Estimate the L1 Sensitivity
        sens = np.max(yhat) - np.min(yhat)
        
        y_private = yhat + utils.lap_noise(loc=0, scale=sens / epsilon, size=n)
        return y_private

class linRegL2(linReg):

    def __init__(self, X, y, lammy, verbose=0, maxEvals=100):
        self.lammy = lammy
        self.verbose = verbose
        self.maxEvals = maxEvals

        self.X = X
        self.y = y
        self.alpha = 1
        
        n, self.d = self.X.shape
        self.w = np.zeros(self.d)        

    def funObj(self, ww, X, y):

        xwy = (X.dot(ww) - y)
        f = 0.5 * xwy.T.dot(xwy) + 0.5 * self.lammy * ww.T.dot(ww)
        g = X.T.dot(xwy) + self.lammy * ww

        return f, g
