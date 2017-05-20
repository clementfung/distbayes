from __future__ import division
import numpy as np
import minimizers
import utils
import pdb


class logReg:
    # Logistic Regression
    def __init__(self, X, y, verbose=1, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.X = X
        self.y = y
        self.alpha = 1
        
        n, self.d = self.X.shape
        self.w = np.zeros(self.d)        
        #utils.check_gradient(self, self.X, self.y)

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
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self):

        (self.w, self.alpha, f, _) = minimizers.findMin(self.funObj, self.w, self.alpha,
                                         self.maxEvals,
                                         self.verbose,
                                         self.X,
                                         self.y)

        print("Training error: %.3f" % utils.classification_error(self.predict(self.X), self.y))

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
        return np.sign(yhat)

    '''
    Original model
    def privatePredict(self, X, scale):
        _, d = X.shape
        w = self.w + utils.exp_noise(scale=scale, size=d)
        yhat = np.dot(X, w)
        return np.sign(yhat)
    '''

    def privatePredict(self, X, epsilon):
        nn, dd = X.shape
        w = self.w
        yhat = np.dot(X, w)

        # TODO: Estimate the L1 Sensitivity
        sens = 0.25 * dd * dd + 3 * dd

        return np.sign(yhat + utils.lap_noise(loc=0, scale=sens / epsilon, size=nn))

class logRegL2(logReg):

    def __init__(self, X, y, lammy, verbose=1, maxEvals=100):
        self.lammy = lammy
        self.verbose = verbose
        self.maxEvals = maxEvals

        self.X = X
        self.y = y
        self.alpha = 1

        n, self.d = self.X.shape
        self.w = np.zeros(self.d)
        #utils.check_gradient(self, self.X, self.y)

    def funObj(self, ww, X, y):
        yXw = y * X.dot(ww)

        # Calculate the function value
        f = np.sum(np.logaddexp(0, -yXw)) + 0.5 * self.lammy * ww.T.dot(ww)

        # Calculate the gradient value
        res = - y / np.exp(np.logaddexp(0, yXw))
        g = X.T.dot(res) + self.lammy * ww

        return f, g

    def privatePredict(self, X, epsilon):
        nn, dd = X.shape
        w = self.w
        yhat = np.dot(X, w)

        # TODO: Estimate the L1 Sensitivity
        # sens = 0.25 * dd * dd + 3 * dd
        sens = 2 / (nn * self.lammy)

        return np.sign(yhat + utils.lap_noise(loc=0, scale=sens / epsilon, size=nn))


class logRegL1(logReg):

    def __init__(self, X, y, lammy, verbose=1, maxEvals=100):
        self.lammy = lammy
        self.verbose = verbose
        self.maxEvals = maxEvals

        self.X = X
        self.y = y
        self.alpha = 1

        n, self.d = self.X.shape
        self.w = np.zeros(self.d)

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.logaddexp(0, -yXw))

        # Calculate the gradient value
        res = - y / np.exp(np.logaddexp(0, yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self):

        nn, dd = self.X.shape

        # Initial guess
        self.w = np.zeros(dd)
        (self.w, f) = minimizers.findMinL1(self.funObj,
                                        self.w,
                                        self.lammy,
                                        self.maxEvals,
                                        self.verbose,
                                        self.X, self.y)

# L0 Regularized Logistic Regression
class logRegL0(logReg):
    # this is class inheritance:
    # we "inherit" the funObj and predict methods from logReg
    # and we overwrite the __init__ and fit methods below.
    # Doing it this way avoids copy/pasting code. 
    # You can get rid of it and copy/paste
    # the code from logReg if that makes you feel more at ease.
    def __init__(self, X, y, lammy=1.0, verbose=1, maxEvals=400):
        self.verbose = verbose
        self.lammy = lammy
        self.maxEvals = maxEvals

        self.X = X
        self.y = y
        self.alpha = 1

        n, self.d = self.X.shape
        self.w = np.zeros(self.d)

    def fitSelected(self, selected):
        n, d = self.X.shape  
        w0 = np.zeros(self.d)
        minimize = lambda ind: minimizers.findMin(self.funObj, 
                                                  w0[ind], 
                                                  self.alpha,
                                                  self.maxEvals, 
                                                  self.verbose, 
                                                  self.X[:, ind], self.y)

        # re-train the model one last time using the selected features
        self.w = w0
        self.w[selected], _, _, _ = minimize(selected)

    def fit(self):
        n, d = self.X.shape  
        w0 = np.zeros(self.d)
        minimize = lambda ind: minimizers.findMin(self.funObj, 
                                                  w0[ind], 
                                                  self.alpha,
                                                  self.maxEvals, 
                                                  self.verbose, 
                                                  self.X[:, ind], self.y)
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
                temp_w, _, _, _ = minimize(sl)

                #pdb.set_trace()

                loss, _ = self.funObj(temp_w, self.X[:, sl], self.y)

                if loss < minLoss:
                    minLoss = loss
                    bestFeature = i

            selected.add(bestFeature)
        
        # re-train the model one last time using the selected features
        self.w = w0
        self.w[list(selected)], _, _, _ = minimize(list(selected)) 
