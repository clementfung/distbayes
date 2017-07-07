from __future__ import division
import numpy as np
import utils
import pdb

verbose = 0
maxEvals = 100
X = utils.load_dataset("slices")['X']
y = utils.load_dataset("slices")['y']
alpha = 1
d = X.shape[1]
w = np.zeros(d)
lammy = 0.1

print X.shape[0]

def changeVerbose(Verbose=0):
    global verbose
    verbose = Verbose

def changeMaxEvals(MaxEvals=100):
    global maxEvals
    maxEvals = MaxEvals

def changeLammy(Lammy=0.1):
    global lammy
    lammy = Lammy

def privateFun(theta):
    global w

    f, g = funObj(w, X, y)

    alpha = 1
    gamma = 1e-4
    threshold = int(d * theta)

    gg = g.T.dot(g)

    while True:
        delta = - alpha * g
        w_new = w + delta
        f_new, g_new = funObj(w_new, X, y)

        if f_new <= f - gamma * alpha * gg:
            break

        if verbose > 1:
            print("f_new: %.3f - f: %.3f - Backtracking..." % (f_new, f))
         
        # Update step size alpha
        alpha = (alpha**2) * gg/(2.*(f_new - f + alpha*gg))

    # Weird way to get NON top k values
    #param_filter = np.argpartition(abs(delta), -threshold)[:d - threshold]
    #delta[param_filter] = 0

    w = w_new
    return delta

def funObj(ww, X, y):
        
    xwy = (X.dot(ww) - y)
    f = 0.5 * xwy.T.dot(xwy) + 0.5 * lammy * ww.T.dot(ww)
    g = X.T.dot(xwy) + lammy * ww

    return f, g

def predict(X):
    yhat = np.dot(X, w)
    return yhat

def privatePredict(X, epsilon):
    nn, dd = X.shape
    yhat = np.dot(X, w)

    # TODO: Estimate the L1 Sensitivity in a better way
    sens = (dd * dd + 2 * dd + 1) * 2

'''def testMultArgs(argLong, argFloatArray):
    print 'before first'
    print argLong
    print 'after first'

    print 'before second'
    print argFloatArray
    print 'after second'

    print 'before third'
    print np.size(argFloatArray)
    print 'after third'
    '''