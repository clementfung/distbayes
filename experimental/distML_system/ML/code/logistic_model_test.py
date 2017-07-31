from __future__ import division
import numpy as np
import utils
import pdb

Xtest = utils.load_dataset('logTest')['X']
ytest = utils.load_dataset('logTest')['y']

def test(ww):
	ww = np.array(ww)
	yhat = np.sign(np.dot(Xtest, ww))
	error = np.sum(yhat!=ytest) / float(yhat.size)
	print error
	return error