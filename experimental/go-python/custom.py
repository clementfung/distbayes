import numpy as np

test = 5

def helloworld():
	print 'hello, world!'
	return np.ones(5)

def incr():
	global test
	test += 1
	return test