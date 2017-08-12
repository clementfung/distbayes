from sklearn import tree, ensemble, neighbors, svm, linear_model
import math
import numpy as np
import pickle

def GenGlobal(errors, sizes):

  errors = np.array(errors)
  sizes = np.array(sizes)
  errors = np.reshape(errors, (sizes.shape[0], sizes.shape[0]))

  weights = []

  sortedErrors = np.sort(errors, axis = 1)

  mydict = dict()

  rows = errors.shape[0]
  cols = errors.shape[1]

  for i in range(rows):

    firstValue = 1
    if(sortedErrors[i][0] == -np.inf):
      firstValue = 0

    for j in range(cols):
      key = mydict.get(str(sortedErrors[i][j]))
      if key == None:
        mydict[str(sortedErrors[i][j])] = firstValue
        firstValue += 1

    for j in range(cols):
      errors[i][j] = mydict[str(errors[i][j])]

    mydict.clear()

  for i in range(len(sizes)):
    currWeight = 0.0
    if sizes[i] != 0.0:
      for j in range(cols):
        if(errors[i][j] != 0):
          currWeight += math.pow(2, -errors[i][j])

    currWeight *= sizes[i]
    weights.append(currWeight)

  weights = np.array(weights)
  weights[weights < np.median(weights)] = 0.0

  weights = weights / np.sum(weights)

  return weights