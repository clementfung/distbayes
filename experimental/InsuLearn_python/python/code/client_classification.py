from sklearn import tree, ensemble, neighbors, svm, linear_model
import numpy as np
import pickle
import utils

X = 0
y = 0
Xt = 0
yt = 0
model = 0

def Read(trainSet, testSet):
	trainData = utils.load_dataset(trainSet)
	testData = utils.load_dataset(testSet)

	global X, y, Xt, yt
	
	X = trainData['X']
	y = trainData['y']
	Xt = testData['X']
	yt = testData['y']

	print " --- Local data updated."

def Train(modelType):
	global model

	if modelType == "DT":
		model = tree.DecisionTreeClassifier(max_depth=3)

	elif modelType == "RF":
		model = ensemble.RandomForestClassifier()

	elif modelType == "KNN":
		model = neighbors.KNeighborsClassifier()

	elif modelType == "SVM":
		model = svm.LinearSVC()

	elif modelType == "LR":
		model = linear_model.LogisticRegression()

	model = model.fit(X, y)
	yhat = model.predict(X)

	return {"model": pickle.dumps(model), "error": np.sum(yhat != y)/float(yhat.size), "size": float(X.shape[0])}

def TrainErrorLocal(model):
	currModel = pickle.loads(model)
	yhat = currModel.predict(X)
	return {"error": np.sum(yhat != y)/float(yhat.size), "size": float(X.shape[0])}

def TestErrorLocal(model):
	currModel = pickle.loads(model)
	yhat = currModel.predict(Xt)
	return {"error": np.sum(yhat != yt)/float(yhat.size), "size": float(Xt.shape[0])}

def TrainErrorGlobal(gmodel, weights):
	yhat = np.zeros(X.shape[0])

	for i in range(len(gmodel)):
		if weights[i] == 0.0:
			continue
		currModel = pickle.loads(gmodel[i])
		yhat += weights[i] * currModel.predict(X)

	yhat = np.round(yhat)

	return np.sum(yhat != y)/float(yhat.size)

def TestErrorGlobal(gmodel, weights):
	yhat = np.zeros(Xt.shape[0])

	for i in range(len(gmodel)):
		if weights[i] == 0.0:
			continue
		currModel = pickle.loads(gmodel[i])
		yhat += weights[i] * currModel.predict(Xt)

	yhat = np.round(yhat)

	return np.sum(yhat != yt)/float(yhat.size)