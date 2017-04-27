from __future__ import division
import sys
import argparse
import utils
import linear_model
import logistic_model
import global_model
import pdb

# Load Binary and Multi -class data
data = utils.load_dataset("logisticData")
XBin, yBin = data['X'], data['y']
XBinValid, yBinValid = data['Xvalid'], data['yvalid']

data = utils.load_dataset("multiData")
XMulti, yMulti = data['X'], data['y']
XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

if __name__ == "__main__":

    full = logistic_model.logReg(XBin, yBin, verbose=0, maxEvals=400)
    full.fit()
    
    model1 = logistic_model.logReg(XBin[0:124,:], yBin[0:124], verbose=0, maxEvals=400)
    model1.fit()

    model2 = logistic_model.logReg(XBin[125:250,:], yBin[125:250], verbose=0, maxEvals=400)
    model2.fit()

    model3 = logistic_model.logReg(XBin[251:375,:], yBin[251:375], verbose=0, maxEvals=400)
    model3.fit()

    model4 = logistic_model.logReg(XBin[376:500,:], yBin[376:500], verbose=0, maxEvals=400)
    model4.fit()

    global_model = global_model.globalModel()
    global_model.add_model(model1)
    global_model.add_model(model2)

    print("model1 Validation error %.3f" % 
        utils.classification_error(model1.predict(XBinValid), yBinValid))
    print("model2 Validation error %.3f" % 
        utils.classification_error(model2.predict(XBinValid), yBinValid))
    print("model3 Validation error %.3f" % 
        utils.classification_error(model3.predict(XBinValid), yBinValid))
    print("model4 Validation error %.3f" % 
        utils.classification_error(model4.predict(XBinValid), yBinValid))

    print("full Training error %.3f" % 
        utils.classification_error(full.predict(XBin), yBin))
    print("full Validation error %.3f" % 
        utils.classification_error(full.predict(XBinValid), yBinValid))

    print("global Validation error %.3f" % 
        utils.classification_error(global_model.predictAverage(XBinValid), yBinValid))

    global_model.fitWeightedAverage(XBinValid[0:249,:], yBinValid[0:249])

    print("global-weighted Validation error %.3f" % 
        utils.classification_error(global_model.predictWeightedAverage(XBinValid[250:500,:]), yBinValid[250:500]))
