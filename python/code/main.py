from __future__ import division
import sys
import argparse
import utils
import linear_model
import logistic_model
import global_model

# Load Binary and Multi -class data
data = utils.load_dataset("logisticData")
XBin, yBin = data['X'], data['y']
XBinValid, yBinValid = data['Xvalid'], data['yvalid']

data = utils.load_dataset("multiData")
XMulti, yMulti = data['X'], data['y']
XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

if __name__ == "__main__":

    model1 = logistic_model.logReg(XBin[1:250, :], yBin[1:250], verbose=1, maxEvals=400)
    model1.fit()

    model2 = logistic_model.logReg(XBin[251:500, :], yBin[251:500], verbose=1, maxEvals=400)
    model2.fit()

    gm = global_model.globalModel()
    gm.add_model(model1)
    gm.add_model(model2)
    gm.predict(XBinValid)

    print("logRegL1 Validation error %.3f" % utils.classification_error(model1.predict(XBinValid), yBinValid))
    print("logRegL1 Validation error %.3f" % utils.classification_error(model2.predict(XBinValid), yBinValid))
    print("global Validation error %.3f" % utils.classification_error(gm.predict(XBinValid), yBinValid))
    print("# nonZeros: %d" % (model1.w != 0).sum())
