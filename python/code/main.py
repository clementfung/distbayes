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

cut1 = int(XBin.shape[0] * 0.25)
cut2 = int(XBin.shape[0] * 0.5)
cut3 = int(XBin.shape[0] * 0.75)
cut4 = XBin.shape[0]

cutVal = int(XBinValid.shape[0] * 0.5)
cutVal2 = XBinValid.shape[0]

if __name__ == "__main__":

    full = logistic_model.logRegL2(XBin, yBin, lammy=1, verbose=1, maxEvals=400)
    full.fit()

    model1 = logistic_model.logRegL2(XBin[0:cut1,:], yBin[0:cut1], lammy=1, verbose=0, maxEvals=400)
    model1.fit()

    model2 = logistic_model.logRegL2(XBin[cut1+1:cut2,:], yBin[cut1+1:cut2], lammy=1, verbose=0, maxEvals=400)
    model2.fit()

    model3 = logistic_model.logRegL2(XBin[cut2+1:cut3,:], yBin[cut2+1:cut3], lammy=1, verbose=0, maxEvals=400)
    model3.fit()

    model4 = logistic_model.logRegL2(
        XBin[cut3+1:cut4,:], yBin[cut3+1:cut4], lammy=1, verbose=0, maxEvals=400)
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

    global_model.fitWeightedAverage(XBinValid[0:cutVal,:], yBinValid[0:cutVal])

    print("global-weighted Validation error %.3f" % 
        utils.classification_error(global_model.predictWeightedAverage(
            XBinValid[cutVal+1:cutVal2,:], Logistic=True), yBinValid[cutVal+1:cutVal2]))

    pdb.set_trace()
