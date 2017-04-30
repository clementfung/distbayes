from __future__ import division
import utils
import linear_model
import global_model
import pdb

# Load Binary and Multi -class data
data = utils.load_dataset("slices")
XBin, yBin = data['X'], data['y']
XBinValid, yBinValid = data['Xvalid'], data['yvalid']

cut1 = int(XBin.shape[0] * 0.25)
cut2 = int(XBin.shape[0] * 0.50)
cut3 = int(XBin.shape[0] * 0.75)
cut4 = XBin.shape[0]

cutVal = int(XBinValid.shape[0] * 0.5)
cutVal2 = XBinValid.shape[0]

if __name__ == "__main__":

    ## LOCAL MODELS

    model1 = linear_model.linRegL2(XBin[0:cut1,:], yBin[0:cut1], verbose=0, lammy=1, maxEvals=400)
    model2 = linear_model.linRegL2(XBin[cut1+1:cut2,:], yBin[cut1+1:cut2], verbose=0, lammy=1, maxEvals=400)
    model3 = linear_model.linRegL2(XBin[cut2+1:cut3,:], yBin[cut2+1:cut3], verbose=0, lammy=1, maxEvals=400)
    model4 = linear_model.linRegL2(XBin[cut3+1:cut4,:], yBin[cut3+1:cut4], verbose=0, lammy=1, maxEvals=400)

    model1.fit()
    model2.fit()
    model3.fit()
    model4.fit()    

    print("model1 Training error %.3f" % 
        utils.regression_error(model1.predict(XBin[0:cut1,:]), yBin[0:cut1]))
    print("model2 Training error %.3f" % 
        utils.regression_error(model2.predict(XBin[cut1+1:cut2,:]), yBin[cut1+1:cut2]))
    print("model3 Training error %.3f" % 
        utils.regression_error(model3.predict(XBin[cut2+1:cut3,:]), yBin[cut2+1:cut3]))
    print("model4 Training error %.3f" % 
        utils.regression_error(model4.predict(XBin[cut3+1:cut4,:]), yBin[cut3+1:cut4]))

    print("model1 Validation error %.3f" % 
        utils.regression_error(model1.predict(XBinValid), yBinValid))
    print("model2 Validation error %.3f" % 
        utils.regression_error(model2.predict(XBinValid), yBinValid))
    print("model3 Validation error %.3f" % 
        utils.regression_error(model3.predict(XBinValid), yBinValid))
    print("model4 Validation error %.3f" % 
        utils.regression_error(model4.predict(XBinValid), yBinValid))

    ## GLOBAL MODEL

    global_model = global_model.globalModel(verbose=0, maxEvals=400)
    global_model.add_model(model1)
    global_model.add_model(model2)
    global_model.add_model(model3)
    global_model.add_model(model4)

    global_model.fit(theta=0.1)
    print("global 0.1 Training error %.3f" % 
        utils.regression_error(global_model.predict(XBin), yBin))
    print("global 0.1 Validation error %.3f" % 
        utils.regression_error(global_model.predict(XBinValid), yBinValid))

    global_model.fit(theta=0.25)
    print("global 0.25 Training error %.3f" % 
        utils.regression_error(global_model.predict(XBin), yBin))
    print("global 0.25 Validation error %.3f" % 
        utils.regression_error(global_model.predict(XBinValid), yBinValid))

    global_model.fit(theta=0.5)
    print("global 0.5 Training error %.3f" % 
        utils.regression_error(global_model.predict(XBin), yBin))
    print("global 0.5 Validation error %.3f" % 
        utils.regression_error(global_model.predict(XBinValid), yBinValid))

    global_model.fit(theta=1)
    print("global 1 Training error %.3f" % 
        utils.regression_error(global_model.predict(XBin), yBin))
    print("global 1 Validation error %.3f" % 
        utils.regression_error(global_model.predict(XBinValid), yBinValid))

    print("global average Validation error %.3f" % 
        utils.regression_error(global_model.predictAverage(XBinValid), yBinValid))

    global_model.fitWeightedAverage(XBinValid[0:cutVal,:], yBinValid[0:cutVal])
    print("global-weighted Validation error %.3f" % 
        utils.regression_error(global_model.predictWeightedAverage(
            XBinValid[cutVal+1:cutVal2,:], Logistic=False), yBinValid[cutVal+1:cutVal2]))

    ## FULL MODEL

    full = linear_model.linRegL2(XBin, yBin, verbose=0, lammy=1, maxEvals=400)
    full.fit()

    print("full Training error %.3f" % 
        utils.regression_error(full.predict(XBin), yBin))
    print("full Validation error %.3f" % 
        utils.regression_error(full.predict(XBinValid), yBinValid))

