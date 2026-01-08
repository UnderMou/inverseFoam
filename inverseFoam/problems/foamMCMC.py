import numpy as np

import theano.tensor as tt
from theano.compile.ops import as_op

from inverseFoam.rawDataTools.utils import unpack_expDataDict3ph

def make_th_forward_model(forwardModel, Foil_act, Fshear_act):

    # -------------------------------
    # Case 1: no oil, no shear
    # -------------------------------
    if not Foil_act and not Fshear_act:

        @as_op(
            itypes=[tt.dscalar, tt.dscalar,tt.dscalar],
            otypes=[tt.dvector]
        )
        def th_forward(fmmob, fmdry, epdry):
            return forwardModel.th_forward_no_oil_no_shear(fmmob, fmdry, epdry)

        return th_forward

    # -------------------------------
    # Case 2: with oil, no shear
    # -------------------------------
    if Foil_act and not Fshear_act:

        @as_op(
            itypes=[tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar],
            otypes=[tt.dvector]
        )
        def th_forward(fmmob, fmdry, epdry, fmoil, floil, epoil):
            return forwardModel.th_forward_with_oil_no_shear(fmmob, fmdry, epdry, fmoil, floil, epoil)

        return th_forward

    # -------------------------------
    # Case 3: no oil, with shear
    # -------------------------------
    if not Foil_act and Fshear_act:

        @as_op(
            itypes=[tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar],
            otypes=[tt.dvector]
        )
        def th_forward(fmmob, fmdry, epdry, epcap):
            return forwardModel.th_forward_no_oil_with_shear(fmmob, fmdry, epdry, epcap)

        return th_forward

    # -------------------------------
    # Case 4: with oil, with shear
    # -------------------------------
    if Foil_act and Fshear_act:

        @as_op(
            itypes=[tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar],
            otypes=[tt.dvector]
        )
        def th_forward(fmmob, fmdry, epdry, epcap, fmoil, floil, epoil):
            return forwardModel.th_forward_with_oil_with_shear(fmmob, fmdry, epdry, epcap, fmoil, floil, epoil)

        return th_forward

class foamForwardModel:

    def __init__(self, model, input, expData, fixedfmcap, **kwargs):
        
        self.input = input
        self.expData = expData
        self.model = model
        self.fixedfmcap = fixedfmcap

    def th_forward_no_oil_no_shear(self, fmmob, fmdry, epdry):
        
        x = np.array([fmmob,fmdry,epdry])
        res = []

        for i in range(len(self.expData)):

            expDict = self.expData[i]
            sg, sw, so, ut, ug, fg_exp, muapp_exp = unpack_expDataDict3ph(expDict)

            self.model.set_foamParams(x)
            fg, fw, muapp, Nca, Fdry, Fshear, Foil = self.model.evaluate(sg, sw, so, ut, ug)
 
            res.append(muapp)

        return np.concatenate(res)

    def th_forward_with_oil_no_shear(self, fmmob, fmdry, epdry, fmoil, floil, epoil):
        
        x = np.array([fmmob,fmdry,epdry,fmoil,floil,epoil])
        res = []

        for i in range(len(self.expData)):

            expDict = self.expData[i]
            sg, sw, so, ut, ug, fg_exp, muapp_exp = unpack_expDataDict3ph(expDict)

            self.model.set_foamParams(x)
            fg, fw, muapp, Nca, Fdry, Fshear, Foil = self.model.evaluate(sg, sw, so, ut, ug)
 
            res.append(muapp)

        return np.concatenate(res)

    def th_forward_no_oil_with_shear(self, fmmob, fmdry, epdry, epcap):
        fmcap = self.fixedfmcap
        x = np.array([fmmob,fmdry,epdry,fmcap,epcap])
        res = []

        for i in range(len(self.expData)):

            expDict = self.expData[i]
            sg, sw, so, ut, ug, fg_exp, muapp_exp = unpack_expDataDict3ph(expDict)

            self.model.set_foamParams(x)
            fg, fw, muapp, Nca, Fdry, Fshear, Foil = self.model.evaluate(sg, sw, so, ut, ug)
 
            res.append(muapp)
 
        return np.concatenate(res)

    def th_forward_with_oil_with_shear(self, fmmob, fmdry, epdry, epcap, fmoil, floil, epoil):
        fmcap = self.fixedfmcap
        x = np.array([fmmob,fmdry,epdry,fmcap,epcap,fmoil,floil,epoil])
        res = []

        for i in range(len(self.expData)):

            expDict = self.expData[i]
            sg, sw, so, ut, ug, fg_exp, muapp_exp = unpack_expDataDict3ph(expDict)

            self.model.set_foamParams(x)
            fg, fw, muapp, Nca, Fdry, Fshear, Foil = self.model.evaluate(sg, sw, so, ut, ug)
 
            res.append(muapp)
        
        return np.concatenate(res)
    
class mcmcProblemWRR:

    def __init__(self, alpha, expData, **kwargs):

        # gather experimental data into a Y vector
        exp_list = [expData[i]['muApp'] for i in range(len(expData))]
        self.Y = np.concatenate(exp_list)

        # experimental error/uncertainty
        self.std = max(
            np.max(exp['std_muApp']) for exp in expData
        )

        weight_trans = 2.0

        exp_id = 0
        alpha_QS = alpha[exp_id] * np.ones_like(expData[exp_id]['muApp'])
        alpha_QS[np.argmax(expData[exp_id]['muApp'])] *= weight_trans

        exp_id = 1
        alpha_FR1 = alpha[exp_id] * np.ones_like(expData[exp_id]['muApp'])

        exp_id = 2
        alpha_FR2 = alpha[exp_id] * np.ones_like(expData[exp_id]['muApp'])
        alpha_FR2[0] *= weight_trans

        self.alpha_vec = np.concatenate((alpha_QS, alpha_FR1, alpha_FR2))
    
    def get_att(self):

        return self.Y, self.std, self.alpha_vec

