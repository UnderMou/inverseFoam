import sys
import os
import json
import numpy as np

import pymc3 as pm
import theano.tensor as tt
from theano.compile.ops import as_op
import arviz as az

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

class foamMCMC_WRR:

    def __init__(self, ranges, alpha, model, input, expData, **kwargs):
        
        # gather experimental data into a Y vector
        exp_list = [expData[i]['muApp'] for i in range(len(expData))]
        self.Y = np.concatenate(exp_list)
        print(self.Y)

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
        print(self.alpha_vec)

        assert self.Y.shape == self.alpha_vec.shape

        # check model active functions and define parameter ranges
        self.Fshear_act = getattr(model, "Fshear_Act", False) 
        self.Foil_act = getattr(model, "Foil_Act", False) 

        self.fmmob_range = ranges[0]
        self.fmdry_range = ranges[1]
        self.epdry_range = ranges[2]

        if self.Fshear_act: 
            self.fmcap = ranges[3][0]
            self.epcap_range = ranges[4]
        else: 
            self.fmcap = 1
        
        if self.Foil_act:
            self.fmoil_range = ranges[5]
            self.floil_range = ranges[6]
            self.epoil_range = ranges[7]

        # initiate th_forward_model function
        forward_model = foamForwardModel(model, input, expData, self.fmcap)
        self.th_model = make_th_forward_model(forward_model, self.Foil_act, self.Fshear_act)

    def run(self):

        basic_model = pm.Model()

        with basic_model:

            # define priors
            fmmob = pm.Uniform('fmmob',  lower = self.fmmob_range[0], upper = self.fmmob_range[1])
            fmdry = pm.Uniform('fmdry',  lower = self.fmdry_range[0], upper = self.fmdry_range[1])
            epdry = pm.Uniform('epdry',  lower = self.epdry_range[0], upper = self.epdry_range[1])
            if self.Fshear_act: epcap   = pm.Uniform('epcap',  lower = self.epcap_range[0], upper = self.epcap_range[1])
            if self.Foil_act: fmoil   = pm.Uniform('fmoil',  lower = self.fmoil_range[0], upper = self.fmoil_range[1])
            if self.Foil_act: floil   = pm.Uniform('floil',  lower = self.floil_range[0], upper = self.floil_range[1])
            if self.Foil_act: epoil   = pm.Uniform('epoil',  lower = self.epoil_range[0], upper = self.epoil_range[1])

            # eval model
            if self.Fshear_act and self.Foil_act:
                mu_val = self.th_model(fmmob, fmdry, epdry, epcap, fmoil, floil, epoil)
            elif self.Fshear_act and not self.Foil_act:
                mu_val = self.th_model(fmmob, fmdry, epdry, epcap)
            elif not self.Fshear_act and self.Foil_act:
                mu_val = self.th_model(fmmob, fmdry, epdry, fmoil, floil, epoil) 
            else:
                mu_val = self.th_model(fmmob, fmdry, epdry)    

            # define erros in data
            sigma_val = pm.HalfNormal('sigma', sigma=0.5*self.std)

            #define the log-likelihood
            Y_obs = pm.Normal('Y_obs', mu=mu_val * np.sqrt(self.alpha_vec), sigma=sigma_val, observed=self.Y * np.sqrt(self.alpha_vec))

            # instantiate sampler
            step = pm.Slice()
            
            # draw posterior samples
            trace = pm.sample(100000, step=step, tune=5000, cores=4)
    
        # text-based summary of the posteriors
        s = pm.summary(trace).round(2)
        print(s)

        sample_file = 'stars_mcmc.arv'
        fid = open(sample_file, "w")

        if self.Fshear_act and self.Foil_act:
            p0 , p1 , p2 , p3, p4, p5, p6  = trace['fmmob'] , trace['fmdry'] , trace['epdry'] , trace['epcap'], trace['fmoil'], trace['floil'], trace['epoil']
            for k in range(len(p0)):
                fid.write("%d \t %0.8e \t %0.8e \t %0.8e \t %0.8e \t %0.8e \t %0.8e \t %0.8e\n" % (k+1,p0[k],p1[k],p2[k],p3[k],p4[k],p5[k],p6[k]))
            fid.close()
        elif self.Fshear_act and not self.Foil_act:
            p0 , p1 , p2 , p3  = trace['fmmob'] , trace['fmdry'] , trace['epdry'] , trace['epcap']
            for k in range(len(p0)):
                fid.write("%d \t %0.8e \t %0.8e \t %0.8e \t %0.8e \n" % (k+1,p0[k],p1[k],p2[k],p3[k]))
            fid.close()
        elif not self.Fshear_act and self.Foil_act:
            p0 , p1 , p2 , p3, p4, p5  = trace['fmmob'] , trace['fmdry'] , trace['epdry'] , trace['fmoil'], trace['floil'], trace['epoil']
            for k in range(len(p0)):
                fid.write("%d \t %0.8e \t %0.8e \t %0.8e \t %0.8e \t %0.8e \t %0.8e\n" % (k+1,p0[k],p1[k],p2[k],p3[k],p4[k],p5[k]))
            fid.close() 
        else:
            p0 , p1 , p2  = trace['fmmob'] , trace['fmdry'] , trace['epdry']
            for k in range(len(p0)):
                fid.write("%d \t %0.8e \t %0.8e \t %0.8e \n" % (k+1,p0[k],p1[k],p2[k]))
            fid.close()

        axes = az.plot_trace(trace)
        fig = axes.ravel()[0].figure
        fig.savefig('trace_plot_MCMC_LyuTang.png', dpi=300)
        print(pm.summary(trace))