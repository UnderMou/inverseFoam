import sys
import os
import json
import numpy as np

import pymc3 as pm
import arviz as az

from inverseFoam.problems.foamMCMC import foamForwardModel, make_th_forward_model

class foamMCMCanalysis:

    def __init__(self, mcmcProblem, ranges, alpha, model, input, expData, **kwargs):
        
        self.Y, self.std, self.alpha_vec = mcmcProblem.get_att()

        assert hasattr(self, "Y")
        assert hasattr(self, "std")
        assert hasattr(self, "alpha_vec")
        assert len(self.Y) == len(self.alpha_vec)

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

    def run(self, outputPath, n_samples, burnIn):

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
            trace = pm.sample(n_samples, step=step, tune=burnIn, cores=4)
    
        # text-based summary of the posteriors
        s = pm.summary(trace).round(2)
        print(s)

        if not os.path.exists(outputPath):
            os.makedirs(outputPath)

        sample_file = outputPath + '/mcmc.arv'
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
        fig.savefig(outputPath + '/trace_plot.png', dpi=300)
        print(pm.summary(trace))