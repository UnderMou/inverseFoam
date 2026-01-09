import numpy as np
from scipy.linalg import norm
from abc import ABC, abstractmethod

from pymoo.core.problem import Problem

from inverseFoam.rawDataTools.utils import unpack_expDataDict3ph

class foamLeastSquare(Problem, ABC):

    def __init__(self, ranges, alpha, models, input, expData, **kwargs):
        
        self.input = input
        self.expData = expData
        self.alpha = alpha
        self.models = models
        self.ranges = ranges

        assert len(self.expData) == len(self.alpha) == len(self.models), (
            "expData, alpha, and models must have the same length"
        )
        
        xl = []
        xu = []
        for i in range(len(ranges)):
            xl.append(ranges[i][0])
            xu.append(ranges[i][1])
        
        xl = np.array(xl)
        xu = np.array(xu)       

        super().__init__(n_var=len(xl), n_obj=1, n_ieq_constr=0, xl=xl, xu=xu, **kwargs)

    def set_seed(self, seed):
        self.seed = int(seed)

    @abstractmethod
    def _evaluate(self,  x, out, *args, **kwargs):
        """Must be implemented by subclasses"""
        pass

class foamLeastSquareWRR(foamLeastSquare):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def check_monotonicity(self, g, mu_appFRS):
            
        for j in range(1, len(mu_appFRS)):
            if mu_appFRS[j] >= mu_appFRS[j-1]:
                gp = (mu_appFRS[j] - mu_appFRS[j-1]) / (max(mu_appFRS) - min(mu_appFRS))
                if gp > g: g = gp

        return g

    def _evaluate(self, x, out, *args, **kwargs):
        
        errors = []
        alphaG = 0.1

        for i in range(len(self.expData)):

            expDict = self.expData[i]
            sg, sw, so, ut, ug, fg_exp, muapp_exp = unpack_expDataDict3ph(expDict)

            fgs = np.zeros((x.shape[0],sg.shape[0]))
            muapps = np.zeros((x.shape[0],sg.shape[0]))
            for j in range(len(x)):

                self.models[i].set_foamParams(x[j])
                fg, fw, muapp, Nca, Fdry, Fshear, Foil = self.models[i].evaluate(sg, sw, so, ut, ug)
                fgs[j,:] = fg
                muapps[j,:] = muapp
            
            error = np.zeros((x.shape[0], fg_exp.shape[0]))
            for j in range(x.shape[0]):
                error[j,:] = np.abs(muapps[j,:] - muapp_exp) / np.max(muapp_exp) + \
                              np.abs(fgs[j,:] - fg_exp) / np.max(fg_exp)
                
                weight_trans = 2.0
                if i == 0: # Foam quality scan
                    
                    id_muApp_trans = np.argmax(muapp_exp)
                    error[j,id_muApp_trans] *= weight_trans
                
                if i == 1: # 1st Flow rate scan

                    error[j,0] *= weight_trans


            errors.append(error)

        F = np.zeros((x.shape[0], self.n_obj))
        G = np.zeros((x.shape[0], self.n_obj))
        for i in range(x.shape[0]):
            for j in range(len(errors)):
                F[i] += self.alpha[j]*norm(errors[j][i,:], ord=2) 

                # g penalty
                if j > 0:
                    G[i] = self.check_monotonicity(G[i], muapp)

        out["F"] = F.flatten() + alphaG*G.flatten()

class foamLeastSquareLyuTang(foamLeastSquare):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        
        errors = []

        for i in range(len(self.expData)):

            expDict = self.expData[i]
            sg, sw, so, ut, ug, fg_exp, muapp_exp = unpack_expDataDict3ph(expDict)

            fgs = np.zeros((x.shape[0],sg.shape[0]))
            muapps = np.zeros((x.shape[0],sg.shape[0]))
            for j in range(len(x)):

                self.models[i].set_foamParams(x[j])
                fg, fw, muapp, Nca, Fdry, Fshear, Foil = self.models[i].evaluate(sg, sw, so, ut, ug)
                fgs[j,:] = fg
                muapps[j,:] = muapp
            
            error = np.zeros((x.shape[0], fg_exp.shape[0]))
            for j in range(x.shape[0]):
                error[j,:] = np.abs(muapps[j,:] - muapp_exp) / np.max(muapp_exp) + \
                              np.abs(fgs[j,:] - fg_exp) / np.max(fg_exp)

            errors.append(error)
        
        F = np.zeros((x.shape[0], self.n_obj))
        for i in range(x.shape[0]):
            for j in range(len(errors)):
                F[i] += self.alpha[j]*norm(errors[j][i,:], ord=2) 

        out["F"] = F.flatten()
