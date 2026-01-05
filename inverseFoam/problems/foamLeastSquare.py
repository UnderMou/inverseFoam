from pymoo.core.problem import Problem
import numpy as np
from scipy.linalg import norm

class foamLeastSquare(Problem):

    def __init__(self, ranges, alpha, models, input, expData, **kwargs):
        
        self.input = input
        self.expData = expData
        self.alpha = alpha
        self.models = models

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

    def _evaluate(self, x, out, *args, **kwargs):
        
        errors = []

        for i in range(len(self.expData)):

            expDict = self.expData[i]
            sg, sw, so, ut, ug, gradP, fg_exp, muapp_exp = unpack_expDataDict3ph(expDict)

            fgs = np.zeros((x.shape[0],sg.shape[1]))
            muapps = np.zeros((x.shape[0],sg.shape[1]))
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
            for j in range(len(self.expData)):
                F[i] += self.alphas[j]*norm(errors[j][i,:], ord=2) 

        out["F"] = F.flatten()
        