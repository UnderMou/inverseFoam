import numpy as np
from scipy import optimize as opt

# from relativePermeabilityModels import *

class stars3ph:

    def __init__(self, input, NcaModel = 'Boeije', Foil_Act = False, Fshear_Act = False):
        
        # Options
        self.Fshear_Act = Fshear_Act
        self.Foil_Act = Foil_Act

        # Fluid properties
        self.muw = input['fluids']['muw']
        self.mug = input['fluids']['mug']
        self.muo = input['fluids']['muo']
        self.sigma_wg = input['fluids']['sigma_wg']

        # Relative permeabilities
        self.relPerm = coreyBrooks3ph(input['relPerm'])

        # Vectorized methods
        self.vfdry = np.vectorize(self.eval_fdry)
        if self.Fshear_Act: self.vfshear = np.vectorize(self.eval_fshear)
        if self.Foil_Act: self.vfoil = np.vectorize(self.eval_foil)

        # Nca model
        self.NcaModel = NcaModel

    def set_foamParams(self, x):
        """
            x = [fmmob, fmdry, epdry, fmcap, epcap, fmoil, floil, epoil]
                   0      1      2      3      4      5      6      7
        """

        self.fmmob = x[0]
        self.fmdry = x[1]
        self.epdry = x[2]

        if self.Fshear_Act:
            self.fmcap = x[3]
            self.epcap = x[4]
        
        if self.Foil_Act:
            self.fmoil = x[5]
            self.floil = x[6]
            self.epoil = x[7]

    def eval_fdry(self, Sw):
        return 0.5 + np.arctan(self.epdry*(Sw-self.fmdry)) / np.pi
        
    def eval_foil(self, So):
        if (So > self.fmoil):
            Fo = 0.0
        elif (So < self.floil):
            Fo = 1.0
        else:
            Fo = ((self.fmoil - So)/(self.fmoil - self.floil))**self.epoil
        return Fo

    def eval_fshear(self, Nca):
        if (Nca > self.fmcap):
            Fshear = (self.fmcap / Nca)**self.epcap
        else:
            Fshear = 1.0

        return Fshear
    
    def fshear_residual(self, mu_app, mob_w, mob_g, mob_o, sw, so, ut):

        Nca = (mu_app * ut) / self.sigma_wg
        
        Fw = self.eval_fdry(sw)
        Fo = self.eval_foil(so)
        Fs = self.eval_fshear(Nca)
        
        mrf = 1.0 + self.fmmob * Fw * Fo * Fs
        mob_t = mob_w + mob_o + (mob_g/mrf)

        return np.abs(mu_app - (1.0/mob_t))
    
    def muappApprox(self, mob_w, mob_g, mob_o, sw, so, ut):
        results = []
        for mw, mg, mo, s_w, s_o, u_t in zip(mob_w, mob_g, mob_o, sw, so, ut):
            result = opt.minimize_scalar(
                self.fshear_residual,
                args=(mw, mg, mo, s_w, s_o, u_t),
                method='bounded',
                bounds=(1e-6, 10)
            )
            results.append(result.x)

        return np.array(results)
         
    def evaluate(self, Sg, Sw, So, ut, ug):

        krg = self.relPerm.krg(Sg)
        krw = self.relPerm.krw(Sw)
        kro = self.relPerm.kro(So)
        mob_w, mob_o, mob_g = krw/self.muw, kro/self.muo, krg/self.mug

        if self.Fshear_Act:
            if self.NcaModel == 'Boeije':
                muapp = self.muappApprox(mob_w, mob_g, mob_o, Sw, So, ut)
                Nca = muapp * ut / self.sigma_wg
            elif self.NcaModel == 'Zeng':
                Nca = self.muw * ut / self.sigma_wg
            elif self.NcaModel == 'Abbaszadeh':
                Nca = self.muw * ug / self.sigma_wg
            else : 
                raise ValueError(
                    "Not valid NcaModel. Use 'Boeije', 'Zeng' or 'Abbaszadeh'"
                )
        else:
            Nca = np.ones_like(krg)

        Fdry = self.vfdry(Sw)

        if self.Fshear_Act: Fshear = self.vfshear(Nca)
        Fshear = Fshear if self.Fshear_Act else np.ones_like(krg)

        if self.Foil_Act: Foil = self.vfoil(So)
        Foil  = Foil   if self.Foil_Act  else np.ones_like(krg)

        mrf = 1.0 + self.fmmob * Fdry * Foil * Fshear

        mob_t = mob_w + mob_o + mob_g/mrf
        fg = (mob_g / mrf) / mob_t
        fw = mob_w / mob_t
        muapp = 1 / mob_t
        
        return fg, fw, muapp, Nca, Fdry, Fshear, Foil
    