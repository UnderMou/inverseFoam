import numpy as np

class coreyBrooks3ph:

    def __init__(self, infos):

        # Residuals
        self.swc = infos['swc']
        self.sgr = infos['sgr']
        self.sor = infos['sor']

        # Exponents
        self.nw = infos['nw_']
        self.ng = infos['ng_']
        self.no = infos['no_']

        # End-points
        self.krw0 = infos['krw0']
        self.krg0 = infos['krg0']
        self.kro0 = infos['kro0']
    
    def krw(self, Sw):
        return self.krw0 * np.power(np.divide(Sw - self.swc, 1 - self.swc - self.sgr - self.sor), self.nw)

    def krg(self, Sg):
        return self.krg0 * np.power(np.divide(Sg - self.sgr, 1 - self.swc - self.sgr - self.sor), self.ng)
    
    def kro(self, So):
        return self.kro0 * np.power(np.divide(So - self.sor, 1 - self.swc - self.sgr - self.sor), self.no)