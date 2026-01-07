import numpy as np

def convert_cP_to_Pas(muapp):
    return muapp / 1000

def convert_ftDay_to_mSeg(ut):
    return ut * 3.5278e-6

def unpack_expDataDict3ph(expDataDict):
    
    sg = np.array(expDataDict["phase_saturations"]["Sg"])
    sw = np.array(expDataDict["phase_saturations"]["Sw"])
    so = np.array(expDataDict["phase_saturations"]["So"])
    ut = np.array(expDataDict["velocities"]["ut"])
    ug = np.array(expDataDict["velocities"]["ug"])
    fg_exp = np.array(expDataDict["fg"])
    muapp_exp = np.array(expDataDict["muApp"])

    return sg, sw, so, ut, ug, fg_exp, muapp_exp