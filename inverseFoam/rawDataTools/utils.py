import numpy as np

def unpack_expDataDict3ph(expDataDict):
    
    sg = np.array(expDataDict["phase_saturations"]["Sg"])
    sw = np.array(expDataDict["phase_saturations"]["Sw"])
    so = np.array(expDataDict["phase_saturations"]["So"])
    ut = np.array(expDataDict["velocities"]["ut"])
    ug = np.array(expDataDict["velocities"]["ug"])
    gradP = np.array(expDataDict["gradP"])
    fg_exp = np.array(expDataDict["fg"])
    muapp_exp = np.array(expDataDict["muApp"])

    return sg, sw, so, ut, ug, gradP, fg_exp, muapp_exp