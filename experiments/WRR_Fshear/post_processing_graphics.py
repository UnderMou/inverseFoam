import sys

sys.path.append("../../")

import numpy as np
import json

import matplotlib.pyplot as plt
import scienceplots

from inverseFoam.postProcessing.dataPostProcessing import *
from inverseFoam.rawDataTools.utils import unpack_expDataDict3ph

if __name__ == "__main__":
    plt.style.use('science')

    results_path = 'results/approachB'

    results, results_path = read_estimates(results_path)
    result_keys = list(results.keys())

    with open('rawData/interpolatedExperimentalData.json', 'r') as f:
        exp_dicts = json.load(f)

    exp_keys = list(exp_dicts.keys())

    exp_idxs = [0, 1, 3]
    keys_exp = [exp_keys[i] for i in exp_idxs]
    expData = [exp_dicts[k] for k in keys_exp]

    # # Foam quality scan experiment
    exp_id = 0  
    expDict = expData[exp_id]    
    for j in range(len(results)):
        res = results[result_keys[j]]

        fg = res[keys_exp[exp_id]]['fg']
        muapp = res[keys_exp[exp_id]]['muapp']
        plt.plot(fg,muapp)

    sg_exp, sw_exp, so_exp, ut_exp, ug_exp, fg_exp, muapp_exp = unpack_expDataDict3ph(expDict)
    std_muapp_exp = expDict['std_muApp']
    plt.errorbar(fg_exp, muapp_exp, yerr=std_muapp_exp, fmt='bo', capsize=5, label=keys_exp[exp_id])

    plt.show()
        

    # Flow rate scan experiment fg = 0.6
    exp_id = 1 
    expDict = expData[exp_id]    
    for j in range(len(results)):
        res = results[result_keys[j]]

        ut = res[keys_exp[exp_id]]['ut']
        muapp = res[keys_exp[exp_id]]['muapp']
        plt.plot(ut,muapp)

    sg_exp, sw_exp, so_exp, ut_exp, ug_exp, fg_exp, muapp_exp = unpack_expDataDict3ph(expDict)
    std_muapp_exp = expDict['std_muApp']
    plt.errorbar(ut_exp, muapp_exp, yerr=std_muapp_exp, fmt='bo', capsize=5, label=keys_exp[exp_id])

    plt.show()

    # Flow rate scan experiment fg = 0.8
    exp_id = 2
    expDict = expData[exp_id]    
    for j in range(len(results)):
        res = results[result_keys[j]]

        ut = res[keys_exp[exp_id]]['ut']
        muapp = res[keys_exp[exp_id]]['muapp']
        plt.plot(ut,muapp)

    sg_exp, sw_exp, so_exp, ut_exp, ug_exp, fg_exp, muapp_exp = unpack_expDataDict3ph(expDict)
    std_muapp_exp = expDict['std_muApp']
    plt.errorbar(ut_exp, muapp_exp, yerr=std_muapp_exp, fmt='bo', capsize=5, label=keys_exp[exp_id])

    plt.show()
            