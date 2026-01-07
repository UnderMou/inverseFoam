import sys

sys.path.append("../../")

# from inverseFoam.models.relativePermeabilityModels import coreyBrooks3ph
from inverseFoam.models.foamModels import stars3ph
# from inverseFoam.problems.foamLeastSquare import foamLeastSquareWRR
# from inverseFoam.analysis.foamFit import foamFit, optSetup
from inverseFoam.rawDataTools.utils import unpack_expDataDict3ph
from inverseFoam.postProcessing.dataPostProcessing import *

import numpy as np
import yaml
import json

import matplotlib.pyplot as plt

if __name__ == "__main__":

    results_path = 'results/approachB'

    _, results_path = read_estimates(results_path)
    print(results_path)

    with open("rawData/input.yaml") as f:
        input = yaml.safe_load(f)

    with open('rawData/interpolatedExperimentalData.json', 'r') as f:
        exp_dicts = json.load(f)

    exp_keys = list(exp_dicts.keys())

    exp_idxs = [0, 1, 3]
    keys_exp = [exp_keys[i] for i in exp_idxs]
    expData = [exp_dicts[k] for k in keys_exp]

    models = [stars3ph(input, NcaModel = 'Zeng', Fshear_Act = True),
              stars3ph(input, NcaModel = 'Zeng', Fshear_Act = True),
              stars3ph(input, NcaModel = 'Zeng', Fshear_Act = True)]

    # # Foam quality scan experiment
    exp_id = 0  
    expDict = expData[exp_id]    
    for j in range(len(results_path)):
        res = eval_model_FQS(models[exp_id], results_path[j], input, expDict, keys_exp[exp_id], flow = '2ph')
        with open(results_path[j], 'w') as f:
            json.dump(res, f, indent=4)

    # Flow rate scan experiment fg = 0.6
    exp_id = 1  
    expDict = expData[exp_id]    
    for j in range(len(results_path)):
        res = eval_model_FRS(models[exp_id], results_path[j], input, expDict, keys_exp[exp_id], flow = '2ph')
        with open(results_path[j], 'w') as f:
            json.dump(res, f, indent=4)

    # Flow rate scan experiment fg = 0.8
    exp_id = 2
    expDict = expData[exp_id]    
    for j in range(len(results_path)):
        res = eval_model_FRS(models[exp_id], results_path[j], input, expDict, keys_exp[exp_id], flow = '2ph')
        with open(results_path[j], 'w') as f:
            json.dump(res, f, indent=4)
            