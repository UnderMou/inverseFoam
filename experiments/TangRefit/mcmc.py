import sys

sys.path.append("../../")

from inverseFoam.models.relativePermeabilityModels import coreyBrooks3ph
from inverseFoam.models.foamModels import stars3ph
from inverseFoam.problems.foamMCMC import mcmcProblemTangRefit
from inverseFoam.analysis.foamMCMCanalysis import foamMCMCanalysis

import numpy as np
import yaml
import json

if __name__ == "__main__":
    
    # Experimental data withoil
    with open("rawData/input_oil.yaml") as f:
        input_oil = yaml.safe_load(f)          

    with open('rawData/interpolatedExperimentalData_oil.json', 'r') as f:
        expData_oil = json.load(f)

    # Experimental data withOUT oil
    with open("rawData/input_noil.yaml") as f:
        input_noil = yaml.safe_load(f)          

    with open('rawData/interpolatedExperimentalData_noil.json', 'r') as f:
        expData_noil = json.load(f)
    
    keys_oil = list(expData_oil.keys())
    keys_noil = list(expData_noil.keys())
    expData = [expData_noil[keys_noil[2]]]
    alphas = [1.0] 

    models = [stars3ph(input_oil, NcaModel = 'Boeije', Fshear_Act = True, Foil_Act = True)]

    ranges = [
        [10, 500000],                                                                                    # fmmob
        [input_oil['relPerm']['swc'], 1 - input_oil['relPerm']['sgr'] - input_oil['relPerm']['sor']],    # fmdry
        [10, 20000],                                                                                     # epdry
        [1.54e-4, 1.54e-4],                                                                              # fmcap
        [1e-6, 5],                                                                                       # epcap
        [0.01, 1.0],                                                                                     # fmoil
        [0.01, 1.0],                                                                                     # floil
        [0.01, 10],                                                                                      # epoil
    ]

    mcmcProblem = mcmcProblemTangRefit(models[0], alphas, expData)
    mcmcAnalysis = foamMCMCanalysis(mcmcProblem, ranges, models[0], input_oil, expData)
    
    outputPath = 'results/'
    n_samples = 1000
    burnIn = 100

    mcmcAnalysis.run(outputPath, n_samples, burnIn)


   