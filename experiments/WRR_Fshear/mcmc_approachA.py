import sys

sys.path.append("../../")

from inverseFoam.models.relativePermeabilityModels import coreyBrooks3ph
from inverseFoam.models.foamModels import stars3ph
from inverseFoam.problems.foamMCMC import mcmcProblemWRR
from inverseFoam.analysis.foamMCMCanalysis import foamMCMCanalysis

import numpy as np
import yaml
import json

if __name__ == "__main__":
    
    with open("rawData/input.yaml") as f:
        input = yaml.safe_load(f)          

    with open('rawData/interpolatedExperimentalData.json', 'r') as f:
        expData = json.load(f)
    
    keys = list(expData.keys())
    expData = [expData[keys[0]]]
    alphas = [1.0] 

    models = [stars3ph(input, Fshear_Act = False)]

    ranges = [
        [10, 1000],                                               # fmmob
        [input['relPerm']['swc'], 1 - input['relPerm']['sgr']],    # fmdry
        [10, 1000]                                                 # epdry
    ]

    mcmcProblem = mcmcProblemWRR(models[0], alphas, expData)
    mcmcAnalysis = foamMCMCanalysis(mcmcProblem, ranges, models[0], input, expData)
    
    outputPath = 'results/mcmc_approachA'
    n_samples = 100000
    burnIn = 20000

    mcmcAnalysis.run(outputPath, n_samples, burnIn)


   