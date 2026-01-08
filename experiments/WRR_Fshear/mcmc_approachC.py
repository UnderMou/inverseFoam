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
    expData = [expData[keys[0]], expData[keys[1]], expData[keys[3]]]
    alphas = [0.2, 0.4, 0.4] 

    models = [stars3ph(input, NcaModel = 'Zeng', Fshear_Act = True),
              stars3ph(input, NcaModel = 'Zeng', Fshear_Act = True),
              stars3ph(input, NcaModel = 'Zeng', Fshear_Act = True)]

    ranges = [
        [10, 10000],                                               # fmmob
        [input['relPerm']['swc'], 1 - input['relPerm']['sgr']],    # fmdry
        [10, 1000],                                                # epdry
        [8.4e-8, 8.4e-8],                                          # fmcap
        [0,1]                                                      # epcap
    ]

    mcmcProblem = mcmcProblemWRR(alphas, expData)
    mcmcAnalysis = foamMCMCanalysis(mcmcProblem, ranges, alphas, models[0], input, expData)
    
    outputPath = 'results/mcmc_approachC'
    n_samples = 20000
    burnIn = 5000

    mcmcAnalysis.run(outputPath, n_samples, burnIn)


   