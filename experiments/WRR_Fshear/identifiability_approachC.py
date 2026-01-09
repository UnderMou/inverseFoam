import sys

sys.path.append("../../")

from inverseFoam.models.relativePermeabilityModels import coreyBrooks3ph
from inverseFoam.models.foamModels import stars3ph
from inverseFoam.problems.foamIdentifiability import identifiabilityProblemWRR
from inverseFoam.analysis.foamIdentifiabilityAnalysis import foamIdentifiabilityAnalysis

import numpy as np
import yaml
import json

from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination.default import DefaultSingleObjectiveTermination

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
        [10, 4000],                                                # epdry
        [8.4e-8, 8.4e-8],                                          # fmcap
        [0,1]                                                      # epcap
    ]

    Np = 100
    Nrunpp = 3
    output_path = 'results/identifiability_approachC'

    problem = identifiabilityProblemWRR(ranges, models, input, expData, alphas, Np, Nrunpp, output_path=output_path)
    
    algorithm = DE(
        pop_size=50,
        sampling= FloatRandomSampling(),
        variant="DE/best/1/bin",
        CR=0.6,
        F=0.7,
        dither="vector",
        jitter=False
    )

    termination = DefaultSingleObjectiveTermination(
                        n_max_gen=2000,
                        n_max_evals=400000,
                        period=30)
    
    num_workers = 4
    identifiabilityAnalysis = foamIdentifiabilityAnalysis(problem, algorithm, termination, num_workers)
    identifiabilityAnalysis.run()
