import sys

sys.path.append("../../")

from inverseFoam.models.relativePermeabilityModels import coreyBrooks3ph
from inverseFoam.models.foamModels import stars3ph
from inverseFoam.problems.foamLeastSquare import foamLeastSquareWRR
from inverseFoam.analysis.foamFit import foamFit, optSetup

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
    expData = [expData[keys[0]]]
    alphas = [1.0] 

    models = [stars3ph(input, NcaModel = 'Zeng', Fshear_Act = True)]

    ranges = [
        [10, 10000],                                               # fmmob
        [input['relPerm']['swc'], 1 - input['relPerm']['sgr']],    # fmdry
        [10, 4000],                                                # epdry
        [8.4e-8, 8.4e-8],                                          # fmcap
        [0,1]                                                      # epcap
    ]

    problem = foamLeastSquareWRR(ranges, alphas, models, input, expData)

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
    
    opt_setup = optSetup(problem, algorithm, termination)

    output_path = 'results/estimate_approachB'
    num_workers = 4
    num_runs = 200
    fitting = foamFit(opt_setup, num_workers, num_runs, output_path = output_path)

    # fitting.run_serial()
    fitting.run_parallel()