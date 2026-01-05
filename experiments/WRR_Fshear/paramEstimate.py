import sys

sys.path.append("../../")

from inverseFoam.models.relativePermeabilityModels import coreyBrooks3ph
from inverseFoam.models.foamModels import stars3ph
from inverseFoam.problems import foamLeastSquare
from inverseFoam.analysis.foamFit import foamFit, optSetup

import numpy as np
import yaml
import json

from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination.default import DefaultSingleObjectiveTermination

if __name__ == "__main__":

    ranges = [[10, 10000],   # fmmob
              [0, 1],        # fmdry
              [10, 5000]]    # epdry
    
    alphas = [1.0]           # weights for each experiment

    with open('rawData/interpolatedExperimentalData.json', 'r') as f:
        expData = json.load(f)
    
    keys = list(expData.keys())
    expData = expData[keys[0]]

    with open("rawData/input.yaml") as f:
        input = yaml.safe_load(f)

    models = [stars3ph(input)]

    problem = foamLeastSquare(ranges, alphas, models, input, expData)

    algorithm = DE(
        pop_size=200,
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

    num_workers = 3
    num_runs = 9
    fitting = foamFit(opt_setup, num_workers, num_runs)

    fitting.run()