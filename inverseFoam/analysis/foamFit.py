import multiprocessing
import json
import numpy as np
import os
from copy import deepcopy

from pymoo.optimize import minimize

class optSetup():

    def __init__(self, problem, algorithm, termination):
        
        self.problem = problem
        self.algorithm = algorithm
        self.termination = termination

class foamFit():

    def __init__(self, opt_setup, num_workers, num_runs, first_seed = 1):

        self.np = num_workers
        self.nrun = num_runs
        self.dir = os.getcwd()

        self.seed = np.arange(first_seed, num_runs + 1, dtype=int)
        self.data = [[self.seed[i], opt_setup] for i in range(len(self.seed))]

    def run(self):

        with multiprocessing.Pool(processes=self.np) as pool:
            results = pool.map(self.process_item, self.data)

    def process_item(self, item):
        seed, opt_setup = item[0], item[1]
        opt_run = deepcopy(opt_setup)
        self.run_optimization(seed, opt_run)

    def run_optimization(self, seed, opt_setup):

        path = self.dir + f'/foamFit_run/seed_{seed}.json'

        opt_setup.problem.set_seed(seed)

        res = minimize(opt_setup.problem,
                opt_setup.algorithm,
                opt_setup.termination,
                seed=seed,
                verbose=False)
        
        file = {
            'X': res.X.tolist(),
            'F': res.F.tolist(),
            'seed': int(seed)
        }
        
        with open(path, 'w') as f:
            json.dump(file, f)

    