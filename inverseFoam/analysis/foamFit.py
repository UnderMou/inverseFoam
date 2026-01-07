import multiprocessing
import json
import numpy as np
import os
from copy import deepcopy

from pymoo.optimize import minimize

def run_optimization(path, seed, opt_setup):

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
        
        with open(path + f'/seed_{seed}.json', 'w') as f:
            json.dump(file, f)

def process_item(item):
        path, seed, opt_setup = item[0], item[1], item[2]
        opt_run = deepcopy(opt_setup)
        run_optimization(path, seed, opt_run)

class optSetup():

    def __init__(self, problem, algorithm, termination):
        
        self.problem = problem
        self.algorithm = algorithm
        self.termination = termination

class foamFit():

    def __init__(self, opt_setup, num_workers, num_runs, first_seed = 1, output_path =  "foamFit_run"):

        path = os.path.join(os.getcwd(), output_path)
        if not os.path.exists(path):
            os.makedirs(path)

        self.np = num_workers
        self.nrun = num_runs

        self.seed = np.arange(first_seed, num_runs + 1, dtype=int)
        self.data = [[path, self.seed[i], opt_setup] for i in range(len(self.seed))]

    def run_parallel(self):

        with multiprocessing.Pool(processes=self.np) as pool:
            results = pool.map(process_item, self.data)
    
    def run_serial(self):
        
        for i in range(len(self.data)):

            item = self.data[i]
            path, seed, opt_setup = item[0], item[1], item[2]
            run_optimization(path, seed, opt_setup)

    

    

    