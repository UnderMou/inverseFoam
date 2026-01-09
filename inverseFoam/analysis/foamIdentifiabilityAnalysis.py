import os
import csv
import ast

from tqdm import tqdm
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt

from pymoo.optimize import minimize

def process_item(item):

    problem, nrunpp, firstSeed, algorithm, termination = item
    
    XPL_arr = run_identifiability_analysis(problem, algorithm, termination, nrunpp, firstSeed)
    
    return XPL_arr

def run_identifiability_analysis(problem, algorithm, termination, nrunpp, firstSeed):

    obj_func_values = []
    for i in range(nrunpp):
        seed = firstSeed + i 

        res = minimize(problem,
                algorithm,
                termination,
                seed=seed,
                verbose=False)

        obj_func_values.append(res.F)
    
    obj_func_values = np.array(obj_func_values)
    
    return np.min(obj_func_values)

class foamIdentifiabilityAnalysis():

    def __init__(self, problem, algorithm, termination, num_workers):
        
        data = []
        # Define process item
        for i in range(len(problem.data)):
            item = problem.data[i] + (algorithm, termination)
            data.append(item)
        self.data = data

        self.np = num_workers
        self.problem = problem

    def run(self):

        with mp.Pool(processes=self.np) as pool:
            res = list(
                tqdm(
                    pool.imap(process_item, self.data),
                    total=len(self.data),
                    desc="Running jobs"
                )
            )
        
        self.Fshear_act = any([getattr(model, "Fshear_Act", False) for model in self.problem.data[0][0].models]) 
        self.Foil_act = any([getattr(model, "Foil_Act", False) for model in self.problem.data[0][0].models])   

        N_params = 3
        if self.Fshear_act: N_params += 1
        if self.Foil_act: N_params += 3
                
        results = []
        for i in range(N_params):       
            array = res[i*self.problem.Np:i*self.problem.Np+self.problem.Np]
            results.append(array)

        filename = self.problem.path + '/XPLs.csv'

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(results)
        
        ### Post processing
        arrays = []
        with open(filename, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                # row is a list of strings, each string is "[...]"
                arrays = [ast.literal_eval(entry) for entry in row]
                # break  # assuming one row; remove if multiple rows
        XPL = np.array(arrays, dtype=float)
        # print(XPL)
        
        with open(self.problem.path + '/ranges.csv', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                # row is a list of strings, each string is "[...]"
                arrays = [ast.literal_eval(entry) for entry in row]
                # break  # assuming one row; remove if multiple rows
        ranges = np.array(arrays, dtype=float)
        # print(ranges)

        j = 0
        for i in range(XPL.shape[0]):
            x = np.linspace(ranges[i][0], ranges[i][-1], XPL[i,:].shape[0])
            plt.plot(x,XPL[i,:])
            plt.show()