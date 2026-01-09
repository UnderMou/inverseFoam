import multiprocessing
import json
import numpy as np
import os
import csv
from copy import deepcopy

from inverseFoam.problems.foamLeastSquare import foamLeastSquareWRR

class identifiabilityProblemWRR():

    def __init__(self, ranges_orig, models, input, expData, alphas, num_points, num_runs_perpoint, first_seed = 1, output_path =  "foamIdentifiability_run"):

        self.models = models
        self.ranges = ranges_orig

        path = os.path.join(os.getcwd(), output_path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = output_path

        self.nrunpp = num_runs_perpoint
        self.Np = num_points
        self.firstSeed = first_seed

        # Define ranges
        ranges = []
        for i in range(len(ranges_orig)):
            array = np.linspace(ranges_orig[i][0], ranges_orig[i][1], self.Np)
            ranges.append(array)
        
        # Define problem data
        data = []
        ranges2write = deepcopy(ranges)
        for i in range(len(ranges_orig)):
            if not np.isclose(ranges_orig[i][0], ranges_orig[i][1]):
                for j in range(ranges[i].shape[0]):
                    problemRange = deepcopy(ranges_orig)
                    problemRange[i] = [ranges[i][j], ranges[i][j]]
                    problem = foamLeastSquareWRR(problemRange, alphas, models, input, expData)
                    item = (problem, self.nrunpp, self.firstSeed)
                    data.append(item)
            else:
                ranges2write.pop(i)
            
        ranges2write = [rang.tolist() for rang in ranges2write]
        with open(output_path + '/ranges.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(ranges2write)
        
        self.data = data  
