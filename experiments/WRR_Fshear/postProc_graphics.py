import sys
import os

sys.path.append("../../")

import numpy as np
import pandas as pd
import json

import matplotlib.pyplot as plt
import scienceplots
import csv
import ast
import corner
import scipy.stats as st

from uqpylab import sessions, display_util

from inverseFoam.postProcessing.dataPostProcessing import *
from inverseFoam.postProcessing.graphicsPostProcessing import *
from inverseFoam.rawDataTools.utils import unpack_expDataDict3ph

if __name__ == "__main__":
    plt.style.use('science')

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12
    })

    approach = 'approachB'
    output_path_PL = 'results/identifiability_' + approach
    output_path_estimate = 'results/estimate_' + approach
    output_path_mcmc = 'results/mcmc_' + approach

    folder = os.path.basename(output_path_estimate)

    results, results_path = read_estimates(output_path_estimate)
    result_keys = list(results.keys())

    with open('rawData/interpolatedExperimentalData.json', 'r') as f:
        exp_dicts = json.load(f)

    exp_keys = list(exp_dicts.keys())

    exp_idxs = [0, 1, 3]
    keys_exp = [exp_keys[i] for i in exp_idxs]
    expData = [exp_dicts[k] for k in keys_exp]

    # Filter
    F = gather_field(results, result_keys, "F")

    # fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    # plot_F_histogram(F, ax=ax)
    # plt.tight_layout()
    # plt.show()

    mask = F < 0.22
    result_keys = [k for k, m in zip(result_keys, mask) if m]
    results = {k: results[k] for k in result_keys}   
    F = F[mask]
    # fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    # plot_F_histogram(F, ax=ax)
    # plt.tight_layout()
    # plt.show()

    
    
    # # X histograms
    # xlabel = ['fmmob', 'fmdry', 'epdry', 'fmcap', 'epcap']
    # X = gather_field(results, result_keys, "X")
    # fig, ax = plt.subplots(1, len(xlabel), figsize=(20, 4))
    # for i in range(len(ax)):
    #     if not folder.endswith("approachA"):
    #         plot_F_histogram(X[:,i], ax=ax[i], xlabel = xlabel[i])
    #     else:
    #         if i < 3:
    #             plot_F_histogram(X[:,i], ax=ax[i], xlabel = xlabel[i])
    # plt.tight_layout()
    # plt.show()






    # # # Foam quality scan experiment
    # exp_id = 0  
    # expDict = expData[exp_id]    

    # fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    # plot_FQS(
    #     results,
    #     result_keys,
    #     keys_exp,
    #     exp_id,
    #     expDict,
    #     ax=ax
    # )
    # plt.tight_layout()
    # plt.show()


    # if not folder.endswith("approachA"):

    #     # Flow rate scan experiment fg = 0.6

    #     exp_id = 1 
    #     expDict = expData[exp_id]   

    #     fig, ax = plt.subplots(1, 1, figsize=(5, 4)) 
    #     plot_FRS(
    #         results,
    #         result_keys,
    #         keys_exp,
    #         exp_id,
    #         expDict,
    #         ax=ax
    #     )
    #     plt.tight_layout()
    #     plt.show()

    #     # Flow rate scan experiment fg = 0.8

    #     exp_id = 2
    #     expDict = expData[exp_id]    
        
    #     fig, ax = plt.subplots(1, 1, figsize=(5, 4)) 
    #     plot_FRS(
    #         results,
    #         result_keys,
    #         keys_exp,
    #         exp_id,
    #         expDict,
    #         ax=ax
    #     )
    #     plt.tight_layout()
    #     plt.show()
            

    
    # # Profile likelihood
    # arrays = []
    # with open(output_path_PL + '/XPLs.csv', newline='') as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         arrays = [ast.literal_eval(entry) for entry in row]
    # XPL = np.array(arrays, dtype=float)
    
    # with open(output_path_PL + '/ranges.csv', newline='') as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         arrays = [ast.literal_eval(entry) for entry in row]
    # ranges = np.array(arrays, dtype=float)
    
    # xlabel = ['fmmob', 'fmdry', 'epdry', 'epcap']
    # alpha = 0.2
    # fig, ax = plt.subplots(1, len(xlabel), figsize=(20, 4))
    # for i in range(len(ax)):
    #     if not folder.endswith("approachA"):
    #         plot_profile_likelihood(
    #             ranges[i],
    #             XPL[i,:],
    #             alpha,
    #             ax=ax[i],
    #             label=xlabel[i]
    #         )
    #     else:
    #         if i < 3:
    #             plot_profile_likelihood(
    #             ranges[i],
    #             XPL[i,:],
    #             alpha,
    #             ax=ax[i],
    #             label=xlabel[i]
    #         )
        
    #     if i > 0:
    #         ax[i].set_ylabel("")
    #         ax[i].tick_params(axis="y", labelleft=False)
    # plt.tight_layout()
    # plt.show()






    # Corner plot
    data = np.loadtxt(output_path_mcmc + '/mcmc.arv')

    if not folder.endswith("approachA"):
        xlabel = ['fmmob', 'fmdry', 'epdry', 'epcap']
    else:
        xlabel = ['fmmob', 'fmdry', 'epdry']

    df = pd.DataFrame(data=data[:,1:],index=data[:,0],columns=xlabel)
    
    # fig = corner.corner(df, show_titles=True, bins=100, labels=xlabel, 
    #                         plot_datapoints=False,  labelpad=0.25, quantiles=[0.05, 0.5, 0.95],
    #                         title_fmt="0.1e")
    
    # plt.plot()
    # plt.tight_layout()
    # plt.show()




    # Pdfs plot

    ############ UQ[py]Lab - INPUT #############
    # Initialize common plotting parameters
    display_util.load_plt_defaults()
    uq_colors = display_util.get_uq_color_order()
    # Start the session
    myToken = '270fe799b4e4e4f004da79cec3dffb07cf9da78a' # The user's token to access the UQCloud API
    UQCloud_instance = 'https://uqcloud.ethz.ch' # The UQCloud instance to use
    mySession = sessions.cloud(host=UQCloud_instance, token=myToken)
    # (Optional) Get a convenient handle to the command line interface
    uq = mySession.cli
    # Reset the session
    mySession.reset()
    # Set random seed for reproducibility
    uq.rng(0,'twister')
    ###########################################

    with open(output_path_mcmc + '/marginals.json', 'r') as f:
        pdf_dicts = json.load(f)
    
    
    xlabel = ['fmmob', 'fmdry', 'epdry', 'epcap']

    if approach == "approachA": xlabel = xlabel[:3]

    fig, axs = plt.subplots(1, len(xlabel), figsize=(12, 3))

    for i in range(len(xlabel)):

        dictInput = {"Marginals":[
                        pdf_dicts[xlabel[i]][0]
        ]}

        del dictInput['Marginals'][0]['Moments']

        myInput = uq.createInput(dictInput)
        uq.print(myInput)
        Samples = uq.getSample(myInput, 1e4)
        axs.flatten()[i].hist(Samples, density=True, bins=100, label='uq.getSample(' + dictInput['Marginals'][0]['Type'] + ')')
    
        X = df[xlabel[i]].tolist()
        
        x_kde = st.gaussian_kde(np.array(X))
        x = np.linspace(min(np.array(X)), max(np.array(X)), 200) 
        y = x_kde(x)
        axs.flatten()[i].plot(x,y,'k',label='KDE from MCMC')

        axs.flatten()[i].grid(True)
        axs.flatten()[i].set_xlabel(xlabel[i])
        axs.flatten()[i].set_ylabel(r'$f_x$')
        axs.flatten()[i].legend(loc='best', fontsize = 5)

    plt.tight_layout()
    plt.show()