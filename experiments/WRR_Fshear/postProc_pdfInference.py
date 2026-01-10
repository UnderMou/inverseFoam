import json
from tqdm import tqdm
import pandas as pd
from copy import deepcopy

from uqpylab import sessions, display_util

if __name__ == "__main__":

    approach = 'approachB'

    output_path_mcmc = 'results/mcmc_' + approach

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


    labels = ['fmmob','fmdry','epdry','epcap']
    candidate_pdfs = [['Gaussian','Uniform'],
                      ['Gaussian'],
                      ['Gaussian'],
                      ['Gaussian','Beta']]
    
    if approach == "approachA":
        labels = labels[:3]
        candidate_pdfs = candidate_pdfs[:3]
    
    if approach == "approachB":
        candidate_pdfs[0] = ['Uniform']
    
    data = pd.read_csv(output_path_mcmc + '/mcmc.arv', delimiter='\t', header=None)
    data.columns = ['id'] + labels
    print(data.head())

    data = data.tail(10000)
    print(data.shape)
    
    resDict = {}
    for i in tqdm(range(len(labels)), desc="Processing"):
    
        X = data[labels[i]].tolist()

        iOpts = {
            'Inference': {
            'Data': X
            },
            'Copula': {
            'Type': 'Independent'
            },
            'Marginals': {'Type' : candidate_pdfs[i]}
        }

        InputHat = uq.createInput(iOpts)
    
        InputOpts = {
            "Marginals": [
                {"Name": labels[i],
                "Type": InputHat['Marginals']['Type'],
                "Parameters": InputHat['Marginals']['Parameters'],
                "Moments": InputHat['Marginals']['Moments']
                }
            ]
        }
        resDict[labels[i]] = deepcopy(InputOpts['Marginals'])

    with open(output_path_mcmc + "/marginals.json", "w") as f:
        json.dump(resDict, f, indent=4)