import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import scipy.stats as st
from math import exp,sqrt,log
import matplotlib.pyplot as plt
import scienceplots
from uqpylab import sessions, display_util
from scipy.optimize import minimize
import json
import copy 

# def loss(params, x_target, y_target):
#     mu, sigma = params
#     y_pred = pdf_param(x_target, mu, sigma)
#     return np.sum((y_target - y_pred)**2)

# def best_fit_distribution(x, y, uq, bins=200, ax=None):
  
#     DISTRIBUTIONS = ['Gaussian']

#     # Best holders
#     best_sse = np.inf

#     # Estimate distribution parameters from data
#     for distribution in DISTRIBUTIONS:

#         # fit dist to data
#         init_params = [np.mean(x), np.std(x)]
#         result = minimize(loss, init_params, args=(x, y, uq), bounds=[(-np.inf, np.inf), (1e-6, np.inf)])

#         # Separate parts of parameters
#         arg = params[:-2]
#         loc = params[-2]
#         scale = params[-1]

#         # Calculate fitted PDF and error with fit in distribution
#         pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
#         sse = np.power((y - pdf) / (np.max(y)), 2.0) 
#         sse = np.sum(sse)
        
#         #print('Distribution::',distribution.name,'Root Mean Square error::',np.sqrt(sse) / (np.power(len(y),2)))
#         #print('Dist. params::',params)
#         #print('')

#     # identify if this distribution is better
#         if (best_sse > sse):
#             best_distribution = distribution
#             best_params = params
#             best_sse = sse
#             best_pdf = pdf

#     return best_distribution.name, best_params, best_pdf

########################################################################
# The execution
########################################################################

if __name__ == "__main__":
    
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

    labels = ['fmmob','epdry','fmdry', 'epsurf','epoil','fmoil','floil']
    file_id     = "results/sample3/inference_oil_muApp/mcmc_chains.npz" #'stars_mcmc.arv'


    data = np.load(file_id)
    for key in data.files:
        print(key, data[key].shape)
    chains = data[data.files[0]]
    chains = chains.reshape(-1, chains.shape[-1])
    df = pd.DataFrame(chains)
    

    data = pd.read_csv(file_id, delimiter='\t', header=None)
    data.columns = ['id'] + labels
    print(data.head())

    types = [['Gaussian','Gamma','Beta'],
             ['Gaussian'],
             ['Gaussian'],
             ['Gaussian'],
             ['Gaussian','Gamma','Beta'],
             ['Gaussian','Gamma','Beta'],
             ['Gaussian']]

    fig, axs = plt.subplots(2, 4, figsize=(10, 3))
    resDict = {}
    for i in range(len(labels)):
    
        X = data[labels[i]].tolist()

        iOpts = {
            'Inference': {
            'Data': X
            },
            'Copula': {
            'Type': 'Independent'
            },
            # 'Marginals': {'Type' : ['Gaussian','Gamma','Beta']}
            # 'Marginals': {'Type' : ['Gaussian']}
            'Marginals': {'Type' : types[i]}
        }

        InputHat = uq.createInput(iOpts)

        # for key, value in InputHat.items():
        #     print(key)
        # print('\n', InputHat['Marginals']['Type'], InputHat['Marginals']['Parameters'])
        # print('\n', InputHat['Marginals']['GoF'])
    
        InputOpts = {
            "Marginals": [
                {"Name": labels[i],
                "Type": InputHat['Marginals']['Type'],
                "Parameters": InputHat['Marginals']['Parameters'],
                "Moments": InputHat['Marginals']['Moments']
                }
            ]
        }
        resDict[labels[i]] = copy.deepcopy(InputOpts['Marginals'])

        del InputOpts['Marginals'][0]['Moments']

        myInput = uq.createInput(InputOpts)
        uq.print(myInput)
        Samples = uq.getSample(myInput, 1e4)
        axs.flatten()[i].hist(Samples, density=True, bins=100, label='uq.getSample('+InputHat['Marginals']['Type']+')')
    
    
        x_kde = st.gaussian_kde(np.array(X))
        x = np.linspace(min(np.array(X)), max(np.array(X)), 200) 
        y = x_kde(x)
        axs.flatten()[i].plot(x,y,'k',label='KDE from MCMC')

        axs.flatten()[i].grid(True)
        axs.flatten()[i].set_xlabel(labels[i])
        axs.flatten()[i].set_ylabel(r'$\mathcal{f}_{\mathit{X}}$')
        axs.flatten()[i].legend(loc='best',fontsize=5)
 
    plt.tight_layout()
    plt.savefig('fittedPDFS.png',dpi=500)
    plt.show()

    with open("fitted_marginals.json", "w") as f:
        json.dump(resDict, f, indent=4)

    exit()

       
    # Now PDFs for the model parameters
    for k in range(1,len(labels)):
        
        print('########################################################################')
        _,tgt = load_sample_data(file_id,k+1)
        comp = int(len(tgt)/4)
        print('Chain size: ', comp)

        for i in range(4): # 4 chains
            rang = range(i*comp, i*comp + comp)
            print(rang[0],rang[-1])

            x_kde = st.gaussian_kde(tgt[rang])
            x = np.linspace(min(tgt[rang]), max(tgt[rang]), 200) 
            y = x_kde(x)
            plt.plot(x,y)

            InputOpts = {
                "Marginals": [
                    {"Name": "param",
                    "Type": "Gaussian",
                    "Parameters": [np.mean(x),1e-7]
                    }
                ]
            }
            myInput = uq.createInput(InputOpts)
            uq.print(myInput)
            theSamples = uq.getSample(myInput, 1e3)
            print(theSamples)
            plt.hist(theSamples, density=True)
            plt.show()
            exit()


#             best_fit_name, best_fit_params, best_pdf = best_fit_distribution(x,y,uq)

#             print('Parameter', labels[k], 'is::',best_fit_name)
#             print('Dist. params::',best_fit_params)
#             # grid, pdf = kde(tgt[rang])
#             # print('Local Mode: ', grid[np.argmax(pdf)], 'Local Median: ', np.median(tgt[rang]), 'Local mean: ', np.mean(tgt[rang]))
#             # 
#             # print('Erro moda = ', abs(grid[np.argmax(pdf)] - best_fit_params[0]))
#             # plot
#             smp = pd.Series(tgt[rang])
        
#             # plt.plot(x,best_pdf,c='k',linewidth=5)
#             plt.figure(figsize=(3,3))
#             sns.kdeplot(smp, color='k')
#             sns.histplot(smp, stat='density',palette="ch:s=.25,rot=-.25", bins=50)
#             #sns.displot(smp, stat='density' ,color='b')

            
#             plt.plot(x,y,'r')

#             arg   = best_fit_params[:-2]
#             loc   = best_fit_params[-2]
#             scale = best_fit_params[-1]
#             best_dist = getattr(st, best_fit_name)
#             pdf_fitted = best_dist.pdf(x, loc=loc, scale=scale, *arg)
#             plt.plot(x, pdf_fitted, 'b-',label=best_fit_name)
            
            
#             plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
#             plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#             plt.xlabel(labels[k])
#             plt.ylabel('Density')
#             plt.xlim((limits[k][0], limits[k][1]))
#             #plt.legend(loc='best')
#             # plt.tight_layout()
#             name = 'pdfs/' + labels_name[k] + '_param_fit' + str(i) + '.png'
#     #        plt.show()
#             plt.savefig(name, dpi=300)
#             plt.clf()

#         print('\n\n')
#         x,tgt = load_sample_data(file_id,k+1)
#         best_fit_name, best_fit_params, best_pdf = best_fit_distribution([x,tgt])
#         print('Parameter', labels[k], 'is::',best_fit_name)
#         print('Dist. params::',best_fit_params)
#         # grid, pdf = kde(tgt)
#         # print('Global Mode: ', grid[np.argmax(pdf)])
#         # print('Global Median: ', np.median(tgt))
#         print('Global mean: ', np.mean(tgt))
#         print('########################################################################')
#         print('')
        

#         arg   = best_fit_params[:-2]
#         loc   = best_fit_params[-2]
#         scale = best_fit_params[-1]

#         best_dist = getattr(st, best_fit_name)
        
#         pdf_fitted = best_dist.pdf(x, loc=loc, scale=scale, *arg)
#         plt.plot(x, pdf_fitted, '-',label=best_fit_name)
        
        
# #        param = st.norm.fit(tgt)
# #        mu , sigma = param
# #        pdf_fitted = st.norm.pdf(x, *param)
# #        print('Normal Fit :: ',labels[k],mu,sigma, 'COV: ',coeff_var(mu,sigma))
# #        plt.plot(x,pdf_fitted,'-',label="normal fit")
        
# #        param = st.lognorm.fit(tgt)
# #        shape, loc, scale = param
# #        mean , std = np.log(scale) , shape
# #        print('Log-Normal Fit :: ',labels[k],mean,std, 'COV: ',coeff_var_ln(std))
# #        pdf_fitted = st.lognorm.pdf(x, *param)
# #        plt.plot(x, pdf_fitted, '-',label="logn fit")

# #        (low, higher) = st.uniform.fit(tgt)
# #        print('Uniform Fit :: ',labels[k],low,higher)
# #        pdf_fitted = st.uniform.pdf(x, low,higher)
# #        plt.plot(x, pdf_fitted, '-',label="uniform fit")

# #        param = st.rayleigh.fit(tgt)
# #        loc, scale = param
# #        print('Rayleigh Fit :: ',labels[k],loc, scale)
# #        pdf_fitted = st.rayleigh.pdf(x, *param)
# #        plt.plot(x, pdf_fitted, '-',label="rayleigh fit")

#         smp = pd.Series(tgt)
#         plt.figure(figsize=(3,3))
#         plt.plot(x,best_pdf,c='b',linewidth=5)
#         sns.kdeplot(smp, color='k')
#         sns.histplot(smp, stat='density',palette="ch:s=.25,rot=-.25", bins=50)
        
#         plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
#         plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#         plt.xlabel(labels[k])
#         plt.ylabel('Density')
#         plt.xlim((limits[k][0], limits[k][1]))
#         #plt.legend(loc='best')
#         # plt.tight_layout()
#         name = 'pdfs/' + labels_name[k] + '_param_fit.png'
# #        plt.show()
#         plt.savefig(name, dpi=300)
#         plt.clf()
    
    
