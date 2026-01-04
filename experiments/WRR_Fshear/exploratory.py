import sys

sys.path.append("../../")

from inverseFoam.rawDataTools.dataInterpolators import *
# from inverseFoam.models.relativePermeability import coreyBrooks3ph

import numpy as np
import yaml
import json
import matplotlib.pyplot as plt
import matplotlib

if __name__ == '__main__':

    figures_out_path = 'rawData/figures/'
    
    # Input data
    with open("rawData/input.yaml") as f:
        inputData = yaml.safe_load(f)

    # Experimental data
    with open("rawData/experimentalData.yaml") as f:
        ExpData = yaml.safe_load(f)
    
    interp_expData = {}

    # Foam quality Scan data
    ExpDataArray = np.array(ExpData["FQS"]["data"])
    fg = ExpDataArray[:,0]
    muApp = ExpDataArray[:,1]
    std_muApp = ExpDataArray[:,2]
    ut = ExpData["FQS"]["ut"] * np.ones_like(fg)

    ug = ut * fg
    uw = ut - ug

    sw, sg = eval_saturation_2ph(fg, muApp, 0.0, inputData["fluids"], inputData["relPerm"])

    # interp_expData[f"Foam_Quality_Scan_ut_{ExpData["FQS"]["ut"]:.2f}"] = {
    interp_expData[f"Foam_Quality_Scan_ut_{ExpData['FQS']['ut']:.2f}"] = {
            'fg': fg.tolist(),
            'muApp': muApp.tolist(),
            'std_muApp': std_muApp.tolist(),
            'velocities': {
                'ug': ug.tolist(),
                'uw': uw.tolist(),
                'ut': ut.tolist()
            },
            'phase_saturations': {
                'Sw': sw.tolist(),
                'Sg': sg.tolist()
            }
    }
    
    # Flow rate scan data
    FRS_keys = key = list(ExpData.keys())[1:]
    for i in range(len(FRS_keys)):

        ExpDataArray = np.array(ExpData[FRS_keys[i]]["data"])
        muApp = ExpDataArray[:,0]
        ut = ExpDataArray[:,1]
        std_muApp = ExpDataArray[:,2]

        fg = ExpData[FRS_keys[i]]["fg"] * np.ones_like(ut)
        sw, sg = eval_saturation_2ph(fg, muApp, 0.0, inputData["fluids"], inputData["relPerm"])

        ug = ut * fg
        uw = ut - ug

        interp_expData[f"Flow_Rate_Scan_fg_{ExpData[FRS_keys[i]]['fg']:.2f}"] = {
            'fg': fg.tolist(),
            'muApp': muApp.tolist(),
            'std_muApp': std_muApp.tolist(),
            'velocities': {
                'ug': ug.tolist(),
                'uw': uw.tolist(),
                'ut': ut.tolist()
            },
            'phase_saturations': {
                'Sw': sw.tolist(),
                'Sg': sg.tolist()
            }
    }
        
    with open('rawData/' + 'interpolatedExperimentalData.json', 'w') as f:
        json.dump(interp_expData, f, indent=4)

    with open('rawData/' + 'interpolatedExperimentalData.json', 'r') as f:
        loaded_data = json.load(f)

    # Plot experimental data 

    ExpDataArray = np.array(ExpData["FQS"]["data"])
    fg = ExpDataArray[:,0]
    muApp = ExpDataArray[:,1]
    std_muApp = ExpDataArray[:,2]

    plt.errorbar(fg, muApp, yerr=std_muApp, fmt='bo', capsize=5, label=r'\textrm{Exp. Data} - $u_t\;\approx\;6.8$ [ft/day]')
    plt.ylabel(r"$\mu_{app} \, [\mathrm{cP}]$")
    plt.xlabel(r"$f_g \, [-]$")
    plt.savefig(figures_out_path + 'foamQualityScan_data.png', dpi=300)
    plt.close()

    FRS_keys = key = list(ExpData.keys())[1:]
    for i in range(len(FRS_keys)):

        ExpDataArray = np.array(ExpData[FRS_keys[i]]["data"])
        muApp = ExpDataArray[:,0]
        ut = ExpDataArray[:,1]
        std_muApp = ExpDataArray[:,2]

        plt.errorbar(ut, muApp, yerr=std_muApp, fmt='o', capsize=5, label=r'$f_g\;=\;$'+str(ExpData[FRS_keys[i]]["fg"]))
    
    plt.legend()
    plt.ylabel(r"$\mu_{app} \, [\mathrm{cP}]$")
    plt.xlabel("$u_t \, [ft/day]$")
    plt.savefig(figures_out_path + 'flowRateScan_data.png', dpi=300)
    plt.close()
        