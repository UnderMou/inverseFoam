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
    with open("rawData/input_noil.yaml") as f:
        inputData = yaml.safe_load(f)

    # Experimental data
    with open("rawData/experimentalData_noil.yaml") as f:
        ExpData = yaml.safe_load(f)
    

    # Interpolate from raw data gradP(uw,ug)
    uw_grid, ug_grid, uw_mesh, ug_mesh, gradP_grid = interpolate_uwug_gradP(ExpData)
    uwug = np.array(ExpData['uw_ug']['data'])

    # Plot interpolated gradP(uw,ug)
    contourf = plt.contour(uw_mesh, ug_mesh, gradP_grid, levels=15, cmap='jet')
    plt.scatter(uwug[:,0], uwug[:,1], c=uwug[:,2], cmap='jet', edgecolor='k')
    cbar = plt.colorbar(contourf)
    cbar.set_label(r'$\nabla p$ [psi/ft]')
    plt.xlabel(r'$u_w$ [ft/day]')
    plt.ylabel(r'$u_g$ [ft/day]')
    plt.title(r'$u_o/u_w = 1/4$')
    plt.grid(True)
    plt.tight_layout()
    plt.xlim([0,1.8])
    plt.ylim([0,7])
    plt.savefig(figures_out_path + 'uwug_gradP_griddata_noil.png', dpi=300)
    plt.close()

    # Sample different foam-quality scan data from interpolated gradP(uw,ug) data
    ut_des_values = [2.0, 2.3, 3.0, 4.0, 4.5]
    uw_des_intervals = [np.linspace(0.145,1.1,30),
                        np.linspace(0.145,1.33,30),
                        np.linspace(0.145,1.7,50),
                        np.linspace(0.145,1.73,40),
                        np.linspace(0.16,1.73,40)]
    
    interp_expData = {}
    
    for i in range(len(ut_des_values)):
        uw_des = uw_des_intervals[i]
        ut_des = ut_des_values[i]

        ug_des, gradP_des, uo_des = sample_from_interp_gradPuwug(ut_des, uw_des, ExpData["uw_ug"]["wor"], ug_grid, uw_grid, gradP_grid)
        fg_des = ug_des / (uw_des_intervals[i] + ug_des + uo_des)

        k = inputData["core"]["k"]      # k = [m2]
        gradP_des /= 4.419e-5           # gradP = [Pa/m]
        ut_des /= 283465                # ut = [m/s]
        
        muapp_des = (gradP_des * k) / (ut_des)
        ut_des = ut_des*np.ones_like(uw_des)

        # reverse
        fg_des = fg_des[::-1]
        muapp_des = muapp_des[::-1]
        ug_des = ug_des[::-1]
        uw_des = uw_des[::-1]
        uo_des = uo_des[::-1]
        gradP_des = gradP_des[::-1]
        ut_des = ug_des + uw_des + uo_des

        # eval saturation 
        sw_des, sg_des = eval_saturation_2ph(fg_des, muapp_des, ExpData["uw_ug"]["wor"], inputData["fluids"], inputData["relPerm"])

        # encapsulate data
        interp_expData[f"Foam_Quality_Scan_ut_{ut_des_values[i]:.2f}"] = {
            'fg': fg_des.tolist(),
            'muApp': muapp_des.tolist(),
            'std_muApp': np.ones_like(muapp_des).tolist(),
            'velocities': {
                'wor': ExpData["uw_ug"]["wor"],
                'ug': ug_des.tolist(),
                'uw': uw_des.tolist(),
                'uo': np.zeros_like(ug_des).tolist(),
                'ut': ut_des.tolist()
            },
            'gradP': gradP_des.tolist(),
            'phase_saturations': {
                'Sw': sw_des.tolist(),
                'Sg': sg_des.tolist(),
                'So': np.zeros_like(sg_des).tolist()
            }
        }
    
    with open('rawData/' + 'interpolatedExperimentalData_noil.json', 'w') as f:
        json.dump(interp_expData, f, indent=4)

    with open('rawData/' + 'interpolatedExperimentalData_noil.json', 'r') as f:
        loaded_data = json.load(f)

    contourf = plt.contour(uw_mesh, ug_mesh, gradP_grid, levels=15, cmap='jet')
    # plt.scatter(uwug[:,0], uwug[:,1], c=uwug[:,2], cmap='jet', edgecolor='k')
    cbar = plt.colorbar(contourf)
    cbar.set_label(r'$\nabla p$ [psi/ft]')

    for i in range(len(interp_expData)):
        key = list(interp_expData.keys())[i]

        uw = interp_expData[key]["velocities"]["uw"]
        ug = interp_expData[key]["velocities"]["ug"]
        gradP = interp_expData[key]["gradP"]

        scatter = plt.scatter(uw, ug, c=gradP, cmap='jet', edgecolor='k')
        plt.plot([uw[0], uw[-1]], [ug[0], ug[-1]], 'k--')

    plt.xlabel(r'$u_w$ [ft/day]')
    plt.ylabel(r'$u_g$ [ft/day]')
    plt.title(f'$u_o/u_w = {interp_expData[key]["velocities"]["wor"]}$')
    plt.grid(True) 
    plt.tight_layout()
    plt.xlim([0,1.8])
    plt.ylim([0,7])
    plt.savefig(figures_out_path + 'uwug_gradP_interpolated_noil.png', dpi=300)
    plt.close()

    for i in range(len(interp_expData)):
        key = list(interp_expData.keys())[i]

        fg = interp_expData[key]["fg"]
        muApp = interp_expData[key]["muApp"]
        ut = interp_expData[key]["velocities"]["ut"][0]

        plt.plot(fg, muApp, label=f'$u_t = {ut}$ [ft./day]')
    
    plt.xlabel(r'$f_g$ [-]', fontsize=14)
    plt.ylabel(r'$\mu_{app}p$ [Pa.s]', fontsize=14)
    plt.title(r'$u_o/u_w = 0.0$')
    plt.grid(True)
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(figures_out_path + 'fg_muapp_noil.png', dpi=300)
    plt.close()
    
