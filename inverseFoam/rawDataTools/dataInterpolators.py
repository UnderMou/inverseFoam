import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator

def interpolate_uwug_gradP(ExpData):
        # Interpolate experimental gradP(uw,ug) experimental data to augment data
        
        uwug = np.array(ExpData['uw_ug']['data'])
        uw_grid = np.linspace(min(uwug[:,0]), max(uwug[:,0]), 200)
        ug_grid = np.linspace(min(uwug[:,1]), max(uwug[:,1]), 200)
        uw_mesh, ug_mesh = np.meshgrid(uw_grid, ug_grid)
        grid_points = np.column_stack((uwug[:,0], uwug[:,1]))
        gradP_grid = griddata(grid_points, uwug[:,2], (uw_mesh, ug_mesh), method='linear')

        return uw_grid, ug_grid, uw_mesh, ug_mesh, gradP_grid

def sample_from_interp_gradPuwug(ut_des, uw_des, wor, ug_grid, uw_grid, grad_P_grid):
        # Sample from interpolated gradP(uw,ug) diagram given a desired ut and wor = uo/uw ratio

        ug_des = ut_des - (1 + wor) * uw_des
        uo_des = (wor) * uw_des
        interp_func = RegularGridInterpolator((ug_grid, uw_grid), grad_P_grid)
        points = np.column_stack((ug_des, uw_des))
        grad_P_des = interp_func(points)
        return ug_des, grad_P_des, uo_des

def eval_saturation_3ph(fg, muapp, wo_rel, fluidsProp, relPerm):
        # Evaluate the phases saturation given fg_muApp data and water-oil velocity ratio
        muw = fluidsProp['muw']
        muo = fluidsProp['muo']

        krw0 = relPerm['krw0']
        nw = relPerm['nw_']
        kro0 = relPerm['kro0']
        no = relPerm['no_']
        swc = relPerm['swc']
        sgr = relPerm['sgr']
        sor = relPerm['sor']

        swe = (((1-fg) / (1+wo_rel)) * muw/(muapp*krw0) )**(1/nw)
        swe = np.clip(swe, 0, 1)
        Sw = (1-swc-sgr-sor) * swe + swc

        soe = np.array([
            ( ((1-fgx) / (1+1/wo_rel)) * muo / (muappx * kro0) )**(1/no) 
            for fgx, muappx in zip(fg, muapp)
        ])
        soe = np.clip(soe, 0, 1)
        So = (1-swc-sgr-sor) * soe + sor

        Sg = 1.0 - Sw - So

        return Sw, Sg, So

def eval_saturation_2ph(fg, muapp, wo_rel, fluidsProp, relPerm):
        # Evaluate the phases saturation given fg_muApp data

        muw = fluidsProp['muw']
        krw0 = relPerm['krw0']
        nw = relPerm['nw_']
        swc = relPerm['swc']
        sgr = relPerm['sgr']

        swe = (((1-fg) / (1+wo_rel)) * muw/(muapp*krw0) )**(1/nw)
        swe = np.clip(swe, 0, 1)
        Sw = (1-swc-sgr) * swe + swc

        Sg = 1.0 - Sw 

        return Sw, Sg