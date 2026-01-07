import os
import json
import re
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from inverseFoam.rawDataTools.utils import unpack_expDataDict3ph

def read_estimates(directory):
    """
    Reads all .json files in a directory.

    Parameters
    ----------
    directory : str
        Path to directory containing JSON files

    Returns
    -------
    dict
        Dictionary with filename -> parsed JSON content
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"{directory} is not a valid directory")

    data = {}

    files = [
        f for f in os.listdir(directory)
        if f.endswith(".json")
    ]

    files_path = [
        directory + '/' + f for f in sorted(os.listdir(directory), key=natural_key)
        if f.endswith(".json")
    ]

    for filename in sorted(files, key=natural_key):
        path = os.path.join(directory, filename)
        with open(path, "r") as f:
            data[filename] = json.load(f)

    return data, files_path

def natural_key(s):
    """
    Split string into text and integer parts for natural sorting.
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r'(\d+)', s)
    ]

def generate_saturation(input, flow, wor, ns = 100, zero = 1e-6):

    muw = input['fluids']['muw']
    muo = input['fluids']['muo']

    swc = input['relPerm']['swc']
    sgr = input['relPerm']['sgr']
    sor = input['relPerm']['sor']
    krw0 = input['relPerm']['krw0']
    kro0 = input['relPerm']['kro0']
    krg0 = input['relPerm']['krg0']
    nw = input['relPerm']['nw_']
    ng = input['relPerm']['ng_']
    no = input['relPerm']['no_']

    if flow == '2ph':
        sw = np.linspace(swc + zero, 1 - sgr - zero, ns)
        sg = 1.0 - sw
        so = np.zeros_like(sw)

        return sw, sg, so

    if flow == '3ph':
        sw = np.linspace(swc + zero, 1 - sgr - sor - zero, ns)
        swe = (sw - swc) / (1 - sgr - sor - swc)
        krw = krw0*swe**nw
        mob_w = krw/muw

        soe = ( wor * muo * mob_w / kro0 ) ** ( 1./no )
        feasible_mask = (soe < 1) & (swe + soe < 1)

        soe = soe[feasible_mask]
        sw = sw[feasible_mask]
        swe = swe[feasible_mask]
        krw = krw[feasible_mask]
        mob_w = mob_w[feasible_mask]

        so = sor + soe * (1 - swc - sgr - sor)

        sge = 1-swe-soe
        sg = sgr + sge * (1 - swc - sgr - sor)

        return sw, sg, so

def eval_model_FQS(model, res_path, input, expDict, expName, flow = None):

    assert flow is not None and flow in ("2ph", "3ph"), (
        "flow must be defined and be either '2ph' or '3ph'"
    )

    wor = expDict['velocities']['wor']

    sw, sg, so = generate_saturation(input, flow, wor, ns=1000)

    ut_exp = expDict['velocities']['ut']
    ut = np.linspace(np.min(ut_exp), np.max(ut_exp), len(sw))
    ug = ut

    with open(res_path, 'r') as f:
        res = json.load(f)

    x = np.array(res['X'])
    model.set_foamParams(x)
    fg, fw, muapp, Nca, Fdry, Fshear, Foil = model.evaluate(sg, sw, so, ut, ug)

    res[expName] = {
        "sw": sw[::-1].tolist(),
        "sg": sg[::-1].tolist(),
        "so": so[::-1].tolist(),
        "fg": fg[::-1].tolist(),
        "fw": fw[::-1].tolist(),
        "muapp": muapp[::-1].tolist(),
        "Nca": Nca[::-1].tolist(),
        "Fdry": Fdry[::-1].tolist(),
        "Fshear": Fshear[::-1].tolist(),
        "Foil": Foil[::-1].tolist(),
    }

    return res
    
def eval_model_FRS(model, res_path, input, expDict, expName, flow = None, tol = 1e-3, nu = 50):

    assert flow is not None and flow in ("2ph", "3ph"), (
        "flow must be defined and be either '2ph' or '3ph'"
    )

    wor = expDict['velocities']['wor']

    sw_ast, sg_ast, so_ast = generate_saturation(input, flow, wor, ns=20000)

    ut_exp = expDict['velocities']['ut']
    ut = np.linspace(np.min(ut_exp), np.max(ut_exp), nu)
    ug = ut

    sw = np.zeros_like(ut)
    sg = np.zeros_like(ut)
    so = np.zeros_like(ut)
    fw = np.zeros_like(ut)
    fg = np.zeros_like(ut)
    muapp = np.zeros_like(ut)
    Nca = np.zeros_like(ut)
    Fdry = np.zeros_like(ut)
    Fshear = np.zeros_like(ut)
    Foil = np.zeros_like(ut)

    fg_exp = expDict['fg'][0]

    with open(res_path, 'r') as f:
        res = json.load(f)

    x = np.array(res['X'])
    model.set_foamParams(x)

    for i in range(len(ut)):
        ut_ast = ut[i] * np.ones_like(sw_ast)
        ug_ast = ut_ast

        fg_ast, fw_ast, muapp_ast, Nca_ast, Fdry_ast, Fshear_ast, Foil_ast = model.evaluate(sg_ast, sw_ast, so_ast, ut_ast, ug_ast)
        idx = np.argmin(np.abs(fg_ast - fg_exp))
        diff = np.abs(fg_ast[idx] - fg_exp)
        if diff > tol:
            raise ValueError(
                f"No fg value close enough to fg_exp. "
                f"Closest diff = {diff:.3e}, tolerance = {tol:.1e}"
            )
        
        sw[i] = sw_ast[idx]
        sg[i] = sg_ast[idx]
        so[i] = so_ast[idx]
        fw[i] = fw_ast[idx]
        fg[i] = fg_ast[idx]
        muapp[i] = muapp_ast[idx]
        Nca[i] = Nca_ast[idx]
        Fdry[i] = Fdry_ast[idx]
        Fshear[i] = Fshear_ast[idx]
        Foil[i] = Foil_ast[idx]

    res[expName] = {
        "sw": sw.tolist(),
        "sg": sg.tolist(),
        "so": so.tolist(),
        "ut": ut.tolist(),
        "fg": fg.tolist(),
        "fw": fw.tolist(),
        "muapp": muapp.tolist(),
        "Nca": Nca.tolist(),
        "Fdry": Fdry.tolist(),
        "Fshear": Fshear.tolist(),
        "Foil": Foil.tolist(),
    }

    return res   

def run_one_case(args):
    model, result_path, input_data, expDict, exp_key, flow, expType = args

    assert expType is not None and expType in ("FQS", "FRS"), (
        "flow must be defined and be either 'FQS' or 'FRS'"
    )

    if expType == 'FQS':
        res = eval_model_FQS(
            model,
            result_path,
            input_data,
            expDict,
            exp_key,
            flow=flow
        )
    else: 
        res = eval_model_FRS(
            model,
            result_path,
            input_data,
            expDict,
            exp_key,
            flow=flow
        )

    with open(result_path, "w") as f:
        json.dump(res, f, indent=4)

    return result_path