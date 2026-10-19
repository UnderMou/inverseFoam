import importlib.util
import numpy as np
from copy import deepcopy
from types import SimpleNamespace
from scipy.optimize import brentq

try:
    from numba import njit
except ImportError:
    def njit(function):
        return function

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def decode(z):
    """Os quatro parâmetros de espuma do bloco de surfactante usam coordenadas
    logarítmicas. 
    Óleo usa `(log(epoil), floil, qoil)` com `fmoil=floil+(1-floil)*qoil`,
    impondo `epoil>0` e `0<=floil<fmoil<=1`.
    """
    z = np.asarray(z, dtype=float)
    values = np.empty_like(z)
    values[..., :5] = np.exp(z[..., :5])
    values[..., 6] = z[..., 5]
    values[..., 5] = z[..., 5]+(1-z[..., 5])*z[..., 6]
    return values

def encode(theta, names):
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (len(names),) or not np.isfinite(theta).all() or np.any(theta[:5] <= 0):
        raise ValueError("Vetor de parâmetros de espuma inválido.")
    epoil, fmoil, floil = theta[4:]
    if not (0 <= floil < fmoil <= 1):
        raise ValueError("Foil exige 0 <= floil < fmoil <= 1.")
    return np.r_[np.log(theta[:5]), floil, (fmoil-floil)/(1-floil)]

def make_model(theta, inputs, fixed_relperm, foam_module, FMSURF, foam=True):
    inp = deepcopy(inputs)
    assert inp["relPerm"] == fixed_relperm
    relPermModule = foam_module.coreyBrooks3ph(inp["relPerm"])
    model = foam_module.stars3ph(inp, relPermModule, Foil_Act=True, Fshear_Act=False, Fsurf_Act=True)
    model.set_foamParams([theta[0] if foam else 0., theta[2], theta[1], theta[5], theta[6], theta[4]])
    model.fmsurf, model.epsurf = FMSURF, theta[3]
    return model

def make_initial(seed, initial_z, zlo, zhi, start_margin, so_obs, f2_obs, SAT_EPS):
    rng = np.random.default_rng(seed)
    z = initial_z.copy()
    z[:4] = np.clip(z[:4]+rng.normal(0, 2.5, 4), zlo[:4]+start_margin[:4], zhi[:4]-start_margin[:4])
    floil = rng.uniform(0., float(so_obs[f2_obs < 1].min()))
    fmoil = rng.uniform(max(float(so_obs.max()), floil+SAT_EPS), 1.)
    z[4:] = [rng.uniform(zlo[4], zhi[4]), floil, (fmoil-floil)/(1-floil)]
    return np.clip(z, zlo+start_margin, zhi-start_margin)

@njit
def fast_state(theta, t, foam, fsurf, foil_zero, rp, muw, mug):
    nw, ng, krw0, krg0, swc, sgr, sor = rp
    span, denominator = 1-swc-sgr, 1-swc-sgr-sor
    log_s = -np.logaddexp(0., -t)
    log_g = -np.logaddexp(0., t)
    sw = swc+span*np.exp(log_s)
    # Preserva o denominador Corey com sor conforme o YAML.
    log_scale = np.log(span/denominator)
    lw = np.log(krw0/muw)+nw*(log_s+log_scale)
    lg = np.log(krg0/mug)+ng*(log_g+log_scale)
    if foam:
        fdry = .5+np.arctan(theta[1]*(sw-theta[2]))/np.pi
        lg -= np.log1p(theta[0]*fdry*fsurf*foil_zero)
    return lw, lg, sw

@njit
def fast_forward(theta, fgs, flags, concentrations, foil_zero, rp, fmsurf, muw, mug):
    nw, ng, krw0, krg0, swc, sgr, sor = rp
    scale = (1-swc-sgr)/(1-swc-sgr-sor)
    mu, sw = np.empty(len(fgs)), np.empty(len(fgs))
    for i in range(len(fgs)):
        fg, foam, Cs = fgs[i], flags[i], concentrations[i]
        fsurf = (Cs/fmsurf)**theta[3] if Cs < fmsurf else 1.
        if fg == 0:
            mu[i], sw[i] = 1000*muw/(krw0*scale**nw), 1-sgr
        elif fg == 1:
            dry = .5+np.arctan(theta[1]*(swc-theta[2]))/np.pi
            mrf = 1+theta[0]*dry*fsurf*foil_zero if foam else 1.
            mu[i], sw[i] = 1000*mug/(krg0*scale**ng)*mrf, swc
        else:
            target = np.log(fg)-np.log1p(-fg)
            lo, hi = -10000., 10000.
            for iteration in range(65):
                mid = (lo+hi)/2
                lw, lg, saturation = fast_state(theta, mid, foam, fsurf, foil_zero, rp, muw, mug)
                if lg-lw > target:
                    lo = mid
                else:
                    hi = mid
            lw, lg, saturation = fast_state(theta, (lo+hi)/2, foam, fsurf, foil_zero, rp, muw, mug)
            mu[i], sw[i] = 1000*np.exp(-np.logaddexp(lw, lg)), saturation
    return mu, sw

def predict_oil(theta, foam_module, saturations):
    theta = np.asarray(theta, dtype=float)
    epoil, fmoil, floil = theta[4:]
    parameters = SimpleNamespace(epoil=epoil, fmoil=fmoil, floil=floil)
    saturations = np.atleast_1d(np.asarray(saturations, dtype=float))
    return np.array([foam_module.stars3ph.eval_foil(parameters, float(so)) for so in saturations])

def predict(theta, fgs, flags, velocities, concentrations, foam_module, fixed_rp, FMSURF, config):
    fgs, flags = np.asarray(fgs, dtype=float), np.asarray(flags, dtype=bool)
    concentrations, velocities = np.asarray(concentrations, dtype=float), np.asarray(velocities, dtype=float)

    theta = np.asarray(theta, dtype=float)

    foil_zero = float(predict_oil(theta, foam_module, [0.])[0])
    return fast_forward(theta, fgs, flags, concentrations, foil_zero, fixed_rp, FMSURF, config["fluids"]["muw"], config["fluids"]["mug"])

def residual_blocks(theta, foam_module, point_weights, mu_obs, f2_obs, so_obs, mu_scale, f2_scale, fgs, flags, velocities, concentrations, fixed_rp, FMSURF, config):
    predicted, _ = predict(theta, fgs, flags, velocities, concentrations, foam_module, fixed_rp, FMSURF, config)
    surf_r = np.sqrt(1.0*point_weights)*(mu_obs-predicted)/mu_scale
    oil_r = np.sqrt(1.0)*(f2_obs-predict_oil(theta, foam_module, so_obs))/f2_scale
    return surf_r, oil_r

def residual(z, foam_module, point_weights, mu_obs, f2_obs, so_obs, mu_scale, f2_scale, fgs, flags, velocities, concentrations, fixed_rp, FMSURF, config):
    return np.concatenate(residual_blocks(decode(z), foam_module, point_weights, mu_obs, f2_obs, so_obs, mu_scale, f2_scale, fgs, flags, velocities, concentrations, fixed_rp, FMSURF, config))

def objective_components(theta, foam_module, point_weights, mu_obs, f2_obs, so_obs, mu_scale, f2_scale, fgs, flags, velocities, concentrations, fixed_rp, FMSURF, config):
    surf_r, oil_r = residual_blocks(theta, foam_module, point_weights, mu_obs, f2_obs, so_obs, mu_scale, f2_scale, fgs, flags, velocities, concentrations, fixed_rp, FMSURF, config)
    surf_sse, oil_sse = float(surf_r @ surf_r), float(oil_r @ oil_r)
    return dict(surf_SSE=surf_sse, oil_SSE=oil_sse, weighted_normalized_SSE=surf_sse+oil_sse)

def evaluate_sw(model, sw, ut, Cs):
    swc, sgr = model.relPerm.swc, model.relPerm.sgr
    sw = np.clip(np.atleast_1d(sw).astype(float), swc, 1-sgr)
    so = np.zeros_like(sw)  # estes experimentos de viscosidade não contêm óleo
    sg = 1-sw-so
    sg = np.where(sw >= 1-sgr, sgr, np.maximum(sg, sgr))
    return model.evaluate(sg, sw, so, np.full_like(sw, ut), np.zeros_like(sw), Cs=np.full_like(sw, Cs))


def predict_local(theta, fgs, flags, velocities, concentrations):
    models = {flag: make_model(theta, flag) for flag in (True, False)}
    viscosities, saturations = [], []
    for fg, foam, ut, Cs in zip(fgs, flags, velocities, concentrations):
        model = models[bool(foam)]
        swc, sgr = model.relPerm.swc, model.relPerm.sgr
        if fg == 1:
            sw = swc
        elif fg == 0:
            sw = 1-sgr
        else:
            sw = brentq(lambda s: float(evaluate_sw(model, s, ut, Cs)[0][0])-fg, swc, 1-sgr, xtol=1e-13)
        result = evaluate_sw(model, sw, ut, Cs)
        viscosities.append(float(result[2][0])*1000)
        saturations.append(sw)
    return np.array(viscosities), np.array(saturations)