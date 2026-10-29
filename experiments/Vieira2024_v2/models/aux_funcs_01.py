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

def make_initial(seed, initial_z, zlo, zhi, start_margin, so_obs, SAT_EPS):
    rng = np.random.default_rng(seed)
    z = initial_z.copy()
    z[:4] = np.clip(z[:4] + rng.normal(0, 2.5, 4), zlo[:4] + start_margin[:4], zhi[:4] - start_margin[:4])
    floil = rng.uniform(0., float(so_obs.min()))
    fmoil = rng.uniform(max(float(so_obs.max()), floil + SAT_EPS), 1.)
    z[4:] = [rng.uniform(zlo[4], zhi[4]), floil, (fmoil - floil) / (1 - floil)]
    return np.clip(z, zlo + start_margin, zhi - start_margin)
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

def evaluate_oil_state(theta, foam_module_oil, sw, sg, so, inputs, concentrations, FMSURF):
    """Evaluate oil observations using aqueous concentrations and FMSURF in mg/L."""
    oil_relperm = foam_module_oil.coreyBrooks3ph
    fdry = 0.5 + np.arctan(theta[1] * (np.asarray(sw) - theta[2])) / np.pi
    foil = predict_oil(theta, foam_module_oil, so)
    fsurf = np.minimum(np.asarray(concentrations, dtype=float) / FMSURF, 1.0) ** theta[3]
    mrf = 1.0 + theta[0] * fdry * foil * fsurf
    lw = oil_relperm.krw(sw) / inputs["fluids"]["muw"]
    lo = oil_relperm.kro(so) / inputs["fluids"]["muo"]
    lg = oil_relperm.krg(sg) / (inputs["fluids"]["mug"] * mrf)
    total = lw + lo + lg
    return lg / total, 1000 / total

def predict_oil_viscosity(theta, foam_module_oil, sw_oil_obs, sg_oil_obs, so_oil_obs, inputs, concentrations, FMSURF):
    return evaluate_oil_state(theta, foam_module_oil, sw_oil_obs, sg_oil_obs, so_oil_obs, inputs, concentrations, FMSURF)[1]

def residual_blocks(theta, foam_module_noOil, point_weights, mu_obs, so_obs, mu_scale, fgs, flags, velocities, concentrations, fixed_rp, FMSURF, config,
                    foam_module_oil, mu_oil_obs, mu_oil_scale, sw_oil_obs, sg_oil_obs, so_oil_obs, fg_oil_obs, cs_oil_obs):
    predicted, _ = predict(theta, fgs, flags, velocities, concentrations, foam_module_noOil, fixed_rp, FMSURF, config)
    surf_r = np.sqrt(1.0*point_weights)*(mu_obs-predicted)/mu_scale

    fg_predicted, mu_predicted = evaluate_oil_state(
        theta, foam_module_oil, sw_oil_obs, sg_oil_obs, so_oil_obs, config, cs_oil_obs, FMSURF,
    )
    oil_r = np.sqrt(1.0) * (mu_oil_obs - mu_predicted) / mu_oil_scale
    oil_fg_r = np.sqrt(0.0) * (np.asarray(fg_oil_obs) - fg_predicted)

    return surf_r, oil_r, oil_fg_r

def residual(z, foam_module_noOil, point_weights, mu_obs, so_obs, mu_scale, fgs, flags, velocities, concentrations, fixed_rp, FMSURF, config, 
             foam_module_oil, mu_oil_obs, mu_oil_scale, sw_oil_obs, sg_oil_obs, so_oil_obs, fg_oil_obs, cs_oil_obs):
    return np.concatenate(residual_blocks(decode(z), foam_module_noOil, point_weights, mu_obs, so_obs, mu_scale, fgs, flags, velocities, concentrations, fixed_rp, FMSURF, config,
                                          foam_module_oil, mu_oil_obs, mu_oil_scale, sw_oil_obs, sg_oil_obs, so_oil_obs, fg_oil_obs, cs_oil_obs))

def objective_components(theta, foam_module_noOil, point_weights, mu_obs, so_obs, mu_scale, fgs, flags, velocities, concentrations, fixed_rp, FMSURF, config,
                         foam_module_oil, mu_oil_obs, mu_oil_scale, sw_oil_obs, sg_oil_obs, so_oil_obs, fg_oil_obs, cs_oil_obs):
    surf_r, oil_r, oil_fg_r = residual_blocks(theta, foam_module_noOil, point_weights, mu_obs, so_obs, mu_scale, fgs, flags, velocities, concentrations, fixed_rp, FMSURF, config,
                    foam_module_oil, mu_oil_obs, mu_oil_scale, sw_oil_obs, sg_oil_obs, so_oil_obs, fg_oil_obs, cs_oil_obs)
    surf_sse, oil_sse = float(surf_r @ surf_r), float(oil_r @ oil_r)
    oil_fg_sse = float(oil_fg_r @ oil_fg_r)
    return dict(surf_SSE=surf_sse, oil_SSE=oil_sse, oil_fg_SSE=oil_fg_sse,
                weighted_normalized_SSE=surf_sse + oil_sse + oil_fg_sse)

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

def predict_oil_at_fg(theta, so_values, fixed_rp_oil, foam_module_oil, inputs, concentration, FMSURF, fg_target=0.5, fg_tolerance=1e-6, sw_bounds=None):
    """Solve Sw at fixed So and fg, optionally within per-point Sw bounds.

    Bounds are intersected with the residual saturation limits. States without
    a root inside this interval remain NaN; there is no global fallback.
    """
    so_values = np.asarray(so_values, dtype=float)
    if sw_bounds is not None:
        sw_lower = np.broadcast_to(np.asarray(sw_bounds[0], dtype=float), so_values.shape)
        sw_upper = np.broadcast_to(np.asarray(sw_bounds[1], dtype=float), so_values.shape)
    fg_values = np.full(so_values.shape, np.nan)
    sw_values = np.full(so_values.shape, np.nan)
    mu_values = np.full(so_values.shape, np.nan)
    for i, so in enumerate(so_values):
        sw_min = fixed_rp_oil["swc"]
        sw_max = 1 - so - fixed_rp_oil["sgr"]
        if sw_bounds is not None:
            if not np.isfinite(sw_lower[i]) or not np.isfinite(sw_upper[i]):
                continue
            sw_min = max(sw_min, sw_lower[i])
            sw_max = min(sw_max, sw_upper[i])
        if so < fixed_rp_oil["sor"] or sw_max < sw_min:
            continue

        def state(sw):
            sg = max(1 - so - sw, fixed_rp_oil["sgr"])
            fg, mu = evaluate_oil_state(theta, foam_module_oil, sw, sg, so, inputs, concentration, FMSURF)
            return float(fg[0]), float(mu[0])

        if state(sw_min)[0] < fg_target or state(sw_max)[0] > fg_target:
            continue
        sw = brentq(lambda sw: state(sw)[0] - fg_target,
                    sw_min, sw_max, xtol=1e-12)
        fg, mu = state(sw)
        if np.isclose(fg, fg_target, atol=fg_tolerance, rtol=0):
            fg_values[i], sw_values[i], mu_values[i] = fg, sw, mu
    return fg_values, sw_values, mu_values
