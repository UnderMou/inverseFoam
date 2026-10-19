"""Vectorized STARS/Corey forward models for the local PyMC3 notebook.

Fixed data are supplied explicitly through the parameters dictionary. No cloud
client, YAML files, notebook state or saved estimates are needed by this module.
The numerical model matches the former UQPyLab callbacks.
"""

import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(function):
        return function


FOAM_NAMES = ["fmmob", "epdry", "fmdry", "epsurf", "epoil", "fmoil", "floil"]
COORDINATE_NAMES = ["log_fmmob", "log_epdry", "fmdry", "log_epsurf", "log_epoil", "floil", "qoil"]


def decode_parameters(x):
    """Map the seven prior coordinates to physical foam parameters."""
    x = np.asarray(x, dtype=float)
    if x.shape[-1] != 7 or not np.isfinite(x).all():
        raise ValueError("Expected seven finite prior coordinates.")
    theta = np.empty_like(x)
    theta[..., 0] = np.exp(x[..., 0])
    theta[..., 1] = np.exp(x[..., 1])
    theta[..., 2] = x[..., 2]
    theta[..., 3] = np.exp(x[..., 3])
    theta[..., 4] = np.exp(x[..., 4])
    theta[..., 6] = x[..., 5]
    theta[..., 5] = x[..., 5] + (1-x[..., 5])*x[..., 6]
    return theta


def encode_parameters(theta):
    """Inverse transformation, including the ordering of oil thresholds."""
    theta = np.asarray(theta, dtype=float)
    if theta.shape[-1] != 7 or not np.isfinite(theta).all():
        raise ValueError("Expected seven finite physical parameters.")
    if np.any(theta[..., [0, 1, 3, 4]] <= 0) or np.any(theta[..., 6] < 0) or np.any(
            theta[..., 6] >= theta[..., 5]) or np.any(theta[..., 5] > 1):
        raise ValueError("Require positive foam scales and 0 <= floil < fmoil <= 1.")
    x = np.empty_like(theta)
    x[..., [0, 1, 3, 4]] = np.log(theta[..., [0, 1, 3, 4]])
    x[..., 2] = theta[..., 2]
    x[..., 5] = theta[..., 6]
    x[..., 6] = (theta[..., 5]-theta[..., 6])/(1-theta[..., 6])
    return x


@njit
def _state(theta, t, fsurf, rp, muw, mug):
    nw, ng, krw0, krg0, swc, sgr, sor = rp
    span = 1-swc-sgr
    log_s = -np.logaddexp(0., -t)
    log_g = -np.logaddexp(0., t)
    sw = swc + span*np.exp(log_s)
    log_scale = np.log(span/(span-sor))
    lw = np.log(krw0/muw) + nw*(log_s+log_scale)
    lg = np.log(krg0/mug) + ng*(log_g+log_scale)
    fdry = .5 + np.arctan(theta[1]*(sw-theta[2]))/np.pi
    # These viscosity experiments have So=0, kro=0 and Foil(0)=1.
    lg -= np.log1p(theta[0]*fdry*fsurf)
    return lw, lg, sw


@njit
def _viscosity_batch(thetas, fgs, concentrations, rp, fmsurf, muw, mug):
    nw, ng, krw0, krg0, swc, sgr, sor = rp
    scale = (1-swc-sgr)/(1-swc-sgr-sor)
    viscosities = np.empty((len(thetas), len(fgs)))
    saturations = np.empty_like(viscosities)
    for row in range(len(thetas)):
        theta = thetas[row]
        for i in range(len(fgs)):
            fg, cs = fgs[i], concentrations[i]
            fsurf = (cs/fmsurf)**theta[3] if cs < fmsurf else 1.
            if fg == 0:
                viscosities[row, i] = 1000*muw/(krw0*scale**nw)
                saturations[row, i] = 1-sgr
            elif fg == 1:
                fdry = .5+np.arctan(theta[1]*(swc-theta[2]))/np.pi
                viscosities[row, i] = 1000*mug/(krg0*scale**ng)*(1+theta[0]*fdry*fsurf)
                saturations[row, i] = swc
            else:
                target = np.log(fg)-np.log1p(-fg)
                lo, hi = -10000., 10000.
                for _ in range(65):
                    mid = (lo+hi)/2
                    lw, lg, sw = _state(theta, mid, fsurf, rp, muw, mug)
                    if lg-lw > target:
                        lo = mid
                    else:
                        hi = mid
                lw, lg, sw = _state(theta, (lo+hi)/2, fsurf, rp, muw, mug)
                viscosities[row, i] = 1000*np.exp(-np.logaddexp(lw, lg))
                saturations[row, i] = sw
    return viscosities, saturations


def apparent_viscosity(thetas, fgs, concentrations, parameters):
    """Physical viscosities in cP and Sw, always at So=0 (batch x observations)."""
    thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
    fgs, concentrations = np.asarray(fgs, dtype=float), np.asarray(concentrations, dtype=float)
    if thetas.shape[1] not in (4, 7) or not np.isfinite(thetas).all():
        raise ValueError("Supply four or seven physical foam parameters per row.")
    if np.any(thetas[:, [0, 1, 3]] <= 0):
        raise ValueError("Positive fmmob, epdry and epsurf are required.")
    if fgs.ndim != 1 or fgs.shape != concentrations.shape or not np.isfinite(fgs).all() or np.any((fgs < 0) | (fgs > 1)):
        raise ValueError("Invalid gas fractions or concentration array shape.")
    if not np.isfinite(concentrations).all() or np.any(concentrations < 0):
        raise ValueError("Concentrations must be finite and nonnegative.")
    rp_dict = parameters["relPerm"]
    rp = np.array([rp_dict[k] for k in ("nw_", "ng_", "krw0", "krg0", "swc", "sgr", "sor")], dtype=float)
    if not np.isfinite(rp).all() or min(rp[:4]) <= 0 or min(rp[4:]) < 0 or sum(rp[4:]) >= 1:
        raise ValueError("Invalid fixed Corey parameters.")
    if float(parameters.get("So_viscosity", 0.)) != 0.:
        raise ValueError("The viscosity experiments require So=0.")
    fmsurf = float(parameters["fmsurf"])
    muw, mug = float(parameters["muw"]), float(parameters["mug"])
    if min(fmsurf, muw, mug) <= 0:
        raise ValueError("fmsurf and fluid viscosities must be positive.")
    return _viscosity_batch(thetas[:, :4], fgs, concentrations, rp, fmsurf, muw, mug)


def foil(thetas, saturations):
    """Foil for batches of seven physical parameters, using the local STARS law."""
    thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
    saturations = np.atleast_1d(np.asarray(saturations, dtype=float))
    if thetas.shape[1] != 7 or not np.isfinite(thetas).all():
        raise ValueError("Supply seven finite physical parameters per row.")
    epoil, fmoil, floil = thetas[:, 4, None], thetas[:, 5, None], thetas[:, 6, None]
    if np.any(epoil <= 0) or np.any(floil < 0) or np.any(floil >= fmoil) or np.any(fmoil > 1):
        raise ValueError("Require epoil > 0 and 0 <= floil < fmoil <= 1.")
    if saturations.ndim != 1 or not np.isfinite(saturations).all() or np.any((saturations < 0) | (saturations > 1)):
        raise ValueError("Oil saturations must be between 0 and 1.")
    ratio = np.clip((fmoil-saturations[None, :])/(fmoil-floil), 0., 1.)
    return ratio**epoil


def viscosity_model(x, parameters):
    """Four prior coordinates -> 21 weighted normalized outputs."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[1] != 4:
        raise ValueError("Supply four viscosity prior coordinates.")
    theta = x.copy()
    theta[:, [0, 1, 3]] = np.exp(x[:, [0, 1, 3]])
    mu, _ = apparent_viscosity(theta, parameters["fg"], parameters["Cs_mg_L"], parameters)
    return mu*np.sqrt(np.asarray(parameters["weights"], dtype=float))[None, :]/float(parameters["mu_scale"])


def oil_model(x, parameters):
    """Three prior coordinates -> five normalized Foil outputs."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[1] != 3:
        raise ValueError("Supply three oil prior coordinates.")
    theta = np.ones((len(x), 7))
    theta[:, 4] = np.exp(x[:, 0])
    theta[:, 6] = x[:, 1]
    theta[:, 5] = x[:, 1]+(1-x[:, 1])*x[:, 2]
    return foil(theta, parameters["So"])*np.sqrt(float(parameters["weight"]))/float(parameters["f2_scale"])
