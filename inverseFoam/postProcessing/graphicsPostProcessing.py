import numpy as np
import matplotlib.pyplot as plt

from inverseFoam.postProcessing.dataPostProcessing import *

def plot_F_histogram(F, ax=None, bins=50, xlabel = None):
    """
    Plot a histogram of values with mean and median lines.

    Parameters
    ----------
    F : array-like
        1D array values
    ax : matplotlib.axes.Axes, optional
        Axis to draw the plot on. If None, a new figure and axis are created.
    bins : int, optional
        Number of histogram bins
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=120)

    # Histogram
    ax.hist(
        F,
        bins=bins,
        density=False,
        alpha=0.75,
        edgecolor="black"
    )

    # Statistics
    ax.axvline(F.mean(), linestyle="--", linewidth=2,
               label=f"Mean = {F.mean():.3g}")
    ax.axvline(np.median(F), linestyle=":", linewidth=2,
               label=f"Median = {np.median(F):.3g}")

    # Labels and style
    if xlabel is None: ax.set_xlabel("Values", labelpad=15)
    else: ax.set_xlabel(xlabel, labelpad=15)
    ax.set_ylabel("Frequency")

    if xlabel == 'fmcap': ax.set_xscale('log')

    ax.tick_params(axis='x', rotation=45)

    ax.legend()
    ax.grid(True, alpha=0.3)

    if ax is None:
        plt.tight_layout()
        plt.show()

def plot_FQS(
    results,
    result_keys,
    keys_exp,
    exp_id,
    expDict,
    ax=None,
    fig=None,
    ci=90,
    model_color="tab:blue"
):
    """
    Plot model realizations, mean response, and confidence interval
    over experimental mu_app data.

    Parameters
    ----------
    results : dict
        Results dictionary
    result_keys : list
        Keys indexing individual model realizations
    keys_exp : list or dict
        Mapping of experiment IDs to result keys
    exp_id : int or str
        Experiment identifier
    expDict : dict
        Experimental data dictionary
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    fig : matplotlib.figure.Figure, optional
        Figure object (used if ax is None)
    ci : float, optional
        Confidence interval percentage (default 90)
    model_color : str, optional
        Color for model curves
    """

    # --------------------------------------------------
    # Figure / axis handling
    # --------------------------------------------------
    if ax is None:
        fig = fig or plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)

    # --------------------------------------------------
    # Collect model samples
    # --------------------------------------------------
    fg_samples = []
    muapp_samples = []

    for k in result_keys:
        res = results[k]
        fg = np.asarray(res[keys_exp[exp_id]]['fg'])
        muapp = np.asarray(res[keys_exp[exp_id]]['muapp'])

        fg_samples.append(fg)
        muapp_samples.append(muapp)

    fg_samples = np.asarray(fg_samples)
    muapp_samples = np.asarray(muapp_samples)

    fg_ref = fg_samples[0]

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------
    alpha = (100 - ci) / 2
    lo = alpha
    hi = 100 - alpha

    muapp_mean = np.mean(muapp_samples, axis=0)
    muapp_lo = np.percentile(muapp_samples, lo, axis=0)
    muapp_hi = np.percentile(muapp_samples, hi, axis=0)

    # --------------------------------------------------
    # Plot model realizations
    # --------------------------------------------------
    for j in range(muapp_samples.shape[0]):
        ax.plot(
            fg_ref,
            muapp_samples[j],
            color=model_color,
            alpha=0.15,
            linewidth=1
        )

    # Confidence interval
    ax.fill_between(
        fg_ref,
        muapp_lo,
        muapp_hi,
        color=model_color,
        alpha=0.3,
        label=f"Model {ci}% CI"
    )

    # Mean model
    ax.plot(
        fg_ref,
        muapp_mean,
        color=model_color,
        linewidth=2.5,
        label="Model mean"
    )

    # --------------------------------------------------
    # Experimental data
    # --------------------------------------------------
    (
        sg_exp,
        sw_exp,
        so_exp,
        ut_exp,
        ug_exp,
        fg_exp,
        muapp_exp
    ) = unpack_expDataDict3ph(expDict)

    std_muapp_exp = expDict["std_muApp"]

    ax.errorbar(
        fg_exp,
        muapp_exp,
        yerr=std_muapp_exp,
        fmt="o",
        color="black",
        capsize=4,
        markersize=6,
        label="Experimental data"
    )

    # --------------------------------------------------
    # Styling
    # --------------------------------------------------
    ax.set_xlabel("Gas fractional flow $f_g$")
    ax.set_ylabel("Apparent viscosity $\\mu_{app}$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig, ax

def plot_FRS(
    results,
    result_keys,
    keys_exp,
    exp_id,
    expDict,
    ax=None,
    fig=None,
    ci=90,
    model_color="tab:blue"
):
    """
    Plot mu_app vs total velocity u_t for model realizations,
    mean response, and confidence interval, over experimental data.

    Parameters
    ----------
    results : dict
        Results dictionary
    result_keys : list
        Keys indexing individual model realizations
    keys_exp : list or dict
        Mapping of experiment IDs to result keys
    exp_id : int or str
        Experiment identifier
    expDict : dict
        Experimental data dictionary
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    fig : matplotlib.figure.Figure, optional
        Figure object (used if ax is None)
    ci : float, optional
        Confidence interval percentage (default 90)
    model_color : str, optional
        Color for model curves
    """

    # --------------------------------------------------
    # Figure / axis handling
    # --------------------------------------------------
    if ax is None:
        fig = fig or plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)

    # --------------------------------------------------
    # Collect model samples
    # --------------------------------------------------
    ut_samples = []
    muapp_samples = []

    for k in result_keys:
        res = results[k]

        ut = np.asarray(res[keys_exp[exp_id]]['ut'])
        muapp = np.asarray(res[keys_exp[exp_id]]['muapp'])

        ut_samples.append(ut)
        muapp_samples.append(muapp)

    ut_samples = np.asarray(ut_samples)
    muapp_samples = np.asarray(muapp_samples)

    ut_ref = ut_samples[0]

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------
    alpha = (100 - ci) / 2
    lo = alpha
    hi = 100 - alpha

    muapp_mean = np.mean(muapp_samples, axis=0)
    muapp_lo = np.percentile(muapp_samples, lo, axis=0)
    muapp_hi = np.percentile(muapp_samples, hi, axis=0)

    # --------------------------------------------------
    # Plot model realizations
    # --------------------------------------------------
    for j in range(muapp_samples.shape[0]):
        ax.plot(
            ut_ref,
            muapp_samples[j],
            color=model_color,
            alpha=0.15,
            linewidth=1
        )

    # Confidence interval
    ax.fill_between(
        ut_ref,
        muapp_lo,
        muapp_hi,
        color=model_color,
        alpha=0.3,
        label=f"Model {ci}% CI"
    )

    # Mean model
    ax.plot(
        ut_ref,
        muapp_mean,
        color=model_color,
        linewidth=2.5,
        label="Model mean"
    )

    # --------------------------------------------------
    # Experimental data
    # --------------------------------------------------
    (
        sg_exp,
        sw_exp,
        so_exp,
        ut_exp,
        ug_exp,
        fg_exp,
        muapp_exp
    ) = unpack_expDataDict3ph(expDict)

    std_muapp_exp = expDict["std_muApp"]

    ax.errorbar(
        ut_exp,
        muapp_exp,
        yerr=std_muapp_exp,
        fmt="o",
        color="black",
        capsize=4,
        markersize=6,
        label="Experimental data"
    )

    # --------------------------------------------------
    # Styling
    # --------------------------------------------------
    ax.set_xlabel("Total superficial velocity $u_t$")
    ax.set_ylabel("Apparent viscosity $\\mu_{app}$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig, ax

def plot_profile_likelihood(
    xRange,
    xPL,
    alpha,
    ax=None,
    fig=None,
    color="black",
    linewidth=2.5,
    marker=None,
    label=None
):
    """
    Plot a profile likelihood curve.

    Parameters
    ----------
    xRange : array-like
        Parameter values
    xPL : array-like
        Profile likelihood values
    alpha: float
        quantile of chi_square distribution for confidence level
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    fig : matplotlib.figure.Figure, optional
        Figure object (used if ax is None)
    color : str, optional
        Line color
    linewidth : float, optional
        Line width
    marker : str, optional
        Marker style (e.g. 'o', None)
    label : str, optional
        Legend label
    """

    diff, delta_alpha = eval_confidenceLevel(alpha, xPL)

    # --------------------------------------------------
    # Figure / axis handling
    # --------------------------------------------------
    if ax is None:
        fig = fig or plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    ax.plot(
        xRange,
        diff,
        color=color,
        linewidth=linewidth,
        marker=marker
    )

    ax.axhline(delta_alpha, linestyle="--", linewidth=2)
    ax.set_ylim([-0.02,1])

    # --------------------------------------------------
    # Styling
    # --------------------------------------------------
    ax.set_xlabel(label)
    ax.set_ylabel(r"Confidence level, $\alpha$")
    ax.grid(True, alpha=0.3)

    return fig, ax