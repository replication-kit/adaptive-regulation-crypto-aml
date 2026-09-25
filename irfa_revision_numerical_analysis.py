"""
Complete reproducibility script for the second revision of:

    Noisy Regulatory Updating and Persistent Fragmentation
    in Crypto-Asset Anti-Money Laundering

Running this single file reproduces the numerical analysis used in the
second revision and generates the manuscript figures that depend on the
numerical model:

    Figure 2   Representative baseline path (first 120 periods)
    Figure 3   Long-horizon 100-period block averages
    Figure 4A  Sensitivity to observation noise
    Figure 4B  Sensitivity to relocation responsiveness
    Figure 4C  Sensitivity to regulatory adjustment speed
    Figure 4   Combined three-panel sensitivity figure (convenience copy)
    Figure C1  Smooth versus fixed-step updating rule

It also writes the numerical tables used for the long-horizon,
mechanism-ablation, sensitivity, and robustness analyses.

Default output directory: ./irfa_revision_results
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ============================================================
# Baseline design
# ============================================================

J = 5
T = 2000
M = 500
L_TOTAL = 1.0
BASE_SEED = 123

R0 = np.array([0.30, 0.35, 0.40, 0.45, 0.50], dtype=float)

ALPHA_C = -1.00
BETA_C = 3.00
KAPPA_BASE = np.array([0.00, 0.20, 0.40, 0.60, 0.80], dtype=float)

A_BASE = np.array([-0.20, -0.10, 0.00, 0.10, 0.20], dtype=float)
THETA_R0 = 1.50
THETA_C0 = 1.00

ALPHA_D = -1.50
DELTA_R = 2.00
DELTA_C = 1.00

SIGMA_BASE = 0.04

ALPHA_R = 0.00
BETA_R = 12.00
TAU = 0.10
RHO_BASE = 0.35

# Alternative fixed-step updating rule (Appendix C)
ETA = 0.05
TAU_LOWER = 0.08
TAU_UPPER = 0.12

# Principal long-horizon evaluation window
LR_START = 1001
LR_END = 2000

# Figure 2 displays only the first 120 periods of a T=2000 run.
FIGURE2_END = 120

WINDOWS = {
    "60-120": (60, 120),
    "201-500": (201, 500),
    "501-1000": (501, 1000),
    "1001-1500": (1001, 1500),
    "1501-2000": (1501, 2000),
    "1001-2000": (1001, 2000),
}

SIGMA_VALUES = np.array([0.00, 0.02, 0.04, 0.06, 0.08, 0.10])
OMEGA_VALUES = np.array([0.00, 0.50, 1.00, 1.50, 2.00])
RHO_VALUES = np.array([0.10, 0.20, 0.35, 0.50, 0.70])


# ============================================================
# Model functions
# ============================================================


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_common_shocks(T=T, M=M, J=J, base_seed=BASE_SEED):
    """Generate the same jurisdiction-time shocks for all counterfactuals."""
    eps = np.empty((M, T + 1, J), dtype=float)
    for m in range(M):
        rng = np.random.default_rng(base_seed + m)
        eps[m] = rng.standard_normal((T + 1, J))
    return eps


def run_model(
    eps,
    sigma=SIGMA_BASE,
    kappa=KAPPA_BASE,
    a=A_BASE,
    theta_R=THETA_R0,
    theta_C=THETA_C0,
    rho=RHO_BASE,
    R_init=R0,
    rule="smooth",
    force_uniform_allocation=False,
):
    """
    Monte Carlo simulation vectorized over replications.

    Parameters
    ----------
    eps : ndarray, shape (M, T+1, J)
        Pre-generated standard-normal shocks. Reusing eps across cases
        implements common random numbers.

    Returns
    -------
    F_R, F_L : ndarrays, shape (M, T+1)
        Implementation and illicit-activity fragmentation measures.
    """
    M_local, T_plus_1, J_local = eps.shape
    T_local = T_plus_1 - 1

    R = np.tile(np.asarray(R_init, dtype=float), (M_local, 1))
    F_R = np.empty((M_local, T_plus_1), dtype=float)
    F_L = np.empty((M_local, T_plus_1), dtype=float)

    for t in range(T_plus_1):
        C = logistic(ALPHA_C + BETA_C * R - kappa)

        if force_uniform_allocation:
            L = np.full((M_local, J_local), L_TOTAL / J_local)
        else:
            attractiveness = np.exp(a - theta_R * R - theta_C * C)
            L = L_TOTAL * attractiveness / attractiveness.sum(axis=1, keepdims=True)

        lam = L / L_TOTAL
        D = logistic(ALPHA_D + DELTA_R * R + DELTA_C * C)
        S = D * L + sigma * eps[:, t, :]

        Rbar = R.mean(axis=1, keepdims=True)
        F_R[:, t] = np.mean((R - Rbar) ** 2, axis=1)
        F_L[:, t] = np.sum((lam - 1.0 / J_local) ** 2, axis=1)

        if t < T_local:
            if rule == "smooth":
                target = logistic(ALPHA_R + BETA_R * (S - TAU))
                R = (1.0 - rho) * R + rho * target
            elif rule == "fixed":
                step = (
                    (S > TAU_UPPER).astype(float)
                    - (S < TAU_LOWER).astype(float)
                )
                R = np.clip(R + ETA * step, 0.0, 1.0)
            else:
                raise ValueError("rule must be 'smooth' or 'fixed'")

    return F_R, F_L


def run_representative_path(eps_single):
    """Full baseline path used for Figure 2 (seed 123 / replication 0)."""
    T_local = eps_single.shape[0] - 1
    R = np.empty((T_local + 1, J), dtype=float)
    lam = np.empty((T_local + 1, J), dtype=float)
    R[0] = R0

    for t in range(T_local + 1):
        C = logistic(ALPHA_C + BETA_C * R[t] - KAPPA_BASE)
        attractiveness = np.exp(A_BASE - THETA_R0 * R[t] - THETA_C0 * C)
        L = L_TOTAL * attractiveness / attractiveness.sum()
        lam[t] = L / L_TOTAL
        D = logistic(ALPHA_D + DELTA_R * R[t] + DELTA_C * C)
        S = D * L + SIGMA_BASE * eps_single[t]

        if t < T_local:
            target = logistic(ALPHA_R + BETA_R * (S - TAU))
            R[t + 1] = (1.0 - RHO_BASE) * R[t] + RHO_BASE * target

    return R, lam


# ============================================================
# Numerical summaries
# ============================================================


def summarize_replication_averages(F_R, F_L, start, end):
    fr = F_R[:, start:end + 1].mean(axis=1)
    fl = F_L[:, start:end + 1].mean(axis=1)
    return {
        "FR_mean": fr.mean(),
        "FR_p10": np.percentile(fr, 10),
        "FR_p90": np.percentile(fr, 90),
        "FL_mean": fl.mean(),
        "FL_p10": np.percentile(fl, 10),
        "FL_p90": np.percentile(fl, 90),
    }


def baseline_window_analysis(F_R, F_L):
    rows = []
    for label, (start, end) in WINDOWS.items():
        row = {"window": label, "start": start, "end": end}
        row.update(summarize_replication_averages(F_R, F_L, start, end))
        rows.append(row)
    return pd.DataFrame(rows)


def block_analysis(F_R, F_L):
    rows = []
    # 18 non-overlapping 100-period blocks: 201-300, ..., 1901-2000
    for start in range(201, 1902, 100):
        end = start + 99
        row = {"block": f"{start}-{end}", "start": start, "end": end}
        row.update(summarize_replication_averages(F_R, F_L, start, end))
        rows.append(row)
    return pd.DataFrame(rows)


def mechanism_ablation(eps):
    """Five mechanism-ablation cases reported in the revised manuscript."""
    homogeneous_kappa = np.full(J, 0.40)
    homogeneous_a = np.zeros(J)

    cases = [
        (
            "A. Deterministic homogeneous reference",
            dict(
                sigma=0.0,
                kappa=homogeneous_kappa,
                a=homogeneous_a,
                theta_R=THETA_R0,
                theta_C=THETA_C0,
            ),
        ),
        (
            "B. Permanent heterogeneity only",
            dict(
                sigma=0.0,
                kappa=KAPPA_BASE,
                a=A_BASE,
                theta_R=THETA_R0,
                theta_C=THETA_C0,
            ),
        ),
        (
            "C. Noise only",
            dict(
                sigma=SIGMA_BASE,
                kappa=homogeneous_kappa,
                a=homogeneous_a,
                theta_R=THETA_R0,
                theta_C=THETA_C0,
            ),
        ),
        (
            "D. No endogenous relocation feedback",
            dict(
                sigma=SIGMA_BASE,
                kappa=KAPPA_BASE,
                a=A_BASE,
                theta_R=0.0,
                theta_C=0.0,
            ),
        ),
        (
            "E. Full baseline",
            dict(
                sigma=SIGMA_BASE,
                kappa=KAPPA_BASE,
                a=A_BASE,
                theta_R=THETA_R0,
                theta_C=THETA_C0,
            ),
        ),
    ]

    rows = []
    for label, kwargs in cases:
        F_R, F_L = run_model(eps, **kwargs)
        row = {"case": label}
        row.update(summarize_replication_averages(F_R, F_L, LR_START, LR_END))
        rows.append(row)

    return pd.DataFrame(rows)


def strict_no_spatial_channel(eps):
    F_R, F_L = run_model(
        eps,
        sigma=SIGMA_BASE,
        kappa=KAPPA_BASE,
        a=A_BASE,
        theta_R=THETA_R0,
        theta_C=THETA_C0,
        force_uniform_allocation=True,
    )
    row = {"case": "Strict no-spatial-channel: L_j,t = L/J"}
    row.update(summarize_replication_averages(F_R, F_L, LR_START, LR_END))
    return pd.DataFrame([row])


def sensitivity_analysis(eps, baseline_FR, baseline_FL):
    rows = []

    for value in SIGMA_VALUES:
        F_R, F_L = run_model(eps, sigma=float(value))
        row = {"parameter": "sigma", "value": value}
        row.update(summarize_replication_averages(F_R, F_L, LR_START, LR_END))
        row["FR_norm"] = row["FR_mean"] / baseline_FR
        row["FL_norm"] = row["FL_mean"] / baseline_FL
        rows.append(row)

    for value in OMEGA_VALUES:
        F_R, F_L = run_model(
            eps,
            theta_R=float(value * THETA_R0),
            theta_C=float(value * THETA_C0),
        )
        row = {"parameter": "omega", "value": value}
        row.update(summarize_replication_averages(F_R, F_L, LR_START, LR_END))
        row["FR_norm"] = row["FR_mean"] / baseline_FR
        row["FL_norm"] = row["FL_mean"] / baseline_FL
        rows.append(row)

    for value in RHO_VALUES:
        F_R, F_L = run_model(eps, rho=float(value))
        row = {"parameter": "rho", "value": value}
        row.update(summarize_replication_averages(F_R, F_L, LR_START, LR_END))
        row["FR_norm"] = row["FR_mean"] / baseline_FR
        row["FL_norm"] = row["FL_mean"] / baseline_FL
        rows.append(row)

    return pd.DataFrame(rows)


def updating_rule_analysis(eps, baseline_FR, baseline_FL):
    rows = []
    for rule in ["smooth", "fixed"]:
        F_R, F_L = run_model(eps, rule=rule)
        row = {"rule": rule}
        row.update(summarize_replication_averages(F_R, F_L, LR_START, LR_END))
        row["FR_norm"] = row["FR_mean"] / baseline_FR
        row["FL_norm"] = row["FL_mean"] / baseline_FL
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# Figure generation (monochrome)
# ============================================================


def save_figure(fig, outdir, stem):
    """Save each figure as EPS, PDF, and PNG."""
    fig.savefig(outdir / f"{stem}.eps", format="eps", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_figure2(eps, outdir):
    R, lam = run_representative_path(eps[0])

    time = np.arange(FIGURE2_END + 1)
    R_show = R[: FIGURE2_END + 1]
    lam_show = lam[: FIGURE2_END + 1]

    R_mean = R_show.mean(axis=1)
    R_min = R_show.min(axis=1)
    R_max = R_show.max(axis=1)
    lam_min = lam_show.min(axis=1)
    lam_max = lam_show.max(axis=1)
    equal_share = np.full(FIGURE2_END + 1, 1.0 / J)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

    axes[0].fill_between(
        time, R_min, R_max, color="0.85", edgecolor="0.6", linewidth=0.8
    )
    axes[0].plot(time, R_mean, color="black", linewidth=2.0)
    axes[0].set_title("Panel A. Regulatory implementation intensity")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel(r"$R$")
    axes[0].set_ylim(0, 1)
    axes[0].set_xlim(0, FIGURE2_END)
    axes[0].grid(True, color="0.85", linewidth=0.8)
    axes[0].legend(
        handles=[
            Line2D([0], [0], color="black", lw=2.0, label=r"Mean $\bar{R}_t$"),
            Patch(
                facecolor="0.85",
                edgecolor="0.6",
                label=r"Range $[\min_j R_{j,t},\max_j R_{j,t}]$",
            ),
        ],
        frameon=False,
        loc="lower right",
        fontsize=9,
    )

    axes[1].fill_between(
        time, lam_min, lam_max, color="0.85", edgecolor="0.6", linewidth=0.8
    )
    axes[1].plot(time, equal_share, color="black", linewidth=1.5, linestyle="--")
    axes[1].set_title("Panel B. Illicit-activity shares")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel(r"$\lambda$")
    axes[1].set_xlim(0, FIGURE2_END)
    margin = 0.02
    axes[1].set_ylim(max(0, lam_min.min() - margin), min(1, lam_max.max() + margin))
    axes[1].grid(True, color="0.85", linewidth=0.8)
    axes[1].legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="black",
                lw=1.5,
                linestyle="--",
                label=r"Equal allocation $1/J$",
            ),
            Patch(
                facecolor="0.85",
                edgecolor="0.6",
                label=r"Range $[\min_j \lambda_{j,t},\max_j \lambda_{j,t}]$",
            ),
        ],
        frameon=False,
        loc="upper right",
        fontsize=9,
    )

    save_figure(fig, outdir, "Figure2")


def make_figure3(blocks, outdir):
    mid = ((blocks["start"] + blocks["end"]) / 2).to_numpy()
    fr = blocks["FR_mean"].to_numpy()
    fl = blocks["FL_mean"].to_numpy()
    fr_err = np.vstack(
        [fr - blocks["FR_p10"].to_numpy(), blocks["FR_p90"].to_numpy() - fr]
    )
    fl_err = np.vstack(
        [fl - blocks["FL_p10"].to_numpy(), blocks["FL_p90"].to_numpy() - fl]
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.errorbar(
        mid,
        fr,
        yerr=fr_err,
        color="black",
        ecolor="black",
        marker="o",
        markerfacecolor="black",
        markeredgecolor="black",
        linestyle="-",
        linewidth=1.6,
        markersize=5,
        capsize=2.5,
        label=r"Implementation fragmentation, $\bar{F}^{R}_{b}$",
    )
    ax.errorbar(
        mid,
        fl,
        yerr=fl_err,
        color="black",
        ecolor="black",
        marker="s",
        markerfacecolor="white",
        markeredgecolor="black",
        linestyle="--",
        linewidth=1.6,
        markersize=5,
        capsize=2.5,
        label=r"Illicit-activity fragmentation, $\bar{F}^{L}_{b}$",
    )
    ax.set_xlabel("Simulation period")
    ax.set_ylabel("100-period average fragmentation")
    ax.set_xlim(200, 2000)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(True, color="0.88", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()

    save_figure(fig, outdir, "Figure3")


def _plot_sensitivity_panel(ax, d, title, xlabel, show_ylabel=True):
    x = d["value"].to_numpy()
    yR = d["FR_norm"].to_numpy()
    yL = d["FL_norm"].to_numpy()

    ax.plot(
        x,
        yR,
        color="black",
        linestyle="-",
        marker="o",
        markerfacecolor="black",
        markeredgecolor="black",
        linewidth=1.8,
        markersize=5.5,
        label=r"Implementation $\bar{F}^{R}$",
    )
    ax.plot(
        x,
        yL,
        color="black",
        linestyle="--",
        marker="s",
        markerfacecolor="white",
        markeredgecolor="black",
        linewidth=1.8,
        markersize=5.5,
        label=r"Illicit activity $\bar{F}^{L}$",
    )
    ax.axhline(1.0, color="0.45", linestyle=":", linewidth=1.1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel("Normalized long-horizon fragmentation")
    ax.set_ylim(bottom=0)
    ax.grid(True, color="0.85", linewidth=0.8)


def make_figure4(sensitivity, outdir):
    specs = [
        ("sigma", "Panel A. Observation noise", r"$\sigma$", "Figure4A"),
        (
            "omega",
            "Panel B. Relocation responsiveness",
            r"$\omega$",
            "Figure4B",
        ),
        (
            "rho",
            "Panel C. Regulatory adjustment speed",
            r"$\rho$",
            "Figure4C",
        ),
    ]

    # Separate panel files used by the revised manuscript.
    for param, title, xlabel, stem in specs:
        d = sensitivity[sensitivity["parameter"] == param].copy()
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        _plot_sensitivity_panel(ax, d, title, xlabel, show_ylabel=True)
        handles = [
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="-",
                marker="o",
                markerfacecolor="black",
                markeredgecolor="black",
                linewidth=1.8,
                label=r"Implementation $\bar{F}^{R}$",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                marker="s",
                markerfacecolor="white",
                markeredgecolor="black",
                linewidth=1.8,
                label=r"Illicit activity $\bar{F}^{L}$",
            ),
            Line2D(
                [0],
                [0],
                color="0.45",
                linestyle=":",
                linewidth=1.1,
                label="Baseline = 1",
            ),
        ]
        ax.legend(handles=handles, frameon=False, fontsize=8.5)
        fig.tight_layout()
        save_figure(fig, outdir, stem)

    # Combined three-panel copy for convenience and archival reproduction.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    for i, (param, title, xlabel, _) in enumerate(specs):
        d = sensitivity[sensitivity["parameter"] == param].copy()
        _plot_sensitivity_panel(axes[i], d, title, xlabel, show_ylabel=(i == 0))

    handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            marker="o",
            markerfacecolor="black",
            markeredgecolor="black",
            linewidth=1.8,
            label=r"Implementation $\bar{F}^{R}$",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            marker="s",
            markerfacecolor="white",
            markeredgecolor="black",
            linewidth=1.8,
            label=r"Illicit activity $\bar{F}^{L}$",
        ),
        Line2D(
            [0],
            [0],
            color="0.45",
            linestyle=":",
            linewidth=1.1,
            label="Baseline = 1",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    save_figure(fig, outdir, "Figure4")


def make_figureC1(updating, outdir):
    labels = ["Smooth\nupdating", "Fixed-step\nupdating"]
    x = np.arange(len(labels))
    offset = 0.08

    yR = updating["FR_norm"].to_numpy()
    yL = updating["FL_norm"].to_numpy()

    baseline_fr = updating.loc[updating["rule"] == "smooth", "FR_mean"].iloc[0]
    baseline_fl = updating.loc[updating["rule"] == "smooth", "FL_mean"].iloc[0]

    p10R = updating["FR_p10"].to_numpy() / baseline_fr
    p90R = updating["FR_p90"].to_numpy() / baseline_fr
    p10L = updating["FL_p10"].to_numpy() / baseline_fl
    p90L = updating["FL_p90"].to_numpy() / baseline_fl

    errR = np.vstack([yR - p10R, p90R - yR])
    errL = np.vstack([yL - p10L, p90L - yL])

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.errorbar(
        x - offset,
        yR,
        yerr=errR,
        fmt="o-",
        color="black",
        ecolor="black",
        markerfacecolor="black",
        markeredgecolor="black",
        linewidth=1.8,
        markersize=5,
        capsize=4,
        label=r"Implementation $\bar{F}^{R}$",
    )
    ax.errorbar(
        x + offset,
        yL,
        yerr=errL,
        fmt="s--",
        color="black",
        ecolor="black",
        markerfacecolor="white",
        markeredgecolor="black",
        linewidth=1.8,
        markersize=5,
        capsize=4,
        label=r"Illicit activity $\bar{F}^{L}$",
    )
    ax.axhline(1.0, color="0.45", linestyle=":", linewidth=1.1, label="Baseline = 1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Normalized long-horizon fragmentation")
    ax.set_xlabel("Updating rule")
    ax.set_ylim(bottom=0)
    ax.grid(True, color="0.85", linewidth=0.8)
    ax.legend(frameon=False, loc="upper left", fontsize=9)

    save_figure(fig, outdir, "FigureC1")


# ============================================================
# Table / metadata outputs
# ============================================================


def write_mechanism_latex(mechanisms, outdir):
    label_map = {
        "A. Deterministic homogeneous reference": "A. Deterministic homogeneous reference",
        "B. Permanent heterogeneity only": "B. Heterogeneity without noise",
        "C. Noise only": "C. Noise without permanent heterogeneity",
        "D. No endogenous relocation feedback": (
            "D. Full environment without endogenous relocation feedback"
        ),
        "E. Full baseline": "E. Full baseline",
    }

    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\caption{Mechanism-ablation results}",
        r"\label{tab:mechanism-ablation}",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Case & $\bar{F}^{R}$ & $\bar{F}^{L}$ \\",
        r"\midrule",
    ]
    for _, row in mechanisms.iterrows():
        case = label_map[row["case"]]
        lines.append(f"{case} & {row['FR_mean']:.5f} & {row['FL_mean']:.5f} \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{minipage}{0.94\textwidth}",
        r"\footnotesize",
        (
            r"\textit{Notes:} Each entry is the Monte Carlo mean of the corresponding "
            r"fragmentation measure averaged over $t=1001,\ldots,2000$. All cases use the "
            r"same initial implementation vector $R_{j,0}=(0.30,0.35,0.40,0.45,0.50)$. "
            r"The homogeneous cases set $a_j=0$ and $\kappa_j=0.40$ for all jurisdictions. "
            r"Case D retains baseline noise and permanent heterogeneity but sets "
            r"$\theta_R=\theta_C=0$, eliminating endogenous relocation responses to "
            r"implementation and compliance conditions. The reported cases are "
            r"mechanism-ablation counterfactuals, not an additive variance decomposition."
        ),
        r"\end{minipage}",
        r"\end{table}",
    ]
    (outdir / "Table2_mechanism_ablation.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_manifest(outdir):
    manifest = f"""IRFA second-revision reproducibility run

J = {J}
T = {T}
M = {M}
Base seed = {BASE_SEED}
Principal long-horizon window = {LR_START}-{LR_END}
Figure 2 displayed window = 0-{FIGURE2_END}

Baseline parameters
R0 = {R0.tolist()}
alpha_C = {ALPHA_C}
beta_C = {BETA_C}
kappa = {KAPPA_BASE.tolist()}
a = {A_BASE.tolist()}
theta_R = {THETA_R0}
theta_C = {THETA_C0}
alpha_D = {ALPHA_D}
delta_R = {DELTA_R}
delta_C = {DELTA_C}
sigma = {SIGMA_BASE}
alpha_R = {ALPHA_R}
beta_R = {BETA_R}
tau = {TAU}
rho = {RHO_BASE}

Alternative fixed-step updating
eta = {ETA}
tau_lower = {TAU_LOWER}
tau_upper = {TAU_UPPER}

Generated manuscript figures
Figure2.eps/pdf/png
Figure3.eps/pdf/png
Figure4A.eps/pdf/png
Figure4B.eps/pdf/png
Figure4C.eps/pdf/png
Figure4.eps/pdf/png (combined convenience copy)
FigureC1.eps/pdf/png

Generated numerical outputs
baseline_window_summary.csv
baseline_100_period_blocks.csv
mechanism_ablation_summary.csv
strict_no_spatial_channel_summary.csv
sensitivity_long_horizon.csv
updating_rule_long_horizon.csv
reproduction_check.csv
Table2_mechanism_ablation.tex
"""
    (outdir / "run_manifest.txt").write_text(manifest, encoding="utf-8")


# ============================================================
# Main execution
# ============================================================


def main(outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Generating common random numbers...")
    eps = make_common_shocks()

    print("Running full baseline...")
    F_R_base, F_L_base = run_model(eps)

    windows = baseline_window_analysis(F_R_base, F_L_base)
    blocks = block_analysis(F_R_base, F_L_base)

    baseline_long = summarize_replication_averages(
        F_R_base, F_L_base, LR_START, LR_END
    )
    baseline_FR = baseline_long["FR_mean"]
    baseline_FL = baseline_long["FL_mean"]

    print("Running mechanism ablations...")
    mechanisms = mechanism_ablation(eps)

    print("Running strict no-spatial-channel counterfactual...")
    strict = strict_no_spatial_channel(eps)

    print("Running long-horizon sensitivity analysis...")
    sensitivity = sensitivity_analysis(eps, baseline_FR, baseline_FL)

    print("Running smooth vs fixed-step updating comparison...")
    updating = updating_rule_analysis(eps, baseline_FR, baseline_FL)

    # Save numerical outputs.
    windows.to_csv(outdir / "baseline_window_summary.csv", index=False)
    blocks.to_csv(outdir / "baseline_100_period_blocks.csv", index=False)
    mechanisms.to_csv(outdir / "mechanism_ablation_summary.csv", index=False)
    strict.to_csv(outdir / "strict_no_spatial_channel_summary.csv", index=False)
    sensitivity.to_csv(outdir / "sensitivity_long_horizon.csv", index=False)
    updating.to_csv(outdir / "updating_rule_long_horizon.csv", index=False)

    # Reproduction check against the previous manuscript's baseline window.
    old_window = windows.loc[windows["window"] == "60-120"].iloc[0]
    validation = pd.DataFrame(
        [
            {
                "quantity": "FR 60-120",
                "recomputed": old_window["FR_mean"],
                "previous_manuscript_rounded": 0.00348,
            },
            {
                "quantity": "FL 60-120",
                "recomputed": old_window["FL_mean"],
                "previous_manuscript_rounded": 0.00469,
            },
        ]
    )
    validation["difference"] = (
        validation["recomputed"] - validation["previous_manuscript_rounded"]
    )
    validation.to_csv(outdir / "reproduction_check.csv", index=False)

    # Generate all model-based manuscript figures.
    print("Generating Figure 2...")
    make_figure2(eps, outdir)

    print("Generating Figure 3...")
    make_figure3(blocks, outdir)

    print("Generating Figure 4A-C and combined Figure 4...")
    make_figure4(sensitivity, outdir)

    print("Generating Figure C1...")
    make_figureC1(updating, outdir)

    write_mechanism_latex(mechanisms, outdir)
    write_manifest(outdir)

    print("\nReproduction check:")
    print(validation.to_string(index=False))

    print("\nBaseline window comparison:")
    print(windows.to_string(index=False))

    print("\nMechanism ablation:")
    print(mechanisms.to_string(index=False))

    print("\nSensitivity results:")
    print(sensitivity.to_string(index=False))

    print("\nUpdating-rule comparison:")
    print(updating.to_string(index=False))

    print(f"\nAll outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reproduce the IRFA second-revision numerical analysis and figures."
    )
    parser.add_argument(
        "--outdir",
        default="irfa_revision_results",
        help="Output directory (default: irfa_revision_results)",
    )
    args = parser.parse_args()
    main(args.outdir)
