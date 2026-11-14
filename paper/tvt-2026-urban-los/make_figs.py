#!/usr/bin/env python3
"""Data figures of the TVT manuscript on ray-traced urban visibility."""
import json, os, sys
import numpy as np
from figstyle import *  # noqa: F401,F403
import matplotlib.pyplot as plt
from p9_analysis import nearest, TR_LOS, TR_CL_S, TR_SF_NLOS_S, p1410

setup()
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out"); FIG = os.path.join(HERE, "..", "figures")
CITY = {"munich": ("Munich", BLUE, "o"), "etoile": ("Paris", ORANGE, "s"), "florence": ("Florence", GREEN, "^"),
        "san_francisco": ("San Francisco", PURPLE, "D")}
A = json.load(open(os.path.join(OUT, "analysis.json"))); F = json.load(open(os.path.join(OUT, "models.json")))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name)); plt.close(fig); print("wrote", name)


def fig_los():
    fig, axs = plt.subplots(1, 4, figsize=(PAGE, 2.2), sharey=True)
    E = np.arange(10, 91, 1.0)
    for ax, (c, (lab, col, mk)) in zip(axs, CITY.items()):
        els = np.array(A[c]["els"])
        ax.plot(els, 100 * np.array(A[c]["los"]), ls="none", marker=mk, color=col, mfc="white", ms=3.5, label="Ray traced")
        ax.plot(els, 100 * np.array(F[c]["stat"]), color=col, lw=1.1, label="First-facade model")
        ax.plot(els, 100 * np.array(F[c]["canyon"]), color=col, lw=0.8, ls=(0, (5, 1.5, 1, 1.5)), alpha=0.8, label="Idealized canyon")
        ax.step(E, 100 * nearest(E, TR_LOS["urban"]), where="mid", color=GREY, lw=0.8, ls=(0, (4, 1.6)), label="TR 38.811 urban")
        ax.step(E, 100 * nearest(E, TR_LOS["dense_urban"]), where="mid", color=INK, lw=0.8, ls=(0, (1, 1.2)), label="TR 38.811 dense urban")
        ax.set_title(lab, fontsize=8); ax.set_xlabel("Elevation (deg)"); ax.set_xlim(10, 90); ax.set_ylim(0, 100)
    axs[0].set_ylabel("LOS probability (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.1), fontsize=6.5)
    fig.tight_layout(w_pad=0.5); save(fig, "fig2_los.pdf")


def fig_azimuth():
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    bins = np.array([0, 15, 30, 45, 60, 75, 90.01])
    for c, (lab, col, mk) in CITY.items():
        z = np.load(os.path.join(OUT, f"sky_{c}.npz")); s = np.load(os.path.join(OUT, f"street_{c}.npz"))
        los, els, azs = z["los"], z["els"], z["azs"]
        rel = np.abs(((azs[None, :] - s["axis_deg"][:, None] + 90) % 180) - 90)
        i = list(els).index(30.0)
        y = [100 * los[:, i, :][(rel >= lo) & (rel < hi)].mean() for lo, hi in zip(bins[:-1], bins[1:])]
        ax.plot((bins[:-1] + bins[1:]) / 2, y, marker=mk, color=col, label=lab)
    ax.axhline(100 * nearest([30.0], TR_LOS["urban"])[0], color=GREY, lw=0.8, ls=(0, (4, 1.6)), label="TR 38.811 urban")
    ax.set_xlabel("Satellite azimuth from the street axis (deg)"); ax.set_ylabel("LOS probability at 30 deg (%)")
    ax.legend(fontsize=6, ncol=2); fig.tight_layout(); save(fig, "fig3_azimuth.pdf")


def fig_loss():
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    E = np.arange(10, 91, 1.0)
    for c, (lab, col, mk) in CITY.items():
        els = np.array(A[c]["els"]); med = np.array([np.nan if x is None else x for x in A[c]["nlos_med"]])
        lo = np.array([np.nan if x is None else x for x in A[c]["nlos_p10"]]); hi = np.array([np.nan if x is None else x for x in A[c]["nlos_p90"]])
        ax.plot(els, med, marker=mk, color=col, ms=3, label=lab)
        ax.fill_between(els, lo, hi, color=col, alpha=0.08, lw=0)
    ax.step(E, nearest(E, TR_CL_S["dense_urban"]), where="mid", color=INK, lw=1.0, label="TR 38.811 clutter loss")
    ax.set_xlabel("Elevation (deg)"); ax.set_ylabel("Excess loss when blocked (dB)"); ax.set_xlim(15, 85)
    ax.legend(fontsize=6, ncol=2, loc="upper left"); fig.tight_layout(); save(fig, "fig4_loss.pdf")


def fig_deficit():
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    for c, (lab, col, mk) in CITY.items():
        lo = F["loss_loo"][c]
        ax.plot(lo["x"], lo["y"], ls="none", marker=mk, color=col, mfc="white", ms=4, label=lab)
    x = np.logspace(-0.3, 2, 100); a, b = F["loss_fit_all"]["a"], F["loss_fit_all"]["b"]
    ax.plot(x, a + b * np.log10(1 + x), color=INK, lw=1.0, label=r"$\kappa_0+\kappa_1\log_{10}(1+\Delta h)$")
    ax.set_xscale("log"); ax.set_xlabel(r"Ray deficit below the facade edge $\Delta h$ (m)"); ax.set_ylabel("Median excess loss (dB)")
    ax.legend(fontsize=6, ncol=2); fig.tight_layout(); save(fig, "fig5_deficit.pdf")


if __name__ == "__main__":
    for w in (sys.argv[1:] or ["los", "azimuth", "loss", "deficit"]):
        globals()[f"fig_{w}"]()
