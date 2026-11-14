#!/usr/bin/env python3
"""Data figures of the TVT manuscript from the version-2 outputs (analysis2.json and the sky2/street2 maps)."""
import json, os, sys
import numpy as np
from figstyle import *  # noqa: F401,F403
import matplotlib.pyplot as plt
from analysis2 import load, TR_EL, TR_LOS, TR_CL_S, REF, rel_az
from analysis3 import schreiber_los

setup()
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out"); FIG = os.path.join(HERE, "..", "figures")
CITY = {"munich": ("Munich", BLUE, "o"), "etoile": ("Paris", ORANGE, "s"), "florence": ("Florence", GREEN, "^"),
        "san_francisco": ("San Francisco", PURPLE, "D")}
A = json.load(open(os.path.join(OUT, "analysis2.json")))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name)); plt.close(fig); print("wrote", name)


def fig_los():
    fig, axs = plt.subplots(1, 4, figsize=(PAGE, 2.25), sharey=True)
    for ax, (c, (lab, col, mk)) in zip(axs, CITY.items()):
        r = A[c]; els = np.array(r["els"])
        ax.fill_between(els, 100 * np.array(r["los_lo"]), 100 * np.array(r["los_hi"]), color=col, alpha=0.15, lw=0)
        ax.plot(els, 100 * np.array(r["los"]), ls="none", marker=mk, color=col, mfc="white", ms=3.5, label="Ray traced (95% CI)")
        ax.plot(els, 100 * np.array(r["stat_in"]), color=col, lw=1.1, label="First-facade model")
        ax.plot(els, 100 * np.array(r["canyon"]), color=col, lw=0.8, ls=(0, (5, 1.5, 1, 1.5)), label="Idealized canyon")
        ax.plot(REF, 100 * np.interp(REF, TR_EL, TR_LOS["urban"]), ls="none", marker="x", color=GREY, ms=4, label="TR 38.811 urban")
        ax.plot(REF, 100 * np.interp(REF, TR_EL, TR_LOS["dense_urban"]), ls="none", marker="+", color=INK, ms=5, label="TR 38.811 dense urban")
        ax.plot(REF, [100 * schreiber_los(int(e)) for e in REF], color=INK2, lw=0.7, ls=(0, (1, 1.2)), label="Schreiber et al., 1.5 m")
        ax.set_title(lab, fontsize=8); ax.set_xlabel("Elevation (deg)"); ax.set_xlim(12, 88); ax.set_ylim(0, 100)
    axs[0].set_ylabel("LOS probability (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.17), fontsize=6.5)
    fig.tight_layout(w_pad=0.5); save(fig, "fig2_los.pdf")


def fig_azimuth():
    """LOS at 30 deg against satellite azimuth from the axis, first realization: road and own axes against street-free controls, P.681."""
    from analysis3 import p681, wall_dist
    SN = json.load(open(os.path.join(OUT, "street_null2.json"))); A3 = json.load(open(os.path.join(OUT, "analysis3.json")))
    fig, axs = plt.subplots(1, 4, figsize=(PAGE, 2.25), sharey=True)
    edges = np.array([0, 15.1, 30, 45, 60, 74.9, 91.0]); mids = np.array([7.5, 22.5, 37.5, 52.5, 67.5, 82.5])
    for ax, (c, (lab, col, mk)) in zip(axs, CITY.items()):
        s = SN[c]; D = load(c); i = list(D["els"]).index(30.0); T = D["trust"][:, i, :]
        rot = np.array([x["curve"]["own"] for x in s["ctl"]["rotm"]]) * 100; iso = np.array([x["curve"]["own"] for x in s["ctl"]["iso"]]) * 100; rotr = np.array([x["curve"]["road"] for x in s["ctl"]["rotm"]]) * 100
        ax.fill_between(mids, rot.mean(0) - rot.std(0), rot.mean(0) + rot.std(0), color=GREY, alpha=0.25, lw=0)
        ax.plot(mids, 100 * np.array(s["seed1"]["curve"]["road"]), ls="none", marker=mk, color=col, ms=4, label="Scene, road axis")
        ax.plot(mids, rotr.mean(0), color=GREY, lw=0.9, ls=(0, (1, 1.2)), label="Rotated blocks, road axis")
        ax.plot(mids, 100 * np.array(s["seed1"]["curve"]["own"]), ls="none", marker=mk, color=col, mfc="white", ms=4, label="Scene, own axis")
        ax.plot(mids, rot.mean(0), color=GREY, lw=1.1, label="Rotated blocks, own axis")
        ax.plot(mids, iso.mean(0), color=INK, lw=0.9, ls=(0, (4, 1.6)), label="Random rectangles, own axis")
        dm, relr, _ = wall_dist(D, D["ax2"]); P = p681(dm, 30.0, relr, A3[c]["hb_mode"]); R = np.abs(((D["azs"][None, :] - D["ax2"][:, None] + 90) % 180) - 90)
        ax.plot(mids, [100 * P[(R >= a) & (R < b) & T].mean() for a, b in zip(edges[:-1], edges[1:])], color=INK2, lw=0.6, ls=(0, (5, 1.5, 1, 1.5)), label="ITU-R P.681 roadside")
        ax.set_title(lab, fontsize=8); ax.set_xlabel("Azimuth from axis (deg)"); ax.set_ylim(0, 100); ax.set_xticks([0, 30, 60, 90])
    axs[0].set_ylabel("LOS probability at 30 deg (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.17), fontsize=6.5)
    fig.tight_layout(w_pad=0.5); save(fig, "fig3_azimuth.pdf")


def fig_loss():
    from analysis3 import schreiber_cl
    from loss_summary import p681_total
    Ac = json.load(open(os.path.join(OUT, "analysis2_c03.json")))
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    for c, (lab, col, mk) in CITY.items():
        r = Ac[c]; els = np.array(r["els"])
        cen = np.array([np.nan if x is None else x for x in r["nlos_med_cens"]], float)
        p10 = np.array([np.nan if x is None else x for x in r["nlos_p10"]], float)
        p90 = np.array([np.nan if x is None else x for x in r["nlos_p90"]], float)
        ax.plot(els, np.where(np.isfinite(cen), cen, np.nan), marker=mk, color=col, ms=3, lw=0.8, label=lab)
        ax.fill_between(els, p10, np.where(np.isfinite(p90), p90, 80), color=col, alpha=0.05, lw=0)
    E = np.arange(10, 91, 1.0)
    ax.plot(E, np.interp(E, TR_EL, TR_CL_S), color=INK, lw=1.2, label="TR 38.811 clutter loss")
    T = np.arange(10, 81, 10); ax.plot(T, [schreiber_cl(int(t)) for t in T], color=INK2, lw=0.9, ls=(0, (4, 1.6)), label="Schreiber et al., 1.5 m")
    pe, pv, _ = p681_total(); ax.plot(pe, pv, color=RED, lw=0.9, marker="x", ms=3.5, label="P.681 measured, bad state")
    ax.set_xlabel("Elevation (deg)"); ax.set_ylabel("Excess loss when blocked (dB)"); ax.set_xlim(15, 85); ax.set_ylim(0, 70)
    ax.legend(fontsize=5.8, ncol=2, loc="upper left"); fig.tight_layout(); save(fig, "fig4_loss.pdf")


if __name__ == "__main__":
    for w in (sys.argv[1:] or ["los", "azimuth", "loss"]):
        globals()[f"fig_{w}"]()
