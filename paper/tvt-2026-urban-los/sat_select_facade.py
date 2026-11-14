#!/usr/bin/env python3
"""R4 (facade scope of the selection comparison). Map rules read LOS from geometry and do not depend on the wall; the measured rule ranks by ray-traced
path loss and does. Runs sat_select2.run_shell, with the random streams of sat_select2.main, on each facade of one user realization and writes
out/sat_select_facade_<set>.json: the rule summaries per facade, paired user-bootstrap intervals of measured minus marginal map (LOS share, 90th and 50th
percentile path loss), what the measured rule serves when it is not on a clear satellite, and, for the sparse shell, any-LOS shares from the sky maps
against independent draws from the scene's own elevation curve. The _c03 and _c03sd2 rows must reproduce sat_select2_<tag>.json exactly.
    python3 sat_select_facade.py first | second"""
import json, sys
import numpy as np
import sat_select as ss, sat_select2 as s2
from street_null2 import road_axes
OUT = ss.OUT; SETS = dict(first=("_c03", "_c03s", "", "_sc4"), second=("_c03sd2",)); B_LOS = 5000; B_PL = 1000; PAIR = ("HE", "sig10", "marg3", "marg5", "marg10", "marg20", "RSRP6h6", "RSRP3h3", "ORA")


def paired(a, b, rng):
    """Measured a, map b, per-user dicts from run_shell. Percentile intervals over users."""
    n = len(a["los"]); d = a["los"] - b["los"]; ii = rng.integers(0, n, (B_LOS, n)); bl = d[ii].mean(1)
    out = dict(los=float(d.mean()), los_ci=np.percentile(bl, [2.5, 97.5]).tolist())
    for nm, q in (("p90", 90), ("p50", 50)):
        v = float(np.percentile(a["tot"], q) - np.percentile(b["tot"], q)); bd = []
        for i in rng.integers(0, n, (B_PL, n)):
            bd.append(np.percentile(a["tot"][i], q) - np.percentile(b["tot"][i], q))
        out[nm] = v; out[nm + "_ci"] = np.percentile(bd, [2.5, 97.5]).tolist()
    return out


def own_curve_any(c, lat, tag, planes=24, per=11):
    """Any satellite in LOS, sparse shell: sky maps against independent draws from this scene's own LOS-against-elevation curve."""
    Sat, T = ss.shell(planes, per); los = np.load(f"{OUT}/sky2_{c}{tag}.npz")["los"]; nu = los.shape[0]; el, az, _ = ss.topo(Sat, T, lat); vis = el >= 15
    ie = np.clip(np.rint((el - 15) / 5).astype(int), 0, 14); ia = np.rint(az / 5).astype(int) % 72; cur = los.mean((0, 2)); rn = np.random.default_rng(3); a, d = [], []
    for t in range(len(T)):
        v = np.flatnonzero(vis[:, t])
        if len(v) == 0:
            a.append(np.zeros(nu)); d.append(np.zeros(nu)); continue
        E, A = ie[v, t], ia[v, t]; a.append(los[:, E, A].any(1)); d.append((rn.random((nu, len(v))) < cur[E]).any(1))
    return dict(maps=float(np.mean(a)), own_curve=float(np.mean(d)))


def served(keep, k, tag, c, lat):
    """Excess loss of what rule k serves when its satellite is blocked; the mean is over user-time samples."""
    q = keep[k]
    fsv = np.empty_like(q["tot"])
    for j in range(q["tot"].shape[1]):
        fsv[:, j] = q["fs"][q["id"][:, j], j]
    x = q["tot"] - fsv; b = x > 1e-9
    return dict(nlos_share=float(b.mean()), nlos_med=float(np.median(x[b])) if b.any() else None, under6=float((x[b] < 6).mean()) if b.any() else None)


def main(st):
    res = {}; fn = f"{OUT}/sat_select_facade_{st}.json"; rb = np.random.default_rng(77)
    for c, lat in ss.CITIES.items():
        tag0 = SETS[st][0]; F = s2.field(c); U = np.load(f"{OUT}/sky2_{c}{tag0}.npz")["users"]; rng = np.random.default_rng(42); REP = {s: [] for s in s2.SIG}
        EP = {f"sig{s}": s2.eps_at(F, U, s, rng, rep=REP[s]) for s in s2.SIG}; EP["sig0_hx07"] = s2.eps_at(F, U, 0, rng, 0.7); EP["sig0_hx13"] = s2.eps_at(F, U, 0, rng, 1.3)
        road = dict(tab=s2.road_table(c), ax={f"AZroad{s}": road_axes(c, s2.pos_err(F, U, s, rng))[0] for s in (0, 10, 20)})
        rp = np.random.default_rng(4242); MP = {f"marg{s}": s2.posterior(F, REP[s], s, rp) for s in s2.SIG if s > 0}; MP["marg10as5"] = s2.posterior(F, REP[10], 5, rp); MP["marg10as20"] = s2.posterior(F, REP[10], 20, rp)
        res[c] = dict(any=own_curve_any(c, lat, tag0))
        for tag in SETS[st]:
            keep = {}; d = s2.run_shell(c, lat, 72, 22, tag, EP, road, MP, keep=keep); sp = s2.run_shell(c, lat, 24, 11, tag, EP, road, MP)
            P = {f"marg{s}": paired(keep["RSRP6h6"], keep[f"marg{s}"], rb) for s in (3, 5, 10, 20)}
            P3 = {f"marg{s}": paired(keep["RSRP3h3"], keep[f"marg{s}"], rb) for s in (10,)}
            res[c][tag or "marble"] = dict(dense={k: d[k] for k in d}, sparse=sp, paired=P, paired3=P3, served={k: served(keep, k, tag, c, lat) for k in ("RSRP6h6", "marg10", "ORA")})
            print(c, tag or "marble", {k: (round(100 * v["los"], 1), round(v["p90"], 1)) for k, v in d.items() if k in PAIR}, P["marg10"], flush=True)
            del keep
            json.dump(res, open(fn, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
