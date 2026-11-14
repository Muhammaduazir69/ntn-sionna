#!/usr/bin/env python3
"""Round-4 LOS analysis on the stored maps (no field solve): what each level of information about the city explains.
Writes out/analysis3.json. Levels, per direction of each user, Brier score over reference elevations 20 to 80 deg:
 elevation only: TR 38.811 urban / dense, Schreiber et al. at 1.5 m, the pooled curve of the other three cities;
 street axis: P(LOS | e, azimuth from axis), table from the other three cities, with own and cross-fitted axis;
 footprints: F_H(h + t(a) tan e) with the user's own t(a) and F_H from the other three cities;
 P.681 (true modal height) with the user's own wall distance; capped P.681;
 first-facade rule (own t, H) and skyline rule (own map).
Also: curve errors of every model, along/across at 30 deg, half-plane and cross-city transfer at all elevations."""
import json, os
import numpy as np
from analysis2 import load, los_curve, wmean, at_ref, stat_curve, REF, TR_EL, TR_LOS, H_U, OUT, CITIES, itu_stats_los, STATS
K = {"S": None}
TAB2_A = {10: -0.811, 20: -0.679, 30: -0.571, 40: -0.476, 50: -0.390, 60: -0.308, 70: -0.235, 80: -0.167, 90: 0.0}
TAB2_X = {10: 10.553, 20: 8.847, 30: 8.054, 40: 7.745, 50: 7.701, 60: 7.890, 70: 8.422, 80: 9.305, 90: 0.0}
TAB2_W = {10: 2.871, 20: 3.218, 30: 3.394, 40: 3.505, 50: 3.581, 60: 3.580, 70: 3.538, 80: 3.450, 90: 1.0}
CL_A = {10: 28.014, 20: 22.069, 30: 18.675, 40: 16.076, 50: 14.046, 60: 13.382, 70: 12.735, 80: 10.190}
CL_B = {10: 0.616, 20: 0.516, 30: 0.480, 40: 0.380, 50: 0.412, 60: 0.379, 70: 0.247, 80: 0.117}
CL_C = {10: 9.151, 20: 6.341, 30: 4.744, 40: 3.464, 50: 1.000, 60: 1.824, 70: 4.338, 80: 4.716}
CL_W = {10: -9.78e-1, 20: -4.94e-1, 30: -4.24e-1, 40: -1.88e-1, 50: -1.69e-1, 60: -1.28e-6, 70: -1.36e-6, 80: 3.15e-6}


def schreiber_los(th, h=1.5):
    """Eqs. (7)-(8) and Table II of Schreiber et al., S band, below rooftop; B = 1."""
    return float(TAB2_A[th] * (1 - 1 / (1 + np.exp(-(h - TAB2_X[th]) / TAB2_W[th]))) + 1.0)


def schreiber_cl(th, h=1.5, hrt=22.5):
    """Eq. (16) and Table VIII, S band, NLOS, below rooftop, Delta h = h - hRT."""
    return float(CL_A[th] - CL_B[th] * h - CL_C[th] * np.exp(-CL_W[th] * (h - hrt)))


def true_mode(hs):
    u, c = np.unique(np.round(hs), return_counts=True)
    return float(u[np.argmax(c)])


def p681(dm, e, phi, hb):
    s = np.maximum(np.abs(np.sin(np.radians(phi))), np.sin(np.radians(2.5)))
    return 1 - np.exp(-(H_U + dm * np.tan(np.radians(e)) / s) ** 2 / (2 * hb ** 2))


def p681_cap(dm, e, phi, hb, Lc):
    s = np.maximum(np.abs(np.sin(np.radians(phi))), np.sin(np.radians(2.5)))
    return 1 - np.exp(-(H_U + np.minimum(dm / s, Lc) * np.tan(np.radians(e))) ** 2 / (2 * hb ** 2))


def rel_az(azs, ax):
    return np.abs(((azs[None, :] - ax[:, None] + 90) % 180) - 90)


def wall_dist(D, ax):
    n = len(ax); a = np.nan_to_num(ax); kn = ((a + 90) / 5).round().astype(int) % 72; kn2 = (kn + 36) % 72
    relr = D["azs"][None, :] - a[:, None]
    dm = np.where(np.sin(np.radians(relr)) > 0, D["t"][np.arange(n), kn][:, None], D["t"][np.arange(n), kn2][:, None])
    ka = (a / 5).round().astype(int) % 72; ka2 = (ka + 36) % 72
    al = np.r_[D["t"][np.arange(n), ka], D["t"][np.arange(n), ka2]]
    return np.minimum(dm, 500.0), relr, float(np.median(al[al < 500]))


def F_H(hs, z):
    return np.searchsorted(hs, z, side="left") / len(hs)


def ff_cond(D, sel, ax, e, lo, hi, hs=None):
    R = rel_az(D["azs"], np.nan_to_num(ax)); m = (R >= lo) & (R <= hi) & sel[:, None]
    tt = D["t"][m]; h_ = np.sort(D["H"][(D["t"] < 500) & sel[:, None]]) if hs is None else hs
    op = tt >= 500; z = H_U + tt[~op] * np.tan(np.radians(e))
    return float(op.mean() + (1 - op.mean()) * np.mean(F_H(h_, z)))


def tw(D, sel, ax, i, lo, hi):
    R = rel_az(D["azs"], np.nan_to_num(ax)); m = D["trust"][:, i, :] & (R >= lo) & (R <= hi) & sel[:, None]
    return wmean(D["los"][:, i, :], m)


def main():
    Ds = {c: load(c) for c in CITIES}
    for c in Ds:
        Ds[c]["cf"] = np.load(f"{OUT}/axis_cf_{c}.npz")["cf"]; Ds[c]["bh"] = np.load(f"{OUT}/sky2_{c}.npz")["built_h"]
    els = Ds["munich"]["els"]; ri = [list(els).index(e) for e in REF]
    res = {}
    rng = np.random.default_rng(21)
    for c, D in Ds.items():
        n = D["los"].shape[0]; T = D["trust"]; L = D["los"].astype(float); r = {}
        oth = [x for x in Ds if x != c]
        true = los_curve(D); tref = at_ref(els, true)
        # pooled curve and tables from the other three cities
        pc = np.mean([los_curve(Ds[o]) for o in oth], 0)
        tabs = {}
        for key, name in (("ax2", "own"), ("cf", "cf")):
            num = np.zeros((15, 19)); den = np.zeros((15, 19))
            for o in oth:
                Do = Ds[o]; ok = np.isfinite(Do[key]); k = np.rint(rel_az(Do["azs"], Do[key][ok]) / 5).astype(int)
                for i in range(15):
                    w = Do["trust"][ok][:, i, :]; np.add.at(num[i], k[w], Do["los"][ok][:, i, :][w].astype(float)); np.add.at(den[i], k[w], 1.0)
            tabs[name] = num / np.maximum(den, 1)
        hs_o = np.sort(np.concatenate([Ds[o]["H"][Ds[o]["t"] < 500] for o in oth]))
        hb = true_mode(D["bh"]); r["hb_mode"] = hb; r["hb_rayleigh"] = float(np.sqrt(np.mean(D["H"][D["t"] < 500] ** 2) / 2))
        dm_own, relr_own, Lc_own = wall_dist(D, D["ax2"]); dm_cf, relr_cf, Lc_cf = wall_dist(D, D["cf"])
        # model curves: errors over reference elevations
        sch = np.array([schreiber_los(int(e)) for e in REF]); ptr = {k: np.interp(REF, TR_EL, v) for k, v in TR_LOS.items()}
        pcurve = np.array([wmean(p681(dm_own, e, relr_own, hb), T[:, list(els).index(e), :]) for e in REF])
        pcap = np.array([wmean(p681_cap(dm_own, e, relr_own, hb, Lc_own), T[:, list(els).index(e), :]) for e in REF])
        st_in = at_ref(els, stat_curve(D, els)); st_oth = {o: at_ref(els, stat_curve(Ds[o], els)) for o in oth}
        err = lambda v: dict(mean=float(np.mean(np.abs(v - tref))), max=float(np.max(np.abs(v - tref))), bias=float(np.mean(v - tref)))
        r["curves"] = dict(tr_urban=err(ptr["urban"]), tr_dense=err(ptr["dense_urban"]), schreiber=err(sch), p681=err(pcurve), p681_cap=err(pcap), stat_in=err(st_in),
                           stat_cross={o: err(v) for o, v in st_oth.items()}, pooled=err(at_ref(els, pc)))
        # half-plane transfer of eq. (plos) and the azimuth-conditioned form, both splits, every reference elevation
        xu, yu = D["users"][:, 0], D["users"][:, 1]; rows = []; rows_cf = []
        for s_ in (xu, yu):
            w_ = s_ < np.median(s_)
            for a_, b_ in ((w_, ~w_), (~w_, w_)):
                ee, ec = [], []
                for e in REF:
                    i = list(els).index(e)
                    ee.append(max(abs(ff_cond(D, a_, D["ax2"], e, 0, 15) - tw(D, b_, D["ax2"], i, 0, 15)), abs(ff_cond(D, a_, D["ax2"], e, 75, 90) - tw(D, b_, D["ax2"], i, 75, 90))))
                    ec.append(max(abs(ff_cond(D, a_, D["cf"], e, 0, 15) - tw(D, b_, D["cf"], i, 0, 15)), abs(ff_cond(D, a_, D["cf"], e, 75, 90) - tw(D, b_, D["cf"], i, 75, 90))))
                rows.append(ee); rows_cf.append(ec)
        rows = np.array(rows); rows_cf = np.array(rows_cf); i30r = list(REF).index(30.0)
        r["split"] = dict(e30_mean_max=float(rows[:, i30r].mean()), e30_worst=float(rows[:, i30r].max()), all_mean=float(rows.max(1).mean()), all_worst=float(rows.max()),
                          cf_e30_mean=float(rows_cf[:, i30r].mean()), cf_e30_worst=float(rows_cf[:, i30r].max()))
        # along and across at 30 deg
        i30 = list(els).index(30.0); allu = np.ones(n, bool); P30own = p681(dm_own, 30.0, relr_own, hb); P30cf = p681(dm_cf, 30.0, relr_cf, hb); C30 = p681_cap(dm_own, 30.0, relr_own, hb, Lc_own)
        R_o, R_c = rel_az(D["azs"], D["ax2"]), rel_az(D["azs"], np.nan_to_num(D["cf"])); T30 = T[:, i30, :]
        for nm, ax, R_, P_ in (("own", D["ax2"], R_o, P30own), ("cf", D["cf"], R_c, P30cf)):
            al, ac = T30 & (R_ <= 15), T30 & (R_ >= 75)
            r[f"az_{nm}"] = dict(along=wmean(L[:, i30, :], al), across=wmean(L[:, i30, :], ac), p681_along=wmean(P_, al), p681_across=wmean(P_, ac),
                                 ff_along=ff_cond(D, allu, ax, 30.0, 0, 15), ff_across=ff_cond(D, allu, ax, 30.0, 75, 90))
        r["az_own"]["cap_along"], r["az_own"]["cap_across"] = wmean(C30, T30 & (R_o <= 15)), wmean(C30, T30 & (R_o >= 75))
        # Brier scores by information level, pairs at reference elevations
        ix = np.zeros((n, 15, 72), bool); ix[:, ri, :] = True; M = T & ix
        y = L[M]; E_ = np.broadcast_to(els[None, :, None], L.shape)[M]; U_ = np.broadcast_to(np.arange(n)[:, None, None], L.shape)[M]
        def rep(a):
            return np.broadcast_to(a, L.shape)[M]
        preds = {}
        preds["tr_best"] = None
        pu, pd = np.interp(E_, TR_EL, TR_LOS["urban"]), np.interp(E_, TR_EL, TR_LOS["dense_urban"])
        preds["tr_urban"], preds["tr_dense"] = pu, pd
        preds["schreiber"] = np.array([{e_: schreiber_los(int(e_)) for e_ in REF}[e_] for e_ in E_])
        preds["own_curve"] = np.interp(E_, els, true)                # oracle: the city's own curve
        preds["other_curve"] = np.interp(E_, els, pc)
        for nm, tb, R_ in (("axis_own", tabs["own"], R_o), ("axis_cf", tabs["cf"], R_c)):
            k = np.rint(R_ / 5).astype(int); pp = np.stack([tb[i][k] for i in range(15)], 1); preds[nm] = pp[M]
        zt = H_U + D["t"][:, None, :] * np.tan(np.radians(els))[None, :, None]
        ffoot = np.where(D["t"][:, None, :] >= 500, 1.0, F_H(hs_o, zt)); preds["footprint"] = ffoot[M]
        p6 = np.stack([p681(dm_own, e, relr_own, hb) for e in els], 1); preds["p681"] = p6[M]
        rule1 = np.where(D["t"][:, None, :] >= 500, True, zt > D["H"][:, None, :]).astype(float); preds["rule1"] = rule1[M]
        eps1 = np.where(D["t"] < 500, np.degrees(np.arctan2(D["H"] - H_U, D["t"])), -90.0); eps = np.maximum(eps1, D["sky"])
        rule2 = np.stack([eps < e for e in els], 1).astype(float); preds["rule2"] = rule2[M]
        del preds["tr_best"]
        bs = {k: float(np.mean((y - v) ** 2)) for k, v in preds.items()}; bs["tr_best"] = min(bs["tr_urban"], bs["tr_dense"]); r["brier"] = bs
        tr_key = "tr_urban" if bs["tr_urban"] < bs["tr_dense"] else "tr_dense"
        idx = {u: np.flatnonzero(U_ == u) for u in range(n)}; diffs = {k: [] for k in ("axis_own", "axis_cf", "footprint", "rule2")}
        for _ in range(200):
            sel = np.concatenate([idx[u] for u in rng.integers(0, n, n)])
            base = np.mean((y[sel] - preds["other_curve"][sel]) ** 2)
            for k in diffs:
                diffs[k].append(base - np.mean((y[sel] - preds[k][sel]) ** 2))
        r["brier_gain_ci"] = {k: np.percentile(v, [2.5, 97.5]).tolist() for k, v in diffs.items()}
        # street label share and geometry
        ag = np.abs(((D["ax1"] - D["ax2"] + 90) % 180) - 90) <= 15; r["agree_share"] = float(ag.mean())
        for nm, s_ in (("agree", ag), ("non", ~ag)):
            r[f"swing_{nm}"] = tw(D, s_, D["ax2"], i30, 0, 15) - tw(D, s_, D["ax2"], i30, 75, 90)
        w = D["t"][:, :36] + D["t"][:, 36:]; r["narrow_med"] = float(np.median(w.min(1))); r["H_med"] = float(np.median(D["H"][D["t"] < 500])); r["aspect"] = r["H_med"] / r["narrow_med"]
        from scipy.spatial import cKDTree
        r["nn_med"] = float(np.median(cKDTree(D["users"][:, :2]).query(D["users"][:, :2], k=2)[0][:, 1]))
        res[c] = r
        print(c, "brier", {k: round(v, 3) for k, v in bs.items()}, "\n  curves", {k: (round(v["mean"], 3), round(v["max"], 3)) for k, v in r["curves"].items() if "mean" in v}, "\n  az", r["az_own"], r["az_cf"], "\n  split", r["split"], "hb", hb, "agree", r["agree_share"], r["swing_agree"], r["swing_non"], "aspect", r["aspect"])
    json.dump(res, open(f"{OUT}/analysis3.json", "w"), indent=1)


if __name__ == "__main__":
    main()
