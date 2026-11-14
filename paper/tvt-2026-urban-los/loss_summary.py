#!/usr/bin/env python3
"""Loss of blocked links against the TR 38.811 clutter loss, by elevation band and facade setting, 4 cities x 300 users,
and two external references: Schreiber et al. at 1.5 m and the P.681-11 Annex 2 measured 2.2 GHz urban bad state. Writes out/loss_summary.json."""
import json, os
import numpy as np
from analysis2 import TR_EL, TR_CL_S, TR_SF_S, OUT, CITIES
from analysis3 import schreiber_cl
BANDS = ((15, 30), (30, 60), (60, 86)); UNR = 150.0
SET = {"marble": "", "marble_rough": "_sc4", "concrete": "_c03", "concrete_rough": "_c03s", "concrete_seed2": "_c03sd2"}
P681 = {20: (-18.1771, 1.1411, 4.0581), 30: (-17.4276, .9175, -.8009), 45: (-15.4844, .9434, -1.7555), 60: (-14.1435, .6975, -7.5383), 70: (-12.9383, 2.5318, 16.8468)}


def p681_total():
    e = np.array(list(P681)); v = []
    for ma, h1, h2 in P681.values():
        v.append(-10 * np.log10(10 ** (ma / 10) + 10 ** ((h1 * ma + h2) / 10)))
    return e, np.array(v), np.array([-x[0] for x in P681.values()])


def main():
    res = {}; rng = np.random.default_rng(4); pe, pv, pdir = p681_total()
    for nm, tag in SET.items():
        if not all(os.path.exists(f"{OUT}/sky2_{c}{tag}.npz") and np.load(f"{OUT}/sky2_{c}{tag}.npz")["los"].shape[0] == 300 for c in CITIES):
            continue
        res[nm] = {}
        for c in CITIES:
            z = np.load(f"{OUT}/sky2_{c}{tag}.npz"); L = -z["exc"]; los = z["los"]; els = z["els"]
            T = z["d_edge"][:, None, :] * np.tan(np.radians(els))[None, :, None] >= float(z["H95"]); n = L.shape[0]
            E = np.broadcast_to(els[None, :, None], L.shape); cl = np.interp(E, TR_EL, TR_CL_S); sd = np.interp(E, TR_EL, TR_SF_S["dense_urban"])
            X = np.where(L < UNR, L, np.inf); M = (~los) & T; r = dict(bands=[])
            for lo, hi in BANDS:
                k = (E >= lo) & (E < hi) & M; x = X[k]; tr = cl[k]
                # cluster bootstrap by users
                U = np.broadcast_to(np.arange(n)[:, None, None], L.shape)[k]; idx = {u: np.flatnonzero(U == u) for u in range(n)}; bm = []
                for _ in range(200):
                    sel = np.concatenate([idx[u] for u in rng.integers(0, n, n)]); bm.append(np.median(x[sel]))
                ref = float(np.median(tr))
                r["bands"].append(dict(cens=float(np.median(x)), reached=float(np.median(x[np.isfinite(x)])), unreached=float(np.mean(~np.isfinite(x))), tr=ref, margin=ref - float(np.median(x)), margin_ci=[ref - float(np.percentile(bm, 97.5)), ref - float(np.percentile(bm, 2.5))],
                                     below_tr=float(np.mean(x < tr)), cover=float(np.mean((x >= tr - 1.2816 * sd[k]) & (x <= tr + 1.2816 * sd[k]))), below_band=float(np.mean(x < tr - 1.2816 * sd[k])), cover_urban=float(np.mean((x >= tr - 1.2816 * 6.0) & (x <= tr + 1.2816 * 6.0)))))
            res[nm][c] = r
    # external references averaged over the elevation grid of each band
    res["refs"] = {"tr": [], "schreiber": [], "p681": [], "p681_direct": []}
    for lo, hi in BANDS:
        g = np.arange(lo, hi, 5.0); gs = np.clip(g, 20, 80)
        res["refs"]["tr"].append(float(np.mean(np.interp(g, TR_EL, TR_CL_S))))
        res["refs"]["schreiber"].append(float(np.mean(np.interp(g, np.arange(10, 81, 10.0), [schreiber_cl(int(t)) for t in range(10, 81, 10)]))))
        res["refs"]["p681"].append(float(np.mean(np.interp(g, pe, pv)))); res["refs"]["p681_direct"].append(float(np.mean(np.interp(g, pe, pdir))))
    json.dump(res, open(f"{OUT}/loss_summary.json", "w"), indent=1)
    for nm in SET:
        if nm in res:
            for c in CITIES:
                print(nm, c, [(round(b["cens"], 1), round(b["reached"], 1), round(b["margin"], 1)) for b in res[nm][c]["bands"]], "below band", [round(b["below_band"], 2) for b in res[nm][c]["bands"]], "cover", [round(b["cover"], 2) for b in res[nm][c]["bands"]])
    print(res["refs"])


if __name__ == "__main__":
    main()
