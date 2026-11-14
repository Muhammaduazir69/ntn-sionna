#!/usr/bin/env python3
"""R3: held-out Brier ladder. Tables by (elevation, features) are learned on the other three cities and tested on the held-out city's users; only the laws are
held out, since the features of a test user come from that city's own map. Also the confirmatory second realization. Reference elevations 20-80 deg. Writes out/brier_ladder.json."""
import json
import numpy as np
from analysis2 import CITIES, OUT, REF, TR_EL, TR_LOS, H_U
from analysis3 import wall_dist, rel_az, schreiber_los
DB = np.array([0, 3, 5, 7, 10, 14, 20, 30, 50, 100, 1e9])


def load(c, tag):
    z = np.load(f"{OUT}/sky2_{c}{tag}.npz"); s = np.load(f"{OUT}/street2_{c}{tag}.npz"); ax = np.load(f"{OUT}/axes2_{c}{tag}.npz")
    T = z["d_edge"][:, None, :] * np.tan(np.radians(z["els"]))[None, :, None] >= float(z["H95"])
    t_, H_ = s["free"], s["hfac"]; e1 = np.where(t_ < 500, np.degrees(np.arctan2(H_ - H_U, t_)), -90.0); eps = np.maximum(e1, s["sky"])
    return dict(los=z["los"], trust=T, azs=z["azs"], els=z["els"], t=t_, H=H_, eps=eps, ax2=s["axis2_deg"], cf=np.nan_to_num(ax["cf"]), road=ax["road"], st=ax["st"])


def feats(D, key):
    ax = D[key]; dm, _, _ = wall_dist(D, ax); return np.rint(rel_az(D["azs"], ax) / 5).astype(int), np.digitize(dm, DB) - 1


def tb(D):
    return np.digitize(np.minimum(D["t"], 1e8), DB) - 1


def run(train, test, key, ri):
    ks_tr = [key(D) for D in train]; ks_te = key(test); n = len(ks_te); dims = [max([k[i].max() for k in ks_tr] + [ks_te[i].max()]) + 1 for i in range(n)]
    num = np.zeros((15,) + tuple(dims)); den = np.zeros_like(num)
    for D, k in zip(train, ks_tr):
        for i in ri:
            idx = (i * np.ones_like(k[0]),) + tuple(k); np.add.at(num, idx, D["los"][:, i, :].astype(float)); np.add.at(den, idx, 1.0)
    em = num.reshape(15, -1).sum(1) / np.maximum(den.reshape(15, -1).sum(1), 1); nu = test["los"].shape[0]; err = np.zeros((nu, len(ri), 72))
    for j, i in enumerate(ri):
        idx = (i * np.ones_like(ks_te[0]),) + tuple(ks_te); p = np.where(den[idx] >= 20, num[idx] / np.maximum(den[idx], 1), em[i]); err[:, j, :] = (test["los"][:, i, :].astype(float) - p) ** 2
    return err.mean(axis=(1, 2))        # per user


def main():
    res = {}; S1 = {c: load(c, "") for c in CITIES}; ri = [list(S1["munich"]["els"]).index(e) for e in REF]; rng = np.random.default_rng(31)
    LV = dict(elev=lambda D: [np.zeros_like(D["t"], int)], axis_own=lambda D: [feats(D, "ax2")[0]], axis_cf=lambda D: [feats(D, "cf")[0]], axis_road=lambda D: [feats(D, "road")[0]], axis_st=lambda D: [feats(D, "st")[0]],
              side_own=lambda D: [feats(D, "ax2")[1]], own_side=lambda D: list(feats(D, "ax2")), road_side=lambda D: list(feats(D, "road")), cf_side=lambda D: list(feats(D, "cf")), st_side=lambda D: list(feats(D, "st")),
              t=lambda D: [tb(D)], t_own=lambda D: [tb(D), feats(D, "ax2")[0]], t_road=lambda D: [tb(D), feats(D, "road")[0]])
    for tag, name in (("", "seed1"), ("_c03sd2", "seed2")):
        TE = {c: (S1[c] if not tag else load(c, tag)) for c in CITIES}; res[name] = {}
        for c in CITIES:
            tr = [S1[o] for o in CITIES if o != c]; te = TE[c]; u = {k: run(tr, te, f, ri) for k, f in LV.items()}; nu = te["los"].shape[0]
            E_ = te["els"][ri]; ptr = {k: np.interp(E_, TR_EL, v) for k, v in TR_LOS.items()}
            u["tr_best"] = min((((te["los"][:, ri, :].astype(float) - p[None, :, None]) ** 2).mean(axis=(1, 2)) for p in ptr.values()), key=lambda a: a.mean())
            sch = np.array([schreiber_los(int(e)) for e in E_]); u["schreiber"] = ((te["los"][:, ri, :].astype(float) - sch[None, :, None]) ** 2).mean(axis=(1, 2))
            u["skyline_check"] = ((te["los"][:, ri, :].astype(float) - (te["eps"][:, None, :] < E_[None, :, None]).astype(float)) ** 2).mean(axis=(1, 2))
            rand = te["ax2"].copy(); rng.shuffle(rand)
            def ci(a):
                return np.percentile([a[rng.integers(0, nu, nu)].mean() for _ in range(500)], [2.5, 97.5]).tolist()
            r = {k: dict(mean=float(v.mean()), ci=ci(v)) for k, v in u.items()}
            for a_, b_ in (("elev", "axis_own"), ("elev", "axis_road"), ("elev", "axis_cf"), ("side_own", "own_side"), ("side_own", "road_side"), ("road_side", "t"), ("t", "t_own"), ("t", "t_road")):
                d = u[a_] - u[b_]; r[f"gain_{a_}_{b_}"] = dict(mean=float(d.mean()), ci=ci(d))
            res[name][c] = r; print(name, c, {k: round(v["mean"], 3) for k, v in r.items()}, flush=True)
    json.dump(res, open(f"{OUT}/brier_ladder.json", "w"), indent=1)


if __name__ == "__main__":
    main()
