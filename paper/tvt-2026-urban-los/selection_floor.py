#!/usr/bin/env python3
"""Proposition (selection floor): swing that the own-axis rule produces in street-free clutter. Free runs t(a) ~ Exp(lam),
K independent folded runs W = t(a) + t(a+180) ~ Gamma(2, lam), axis across the smallest. Given W = w, t is uniform on (0, w), so
p_ac(z) = E[(1 - z/W_min)^+] = int_z^inf (z / w^2) (P(W > w))^K dw, with P(W > w) = (1 + lam w) exp(-lam w); p_al = exp(-lam z).
With P = exp(-lam z) the mean LOS, everything depends on (P, K) only. K_eff is calibrated on one null and tested on the others."""
import json, os
import numpy as np
from scipy.integrate import quad
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")


def p_ac(P, K):
    lz = -np.log(P)           # lam z
    f = lambda x: lz / x ** 2 * ((1 + x) * np.exp(-x)) ** K
    return quad(f, lz, np.inf, limit=200)[0]


def swing(P, K):
    return P - p_ac(P, K)


def main():
    r = json.load(open(f"{OUT}/street_null.json")); rows = []
    for c, o in r.items():
        for nm in ("iso", "rotm"):
            for s in o[nm]:
                rows.append((c, nm, s["los30"], s["own"]))
    rows_iso = [x for x in rows if x[1] == "iso"]
    Ks = np.arange(1, 37); cal = [x for x in rows_iso if x[0] == "munich"]
    e = [np.mean([(swing(x[2], K) - x[3]) ** 2 for x in cal]) for K in Ks]; K = int(Ks[int(np.argmin(e))])
    res = dict(K_eff=K, table={})
    for c, o in r.items():
        tb = {}
        for nm in ("iso", "rotm"):
            ss = o[nm]; P = np.mean([s["los30"] for s in ss]); tb[nm] = dict(P=P, sim=float(np.mean([s["own"] for s in ss])), model=float(swing(P, K)))
        P = o["real"]["los30"]; tb["real"] = dict(P=P, sim=o["real"]["own"], model=float(swing(P, K)))
        res["table"][c] = tb
    # predictive skill, 12 means: model against a constant set on the same calibration case (R4)
    sim = np.array([tb[k]["sim"] for tb in res["table"].values() for k in tb]); mod = np.array([tb[k]["model"] for tb in res["table"].values() for k in tb])
    from scipy.stats import spearmanr
    res["skill"] = dict(rmse_model=float(np.sqrt(np.mean((sim - mod) ** 2))), rmse_const_mean=float(sim.std()), spearman=float(spearmanr(sim, mod)[0]), above=int((mod > sim).sum()), n=len(sim))
    json.dump(res, open(f"{OUT}/selection_floor.json", "w"), indent=1); print(json.dumps(res, indent=0))
    print("K sensitivity at P=.5", [round(swing(.5, k), 3) for k in (2, 4, 6, 8, 12, 18, 36)])


if __name__ == "__main__":
    main()
