#!/usr/bin/env python3
"""LOS probability and excess loss by elevation for each city, against TR 38.811 and an ITU-R P.1410
geometric model parameterized from each city's mesh; writes out/analysis.json.

P.1410 for a satellite at elevation e and a user at height h_r: buildings stand at a mean spacing
1/sqrt(alpha beta) along the ray's ground track (alpha: built-up fraction, beta: buildings per km^2, so
spacing in km); the ray is at height h_n = h_r + (n + 1/2) s tan(e) above the n-th building's position, and
P_LOS = prod_n [1 - exp(-h_n^2 / (2 gamma^2))] with Rayleigh heights of scale gamma."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CITIES = ("munich", "etoile", "florence", "san_francisco")
TR_EL = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90.0])
TR_LOS = {"dense_urban": np.array([28.2, 33.1, 39.8, 46.8, 53.7, 61.2, 73.8, 82.0, 98.1]) / 100,
          "urban": np.array([24.6, 38.6, 49.3, 61.3, 72.6, 80.5, 91.9, 96.8, 99.2]) / 100}
TR_CL_S = {"dense_urban": np.array([34.3, 30.9, 29.0, 27.7, 26.8, 26.2, 25.8, 25.5, 25.5]),
           "urban": np.array([34.3, 30.9, 29.0, 27.7, 26.8, 26.2, 25.8, 25.5, 25.5])}
TR_SF_NLOS_S = {"dense_urban": np.array([15.5, 13.9, 12.4, 11.7, 10.6, 10.5, 10.1, 9.2, 9.2]),
                "urban": np.full(9, 6.0)}


def nearest(els, table):
    """TR 38.811 rule: the value at the reference elevation (10, 20, ..., 90 deg) nearest to the elevation."""
    idx = np.clip(np.rint(np.asarray(els, float) / 10.0).astype(int) - 1, 0, 8)
    return np.asarray(table)[idx]


def p1410(e_deg, alpha, beta, gamma, h_r=1.5, n_max=200):
    s = 1000.0 / np.sqrt(alpha * beta)             # mean building spacing along the ground track (m)
    n = np.arange(n_max)
    h = h_r + (n + 0.5) * s * np.tan(np.radians(e_deg))
    return float(np.prod(1 - np.exp(-h ** 2 / (2 * gamma ** 2))))


def main():
    S = json.load(open(os.path.join(HERE, "out", "city_stats.json")))
    res = {}
    for c in CITIES:
        f = os.path.join(HERE, "out", f"sky_{c}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f); los, exc, els = z["los"], z["exc"], z["els"]
        st = S[c]
        r = dict(n_users=int(los.shape[0]), els=els.tolist(), los=los.mean(axis=(0, 2)).tolist(),
                 los_p1410=[p1410(e, st["alpha"], st["beta"], st["gamma"]) for e in els],
                 nlos_med=[], nlos_p10=[], nlos_p90=[], nlos_std=[], los_exc_med=[])
        for i in range(len(els)):
            m = ~los[:, i]
            x = -exc[:, i][m]
            x = x[x < 150]                          # cells no ray reached are reported separately
            r["nlos_med"].append(float(np.median(x)) if len(x) else None)
            r["nlos_p10"].append(float(np.percentile(x, 10)) if len(x) else None)
            r["nlos_p90"].append(float(np.percentile(x, 90)) if len(x) else None)
            r["nlos_std"].append(float(np.std(x)) if len(x) else None)
            r["los_exc_med"].append(float(np.median(-exc[:, i][los[:, i]])))
        r["unreached_share"] = float(np.mean(-exc[~los] >= 150))
        # azimuth dependence: spread of the LOS share over azimuth at 30 deg elevation
        i30 = list(els).index(30.0)
        az = los[:, i30, :].mean(axis=0)
        r["los30_az_min"], r["los30_az_max"] = float(az.min()), float(az.max())
        res[c] = r
        print(c, "LOS", np.round(r["los"], 2), "\n   P1410", np.round(r["los_p1410"], 2), "\n   NLOS med", np.round(r["nlos_med"], 1))
    E = np.arange(15, 90, 5.0)
    tr = {k: nearest(E, v).tolist() for k, v in TR_LOS.items()}
    res["tr38811"] = dict(els=E.tolist(), los=tr, cl_s={k: nearest(E, v).tolist() for k, v in TR_CL_S.items()},
                          note="TR 38.811 V15.4.0 Tables 6.6.1-1 and 6.6.2-1/2 (verified against the spec text); nearest reference angle")
    json.dump(res, open(os.path.join(HERE, "out", "analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
