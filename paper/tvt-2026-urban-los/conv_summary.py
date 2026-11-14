#!/usr/bin/env python3
"""Convergence and sensitivity of the excess loss on Munich (60 interior users, the same users in every run):
writes out/conv2.json with, per run, plane distance, reference cell, rays, the median loss of reached blocked
directions, the censored median (unreached = +inf), the 90th percentile, the unreached share and the LOS-pair
normalization percentiles."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
RUNS = [("_po30", 30, 10, 1e8), ("_po100", 100, 10, 1e8), ("_po300", 300, 10, 1e8), ("_po1000", 1000, 10, 1e8),
        ("_po3000", 3000, 10, 1e8), ("_po1000c5", 1000, 5, 4e8), ("_po1000c25", 1000, 2.5, 1e8),
        ("_po1000c25s10", 1000, 2.5, 1e9), ("_po1000c125", 1000, 1.25, 1e8),
        ("_sbase", 1000, 2.5, 1e9), ("_sd6", 1000, 2.5, 1e9), ("_ssc", 1000, 2.5, 1e9), ("_snr", 1000, 2.5, 1e9),
        ("_sph", 1000, 2.5, 1e9), ("_sf16", 1000, 2.5, 1e9), ("_sf35", 1000, 2.5, 1e9)]
res = {}
for tag, po, cell, rays in RUNS:
    p = os.path.join(OUT, f"sky2_munich{tag}.npz")
    if not os.path.exists(p):
        continue
    z = np.load(p)
    if z["los"].shape[0] < 60:
        continue
    L = -z["exc"]; m = ~z["los"]; x = L[m]; xc = np.where(x < 150, x, np.inf); lo = L[z["los"]]
    res[tag] = dict(plane_off=po, cell=cell, rays=rays, plane_dist=float(np.median(z["plane_dist"])),
                    med=float(np.median(x[x < 150])), med_cens=float(np.median(xc)),
                    p90=float(np.percentile(xc, 90)) if np.isfinite(np.percentile(xc, 90)) else None,
                    unreached=float(np.mean(x >= 150)), los_p1=float(np.percentile(lo, 1)),
                    los_p99=float(np.percentile(lo, 99)), los_med=float(np.median(lo)))
json.dump(res, open(os.path.join(OUT, "conv2.json"), "w"), indent=1)
for k, v in res.items():
    print(k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items()})
