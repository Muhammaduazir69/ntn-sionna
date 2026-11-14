#!/usr/bin/env python3
"""Band statistics of the Munich 60-user settings: median of reached blocked directions, censored median and unreached
share in 15-30, 30-60 and 60-85 deg, and the margin to the TR 38.811 clutter loss. Writes out/conv_bands.json."""
import glob, json, os, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
TR_EL = np.arange(10, 91, 10.0); TR_CL = np.array([34.3, 30.9, 29.0, 27.7, 26.8, 26.2, 25.8, 25.5, 25.5])
TAGS = ("_sbase", "_po3000u", "_c056r4e9", "_scp", "_sd6", "_sd9", "_snr", "_ssc", "_sph", "_sf16", "_sf35")
res = {}
for p in sorted(glob.glob(os.path.join(OUT, "sky2_munich_*.npz"))):
    t = os.path.basename(p)[len("sky2_munich"):-4]
    if not (t in TAGS or t.startswith("_w")):
        continue
    z = np.load(p); L = -z["exc"]; los = z["los"]; els = z["els"]
    if los.shape[0] < 60:
        continue
    L, los = L[:60], los[:60]
    bands = []
    for lo, hi in ((15, 30), (30, 60), (60, 86)):
        k = (els >= lo) & (els < hi); x = L[:, k, :][~los[:, k, :]]; r = x[x < 150]
        tr = np.interp(np.broadcast_to(els[k][None, :, None], L[:, k, :].shape)[~los[:, k, :]], TR_EL, TR_CL)
        c = np.where(x < 150, x, np.inf)
        bands.append([float(np.median(r)), float((x >= 150).mean()), float(np.median(c)), float(np.median(tr)),
                      float(np.mean(c < tr))])
    res[t] = dict(bands=bands, los_med=float(np.median(L[los])))
json.dump(res, open(os.path.join(OUT, "conv_bands.json"), "w"), indent=1)
if __name__ == "__main__":
    print("%-12s %s" % ("tag", "med/cens/unr%/shareBelowTR per band"))
    for t, v in res.items():
        print("%-12s" % t, " | ".join("%5.1f %5.1f %4.1f %.2f" % (b[0], b[2], 100 * b[1], b[4]) for b in v["bands"]), " TR", " ".join("%.1f" % b[3] for b in v["bands"]))
