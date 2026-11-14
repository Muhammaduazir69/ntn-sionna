#!/usr/bin/env python3
"""M2: size of the mixed diffraction-then-reflection path the radio-map solver does not trace. Canonical canyon: width W,
facades of height H, user at height h and distance t from the near facade, far source at elevation e behind that
facade. Direct path: knife edge J(nu1). Reflected path: knife edge to the image of the user in the opposite facade
(horizontal distance 2W - t) times the slab reflection power. Only cases with the image also in shadow count: when
it is lit the source reaches the opposite wall directly and the solver traces that specular path. Writes
out/canyon_bound.json with the drop of the loss when the reflected path is added in power."""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); LAM = 299792458.0 / 2e9


def J(nu):
    return 6.9 + 20 * np.log10(np.sqrt((nu - 0.1) ** 2 + 1) + nu - 0.1)


def slab(thi, eps=5.24, c=0.0462, dd=0.7822, f=2e9, d=0.3):
    fg = f / 1e9; eta = eps - 1j * 17.98 * c * fg ** dd / fg; th = np.radians(thi); co = np.cos(th); s2 = np.sin(th) ** 2
    rt_ = np.sqrt(eta - s2); te = (co - rt_) / (co + rt_); tm = (eta * co - rt_) / (eta * co + rt_); e = np.exp(-2j * 2 * np.pi * d / LAM * rt_)
    return 0.5 * (np.abs(te * (1 - e) / (1 - te ** 2 * e)) ** 2 + np.abs(tm * (1 - e) / (1 - tm ** 2 * e)) ** 2)


rows = []
for W in (10, 14, 20, 30):
    for H in (12, 16, 22):
        for e in np.arange(15, 61, 5.0):
            for frac in (0.25, 0.5, 0.75):
                t = frac * W; h = 1.5; te = np.tan(np.radians(e)); ce = np.cos(np.radians(e))
                D1 = H - h - t * te; x2 = 2 * W - t; D2 = H - h - x2 * te
                if D1 <= 0 or D2 <= 0:
                    continue
                d1 = np.hypot(t, H - h); d2 = np.hypot(x2, H - h)
                J1 = J(D1 * ce * np.sqrt(2 / (LAM * d1))); J2 = J(D2 * ce * np.sqrt(2 / (LAM * d2)))
                inc = np.degrees(np.arctan2(H - h, x2))     # incidence on the far wall from its normal
                G = slab(inc)
                drop = 10 * np.log10(1 + G * 10 ** (-(J2 - J1) / 10))
                rows.append((W, H, e, frac, J1, J2, 10 * np.log10(G), drop))
r = np.array(rows)
res = dict(n=len(r), drop_median=float(np.median(r[:, 7])), drop_p90=float(np.percentile(r[:, 7], 90)), drop_max=float(r[:, 7].max()),
           j1_median=float(np.median(r[:, 4])))
json.dump(res, open(os.path.join(HERE, "out", "canyon_bound.json"), "w"), indent=1); print(res)
