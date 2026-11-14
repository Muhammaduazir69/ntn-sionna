"""Edge distance helper of sky_maps2.py without the Sionna import, for the CPU-only null scripts."""
import numpy as np


def dist_to_edge(poly, p, az):
    d = np.stack([np.sin(np.radians(az)), np.cos(np.radians(az))], 1)
    a = poly; b = np.roll(poly, -1, axis=0); e = b - a
    out = np.full(len(az), np.inf)
    for k in range(len(a)):
        den = d[:, 0] * e[k, 1] - d[:, 1] * e[k, 0]
        w = a[k] - p
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (w[0] * e[k, 1] - w[1] * e[k, 0]) / den
            u = (w[0] * d[:, 1] - w[1] * d[:, 0]) / den
        ok = (np.abs(den) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1)
        out = np.where(ok, np.minimum(out, t), out)
    return out
