#!/usr/bin/env python3
"""R4: how the OSM registration holds up. Share of drivable centerline points inside buildings over shifts of up to 20 m in 1 m steps, and the road-axis swing
at 30 degrees at the shift that minimizes that share. Stored registration is not changed. Writes out/osm_shift.json."""
import json
import numpy as np
from scipy.spatial import cKDTree
import street_null2 as sn2
OUT = sn2.OUT; ELS = np.arange(15, 90, 5.0); I30 = list(ELS).index(30.0); AZ = np.arange(0, 360, 5.0); TAN = np.tan(np.radians(ELS))


def swing(l, t, ax):
    R = np.abs(((AZ[None, :] - ax[:, None] + 90) % 180) - 90); al, ac = t & (R <= 15), t & (R >= 75)
    return float((l * al).sum() / al.sum() - (l * ac).sum() / ac.sum())


res = {}
for c in ("munich", "etoile", "florence", "san_francisco"):
    r = np.load(f"{OUT}/raster_{c}.npz"); S = (r["z"] - r["g"]) > 2; x0, y0 = float(r["x0"]), float(r["y0"]); q = np.load(f"{OUT}/osm_roads_{c}.npz"); m, a = q["mid"], q["ang"]; md = m[q["drive"]]
    def share(dx, dy):
        i = np.rint(md[:, 1] + dy - y0).astype(int); j = np.rint(md[:, 0] + dx - x0).astype(int); ok = (i >= 0) & (i < S.shape[0]) & (j >= 0) & (j < S.shape[1]); return float(S[i[ok], j[ok]].mean())
    G = {(dx, dy): share(dx, dy) for dx in range(-20, 21) for dy in range(-20, 21)}; best = min(G, key=G.get); ring = [share(4 * np.cos(t), 4 * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    o = dict(at0=G[(0, 0)], best=list(best), best_dist=float(np.hypot(*best)), best_share=G[best], ring4_min=min(ring), ring4_max=max(ring))
    for tag, nm in (("", "seed1"), ("_c03sd2", "seed2")):
        z = np.load(f"{OUT}/sky2_{c}{tag}.npz"); U = z["users"][:, :2]; l = z["los"][:, I30, :].astype(float); t = (z["d_edge"][:, None, :] * TAN[None, :, None] >= float(z["H95"]))[:, I30, :]
        ax = lambda dx, dy: np.rint(a[cKDTree(m + [dx, dy]).query(U)[1]] / 5) * 5 % 180
        o[nm] = dict(swing0=swing(l, t, ax(0, 0)), swing_best=swing(l, t, ax(*best)), dist_med=float(np.median(cKDTree(m).query(U)[0])))
    res[c] = o; print(c, o, flush=True)
json.dump(res, open(f"{OUT}/osm_shift.json", "w"), indent=1)
