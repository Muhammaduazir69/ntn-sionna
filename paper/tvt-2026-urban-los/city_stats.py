#!/usr/bin/env python3
"""Building statistics of each Sionna city scene from its mesh, for the ITU-R P.1410 geometric LOS model.

A roof-height map on a 5 m grid (a ray cast down from above every grid point; ground where it hits z ~ 0)
gives the built-up area fraction alpha, the number of buildings per km^2 beta (connected components of the
built-up mask), and the height distribution, whose Rayleigh scale gamma P.1410 uses (gamma = sqrt(E[H^2]/2)).
Writes out/city_stats.json."""
import json, os
import numpy as np
import mitsuba as mi
import sionna.rt as rt
from scipy import ndimage
from sky_maps import SCENES, ground_scene, ground_z

HERE = os.path.dirname(os.path.abspath(__file__))


def stats(city, step=5.0, margin=100.0):
    sc = rt.load_scene(SCENES[city]); bb = sc.mi_scene.bbox()
    top = float(bb.max.z) + 10.0
    xs = np.arange(bb.min.x + margin, bb.max.x - margin, step); ys = np.arange(bb.min.y + margin, bb.max.y - margin, step)
    X, Y = np.meshgrid(xs, ys); n = X.size
    o = mi.Point3f(mi.Float(X.ravel().astype(np.float32)), mi.Float(Y.ravel().astype(np.float32)), mi.Float(np.full(n, top, np.float32)))
    si = sc.mi_scene.ray_intersect(mi.Ray3f(o, mi.Vector3f(0.0, 0.0, -1.0)))
    roof = np.where(np.array(si.is_valid()), top - np.array(si.t), 0.0)
    g0 = ground_z(ground_scene(sc), X.ravel(), Y.ravel(), top)
    z = (roof - g0).reshape(X.shape)            # height above the local ground (terrain where present)
    built = z > 3.0
    lab, nb = ndimage.label(built)
    area_km2 = X.size * step * step / 1e6
    hb = np.array([z[lab == k].max() for k in range(1, nb + 1)]) if nb else np.array([0.0])
    return dict(alpha=float(built.mean()), beta=float(nb / area_km2), n_buildings=int(nb),
                gamma=float(np.sqrt(np.mean(hb ** 2) / 2)), h_median=float(np.median(hb)), h_mean=float(hb.mean()),
                h_p90=float(np.percentile(hb, 90)), h_max=float(hb.max()), area_km2=float(area_km2))


if __name__ == "__main__":
    res = {c: stats(c) for c in SCENES}
    for c, v in res.items():
        print(c, {k: round(x, 2) for k, x in v.items()})
    json.dump(res, open(os.path.join(HERE, "out", "city_stats.json"), "w"), indent=1)
