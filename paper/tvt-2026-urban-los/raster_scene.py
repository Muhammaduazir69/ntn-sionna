#!/usr/bin/env python3
"""Rasterize a Sionna RT scene to a 1 m height map (top surface and ground) by downward ray casts; writes
out/raster_<city>.npz. The street-free null of street_null.py works on this raster."""
import os, sys
import numpy as np
import mitsuba as mi
import sionna.rt as rt
from sky_maps import SCENES, ground_scene, ground_z
HERE = os.path.dirname(os.path.abspath(__file__))


def run(city, step=1.0):
    sc = rt.load_scene(SCENES[city]); bb = sc.mi_scene.bbox(); gs = ground_scene(sc); top = float(bb.max.z) + 10.0
    xs = np.arange(bb.min.x + step / 2, bb.max.x, step); ys = np.arange(bb.min.y + step / 2, bb.max.y, step)
    Z = np.zeros((len(ys), len(xs)), np.float32); G = np.zeros_like(Z)
    for j0 in range(0, len(ys), 200):      # chunks keep the GPU load small next to the field runs
        Y, X = np.meshgrid(ys[j0:j0 + 200], xs, indexing="ij"); x = X.ravel(); y = Y.ravel(); n = len(x)
        o = mi.Point3f(mi.Float(x.astype(np.float32)), mi.Float(y.astype(np.float32)), mi.Float(np.full(n, top, np.float32)))
        si = sc.mi_scene.ray_intersect(mi.Ray3f(o, mi.Vector3f(0.0, 0.0, -1.0)))
        Z[j0:j0 + 200] = np.where(np.array(si.is_valid()), np.array(si.p.z), -1e3).reshape(X.shape)
        G[j0:j0 + 200] = ground_z(gs, x, y, top).reshape(X.shape)
    np.savez_compressed(os.path.join(HERE, "out", f"raster_{city}.npz"), z=Z, g=G, x0=xs[0], y0=ys[0], step=step)
    print(city, Z.shape, "built share %.3f" % float(((Z - G) > 2).mean()))


if __name__ == "__main__":
    for c in sys.argv[1:]:
        run(c)
