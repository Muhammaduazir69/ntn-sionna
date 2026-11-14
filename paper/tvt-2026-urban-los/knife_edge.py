#!/usr/bin/env python3
"""Knife-edge prediction of the excess loss toward each blocked direction, from the city mesh alone.

For a user and a blocked direction (e, a) the ray hits a facade at slant distance d1 and height z_hit; the roof
above that point, found by a ray cast down from above, is at height H_roof. With the satellite at infinity the
blocking edge is a single knife edge of height h = H_roof - z_hit above the ray, measured vertically, or
h cos(e) across the ray, so the Fresnel parameter is nu = h cos(e) sqrt(2 / (lambda d1)) and the loss is the
ITU-R P.526 approximation J(nu) = 6.9 + 20 log10( sqrt((nu - 0.1)^2 + 1) + nu - 0.1 ) for nu > -0.78.
Writes out/knife_<city>.npz with nu and J for every user and grid direction (NaN where in line of sight)."""
import os, sys
import numpy as np
import mitsuba as mi
import sionna.rt as rt
from sky_maps import SCENES, FREQ, dirs

HERE = os.path.dirname(os.path.abspath(__file__))


def J(nu):
    nu = np.asarray(nu, float)
    return np.where(nu > -0.78, 6.9 + 20 * np.log10(np.sqrt((nu - 0.1) ** 2 + 1) + nu - 0.1), 0.0)


def run(city):
    z = np.load(os.path.join(HERE, "out", f"sky_{city}.npz"))
    sc = rt.load_scene(SCENES[city])
    top = float(sc.mi_scene.bbox().max.z) + 10.0
    lam = 299792458.0 / FREQ
    D = dirs(); shp = D.shape[:2]; d = D.reshape(-1, 3).astype(np.float32); n = len(d)
    NU = []
    for p in z["users"]:
        o = mi.Point3f(mi.Float(np.full(n, p[0], np.float32)), mi.Float(np.full(n, p[1], np.float32)), mi.Float(np.full(n, p[2], np.float32)))
        si = sc.mi_scene.ray_intersect(mi.Ray3f(o, mi.Vector3f(mi.Float(d[:, 0]), mi.Float(d[:, 1]), mi.Float(d[:, 2]))))
        valid = np.array(si.is_valid()); t = np.array(si.t)
        hit = np.array(p)[None, :] + t[:, None] * d
        # roof height above the hit point: cast down from above the city
        od = mi.Point3f(mi.Float(hit[:, 0].astype(np.float32)), mi.Float(hit[:, 1].astype(np.float32)), mi.Float(np.full(n, top, np.float32)))
        sd = sc.mi_scene.ray_intersect(mi.Ray3f(od, mi.Vector3f(0.0, 0.0, -1.0)))
        roof = top - np.array(sd.t)
        h = np.maximum(roof - hit[:, 2], 0.0)
        e = np.arcsin(d[:, 2])
        nu = h * np.cos(e) * np.sqrt(2.0 / (lam * np.maximum(t, 1.0)))
        NU.append(np.where(valid, nu, np.nan).reshape(shp))
    NU = np.array(NU)
    np.savez_compressed(os.path.join(HERE, "out", f"knife_{city}.npz"), nu=NU, J=np.where(np.isnan(NU), np.nan, J(np.nan_to_num(NU))))
    print(city, "done", NU.shape)


if __name__ == "__main__":
    run(sys.argv[1])
