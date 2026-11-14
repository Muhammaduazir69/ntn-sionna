#!/usr/bin/env python3
"""Local street geometry of every sampled user: horizontal rays at 1.5 m in 72 azimuths give the distance to the
first facade in each direction; the street axis is the azimuth with the longest free run (folded to 0..180 deg),
and the street width is the sum of the free distances perpendicular to it; the height above ground of the
facade hit in each azimuth is read from the roof above it. Writes out/street_<city>.npz with axis_deg,
width_m, free (distance per azimuth) and hfac (facade height per azimuth)."""
import os, sys
import numpy as np
import mitsuba as mi
import sionna.rt as rt
from sky_maps import SCENES, ground_scene, ground_z

HERE = os.path.dirname(os.path.abspath(__file__))
AZ = np.arange(0, 360, 5.0)


def run(city, src="sky", dst="street", tag=""):
    z = np.load(os.path.join(HERE, "out", f"{src}_{city}{tag}.npz"))
    sc = rt.load_scene(SCENES[city])
    a = np.radians(AZ); d = np.stack([np.sin(a), np.cos(a), np.zeros_like(a)], -1).astype(np.float32); n = len(a)
    free, axis, width, hfac, sky, axis2 = [], [], [], [], [], []
    top = float(sc.mi_scene.bbox().max.z) + 10.0
    gs = ground_scene(sc)
    for p in z["users"]:
        o = mi.Point3f(mi.Float(np.full(n, p[0], np.float32)), mi.Float(np.full(n, p[1], np.float32)), mi.Float(np.full(n, p[2], np.float32)))
        si = sc.mi_scene.ray_intersect(mi.Ray3f(o, mi.Vector3f(mi.Float(d[:, 0]), mi.Float(d[:, 1]), mi.Float(d[:, 2]))))
        t = np.where(np.array(si.is_valid()), np.array(si.t), 500.0)
        t = np.minimum(t, 500.0)
        # axis: azimuth (mod 180) with the largest free run in both senses
        both = t[: n // 2] + t[n // 2:]
        k = int(np.argmax(both))
        axis.append(AZ[k])
        k_perp = (k + n // 4) % (n // 2)
        width.append(float(both[k_perp]))
        # independent axis: perpendicular to the narrowest run in both senses (the wall normal)
        axis2.append((AZ[int(np.argmin(both))] + 90.0) % 180.0)
        free.append(t)
        # height above ground of the facade hit in each azimuth: roof above a point 0.5 m inside the facade
        hx = p[0] + (t + 0.5) * d[:, 0]; hy = p[1] + (t + 0.5) * d[:, 1]
        od = mi.Point3f(mi.Float(hx.astype(np.float32)), mi.Float(hy.astype(np.float32)), mi.Float(np.full(n, top, np.float32)))
        sd = sc.mi_scene.ray_intersect(mi.Ray3f(od, mi.Vector3f(0.0, 0.0, -1.0)))
        roof = np.where(np.array(sd.is_valid()), top - np.array(sd.t), 0.0)
        # facade height measured from the USER's ground, the reference of the ray height h + t tan e; on flat
        # scenes this equals the height above the facade's own ground, on terrain it accounts for the slope
        hfac.append(np.where(t < 500.0, roof - (p[2] - 1.5), 0.0))
        # 2.5D skyline: highest elevation angle of roofs or terrain along each azimuth's ground track (5 m steps
        # out to 1 km), from downward casts on the height map; a satellite is clear iff it is above it
        rr = np.arange(2.5, 1000.0, 5.0)
        gx = (p[0] + rr[None, :] * d[:, 0:1]).ravel(); gy = (p[1] + rr[None, :] * d[:, 1:2]).ravel()
        og = mi.Point3f(mi.Float(gx.astype(np.float32)), mi.Float(gy.astype(np.float32)), mi.Float(np.full(gx.size, top, np.float32)))
        sg = sc.mi_scene.ray_intersect(mi.Ray3f(og, mi.Vector3f(0.0, 0.0, -1.0)))
        hz = np.where(np.array(sg.is_valid()), top - np.array(sg.t), -1e9).reshape(n, len(rr))
        sky.append(np.degrees(np.arctan2(hz - p[2], rr[None, :])).max(axis=1))
    np.savez_compressed(os.path.join(HERE, "out", f"{dst}_{city}{tag}.npz"), axis_deg=np.array(axis), width_m=np.array(width),
                        free=np.array(free), hfac=np.array(hfac), sky=np.array(sky), az=AZ, axis2_deg=np.array(axis2))
    print(city, "median width %.1f m" % np.median(width))


if __name__ == "__main__":
    run(sys.argv[1], *(sys.argv[2:5]))
