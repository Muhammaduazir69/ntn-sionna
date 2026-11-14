#!/usr/bin/env python3
"""LOS persistence along streets: for users whose two street-axis estimates agree (in a street), walk +-150 m
along the axis in 1 m steps at 1.5 m above the ground and test LOS toward a satellite at elevation e and
azimuth a0 + rel (rel from the street axis), the satellite direction held fixed over the walk (a vehicle
covers 300 m in about 20 s, while a LEO satellite moves a few degrees). Steps under a roof end the walk.
Writes out/runs_<city>.json: mean LOS and blocked run lengths (m) per (elevation, relative azimuth)."""
import json, os, sys
import numpy as np
import mitsuba as mi
import sionna.rt as rt
from sky_maps import SCENES, ground_scene, ground_z

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
ELS = (20.0, 30.0, 45.0, 60.0); RELS = (0.0, 30.0, 60.0, 90.0); HALF, STEP = 150.0, 1.0


def runs(b):
    """Lengths of runs of True and of False in a boolean sequence, excluding the runs cut by the ends."""
    ch = np.flatnonzero(np.diff(b.astype(int))) + 1
    edges = np.concatenate([[0], ch, [len(b)]])
    L = np.diff(edges); v = b[edges[:-1]]
    inner = np.ones(len(L), bool); inner[0] = inner[-1] = False
    return L[inner & v], L[inner & ~v]


def main(city, tag=""):
    z = np.load(os.path.join(OUT, f"sky2_{city}{tag}.npz")); s = np.load(os.path.join(OUT, f"street2_{city}{tag}.npz"))
    sc = rt.load_scene(SCENES[city]); gs = ground_scene(sc); top = float(sc.mi_scene.bbox().max.z) + 10.0
    agree = np.abs(((s["axis_deg"] - s["axis2_deg"] + 90) % 180) - 90) <= 15
    acc = {f"{e:g}_{r:g}": dict(los=[], nlos=[], share=[]) for e in ELS for r in RELS}
    for p, ax, ok in zip(z["users"], s["axis2_deg"], agree):
        if not ok:
            continue
        u = np.array([np.sin(np.radians(ax)), np.cos(np.radians(ax))])
        k = np.arange(-HALF, HALF + STEP, STEP)
        xy = p[:2][None, :] + k[:, None] * u[None, :]
        zg = ground_z(gs, xy[:, 0], xy[:, 1], top) + 1.5
        P = np.c_[xy, zg].astype(np.float32); n = len(P)
        up = sc.mi_scene.ray_test(mi.Ray3f(mi.Point3f(mi.Float(P[:, 0]), mi.Float(P[:, 1]), mi.Float(P[:, 2])), mi.Vector3f(0.0, 0.0, 1.0)))
        out_ok = ~np.array(up)
        # keep the contiguous outdoor stretch around the user
        c0 = n // 2; lo = c0; hi = c0
        while lo > 0 and out_ok[lo - 1]:
            lo -= 1
        while hi < n - 1 and out_ok[hi + 1]:
            hi += 1
        if hi - lo < 60:
            continue
        Q = P[lo:hi + 1]; m = len(Q)
        for e in ELS:
            for r in RELS:
                az = np.radians(ax + r); er = np.radians(e)
                d = np.array([np.cos(er) * np.sin(az), np.cos(er) * np.cos(az), np.sin(er)], np.float32)
                blk = np.array(sc.mi_scene.ray_test(mi.Ray3f(mi.Point3f(mi.Float(Q[:, 0]), mi.Float(Q[:, 1]), mi.Float(Q[:, 2])),
                                                             mi.Vector3f(float(d[0]), float(d[1]), float(d[2])))))
                l, b = runs(~blk)
                a_ = acc[f"{e:g}_{r:g}"]; a_["los"] += (l * STEP).tolist(); a_["nlos"] += (b * STEP).tolist(); a_["share"].append(float((~blk).mean()))
    res = {k: dict(n_los=len(v["los"]), n_nlos=len(v["nlos"]), mean_los=float(np.mean(v["los"])) if v["los"] else None,
                   mean_nlos=float(np.mean(v["nlos"])) if v["nlos"] else None, los_share=float(np.mean(v["share"])) if v["share"] else None)
           for k, v in acc.items()}
    res["n_walks"] = int(len(acc[f"{ELS[0]:g}_{RELS[0]:g}"]["share"]))
    json.dump(res, open(os.path.join(OUT, f"runs_{city}{tag}.json"), "w"), indent=1)
    print(city, "walks", res["n_walks"], {k: (round(v["mean_los"] or 0), round(v["mean_nlos"] or 0), round(v["los_share"] or 0, 2)) for k, v in res.items() if k.startswith("30_")})


if __name__ == "__main__":
    main(*sys.argv[1:3])
