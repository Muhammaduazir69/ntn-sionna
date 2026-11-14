#!/usr/bin/env python3
"""Ray-traced sky maps of LEO line of sight and excess loss in real city scenes (Sionna RT 2.x).

For each city and each sampled outdoor user position (1.5 m above the ground or terrain, not under a roof):
  LOS        exact: a ray from the user toward each direction (elevation e, azimuth a) of the grid, tested
             against the city mesh (Mitsuba ray_test).
  excess     by reciprocity: a transmitter at the user and a horizontal radio map 30 m above the tallest
             building, wide enough to cover every direction above E_MIN; the path gain of the cell a direction
             points to, divided by the free-space gain over the same distance, is the excess loss toward that
             direction (specular reflections and edge diffraction up to depth 3).
Writes out/sky_<city>.npz. Resumable per city.

    python3 sky_maps.py munich 200
"""
import os, sys, time
import numpy as np
import mitsuba as mi
import sionna.rt as rt

HERE = os.path.dirname(os.path.abspath(__file__))
FREQ, E_MIN, CELL, SAMPLES = 2.0e9, 15.0, 10.0, 10 ** 8
ELS = np.arange(15, 90, 5.0)
AZS = np.arange(0, 360, 5.0)
SCENES = {"munich": rt.scene.munich, "etoile": rt.scene.etoile, "florence": rt.scene.florence,
          "san_francisco": rt.scene.san_francisco}


def dirs():
    E, A = np.meshgrid(np.radians(ELS), np.radians(AZS), indexing="ij")
    return np.stack([np.cos(E) * np.sin(A), np.cos(E) * np.cos(A), np.sin(E)], -1)   # x east, y north, z up


GROUND_NAMES = ("Terrain", "ground", "Ground", "terrain")


def ground_scene(sc):
    """A Mitsuba scene holding only the scene's terrain or ground mesh, or None when the ground is z = 0."""
    for nm in GROUND_NAMES:
        try:
            o = sc.get(nm)
        except Exception:
            o = None
        if o is not None:
            return mi.load_dict({"type": "scene", "ground": o.mi_mesh})
    return None


def ground_z(gs, x, y, top):
    if gs is None:
        return np.zeros(len(x))
    n = len(x)
    o = mi.Point3f(mi.Float(np.asarray(x, np.float32)), mi.Float(np.asarray(y, np.float32)), mi.Float(np.full(n, top, np.float32)))
    si = gs.ray_intersect(mi.Ray3f(o, mi.Vector3f(0.0, 0.0, -1.0)))
    return np.where(np.array(si.is_valid()), np.array(si.p.z), 0.0)


def run(city, n_users, seed=1, depth=3, diffuse=False, samples=SAMPLES, tag="", scatter=None, reflect=True, freq=None):
    out = os.path.join(HERE, "out", f"sky_{city}{tag}.npz")
    sc = rt.load_scene(SCENES[city]); sc.frequency = FREQ if freq is None else freq
    if scatter is not None:   # diffuse scattering coefficient of every material (the scenes ship with 0)
        for m in sc.radio_materials.values():
            m.scattering_coefficient = scatter
        diffuse = True
    sc.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    sc.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    bb = sc.mi_scene.bbox()
    gs = ground_scene(sc)
    top = float(bb.max.z)
    H = top + 30.0
    L = 2.0 * (H - 1.5) / np.tan(np.radians(E_MIN)) + 4 * CELL
    rng = np.random.default_rng(seed)
    D = dirs()
    users, los, exc = [], [], []
    t0 = time.time()
    margin = 100.0
    while len(users) < n_users:
        x0, y0 = rng.uniform(bb.min.x + margin, bb.max.x - margin), rng.uniform(bb.min.y + margin, bb.max.y - margin)
        p = np.array([x0, y0, float(ground_z(gs, [x0], [y0], top + 10.0)[0]) + 1.5])   # 1.5 m above the ground
        if bool(sc.mi_scene.ray_test(mi.Ray3f(mi.Point3f(*map(float, p)), mi.Vector3f(0.0, 0.0, 1.0)))):
            continue   # under a roof: indoor or inside a building
        # exact LOS toward every grid direction
        d = D.reshape(-1, 3).astype(np.float32); n = len(d)
        o = mi.Point3f(mi.Float(np.full(n, p[0], np.float32)), mi.Float(np.full(n, p[1], np.float32)), mi.Float(np.full(n, p[2], np.float32)))
        blk = np.array(sc.mi_scene.ray_test(mi.Ray3f(o, mi.Vector3f(mi.Float(d[:, 0]), mi.Float(d[:, 1]), mi.Float(d[:, 2])))))
        # excess loss by reciprocity
        for name in [o_.name for o_ in sc.transmitters.values()] if hasattr(sc, "transmitters") else []:
            sc.remove(name)
        sc.add(rt.Transmitter("ue", position=[float(x) for x in p]))
        rm = rt.RadioMapSolver()(sc, center=[float(p[0]), float(p[1]), float(H)], orientation=[0.0, 0.0, 0.0],
                                 size=[float(L), float(L)], cell_size=[CELL, CELL], samples_per_tx=samples, max_depth=depth,
                                 los=True, specular_reflection=reflect, diffuse_reflection=diffuse, diffraction=True,
                                 edge_diffraction=True, refraction=False)
        sc.remove("ue")
        g = np.array(rm.path_gain).reshape(-1); cc = np.array(rm.cell_centers).reshape(-1, 3)
        nx = int(round(np.sqrt(len(g))))
        # the cell each grid direction points to on the plane z = H
        s = (H - p[2]) / d[:, 2]
        hit = p[None, :] + s[:, None] * d
        ix = np.clip(np.floor((hit[:, 0] - (p[0] - L / 2)) / CELL).astype(int), 0, nx - 1)
        iy = np.clip(np.floor((hit[:, 1] - (p[1] - L / 2)) / CELL).astype(int), 0, nx - 1)
        # cell_centers ordering: rows along y, columns along x
        G = g.reshape(nx, nx); C = cc.reshape(nx, nx, 3)
        gk = G[iy, ix]; ck = C[iy, ix]
        dist = np.linalg.norm(ck - p[None, :], axis=1)
        fs = (299792458.0 / float(sc.frequency[0] if hasattr(sc.frequency, "__len__") else sc.frequency) / (4 * np.pi * dist)) ** 2
        exc.append((10 * np.log10(np.maximum(gk, 1e-30) / fs)).reshape(D.shape[:2]))
        los.append((~blk).reshape(D.shape[:2]))
        users.append(p)
        if len(users) % 20 == 0:
            print(f"{city}: {len(users)}/{n_users} users, {time.time() - t0:.0f} s", flush=True)
            np.savez_compressed(out, users=np.array(users), los=np.array(los), exc=np.array(exc), els=ELS, azs=AZS, H=H, L=L)
    np.savez_compressed(out, users=np.array(users), los=np.array(los), exc=np.array(exc), els=ELS, azs=AZS, H=H, L=L)
    print(city, "done", time.time() - t0, flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("city"); ap.add_argument("n", type=int, nargs="?", default=200)
    ap.add_argument("--depth", type=int, default=3); ap.add_argument("--diffuse", action="store_true")
    ap.add_argument("--samples", type=float, default=SAMPLES); ap.add_argument("--tag", default="")
    ap.add_argument("--scatter", type=float, default=None)
    ap.add_argument("--noreflect", action="store_true")
    ap.add_argument("--freq", type=float, default=None)
    a = ap.parse_args()
    run(a.city, a.n, depth=a.depth, diffuse=a.diffuse, samples=int(a.samples), tag=a.tag, scatter=a.scatter, reflect=not a.noreflect, freq=a.freq)
