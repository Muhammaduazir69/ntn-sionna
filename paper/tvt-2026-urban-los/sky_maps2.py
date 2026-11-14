#!/usr/bin/env python3
"""Ray-traced sky maps, version 2 (Sionna RT 2.x): interior users, edge-masked directions, plane-distance and
cell-size control, nearest-center cell lookup.

Differences from sky_maps.py, after the review of version 1:
  users      sampled only inside the convex hull of the built area (raster of 10 m downward casts, built where the
             roof is more than 4 m above the ground), at least --inner m from its boundary;
  edge mask  for each user and azimuth the distance d_edge to the hull boundary is stored with H95, the 95th
             percentile of building height above ground; a direction is trusted when d_edge * tan(e) >= H95, so
             the ray clears almost every building height before it leaves the modeled city;
  plane      at --planeOff m above the tallest roof; cells scale with the user-plane distance so that a cell
             subtends the same angle at every plane distance (--cell is the size at planeOff = 30 m);
  lookup     the cell a direction meets is the cell whose center is nearest the hit point (checked);
  pol        --pol V or H at both ends.
Writes out/sky2_<city><tag>.npz."""
import json, os, time
import numpy as np
import mitsuba as mi
import sionna.rt as rt
from scipy.spatial import ConvexHull
from sky_maps import ELS, AZS, SCENES, dirs, ground_scene, ground_z

HERE = os.path.dirname(os.path.abspath(__file__))
E_MIN = 15.0


def built_area(sc, gs, top, step=10.0):
    bb = sc.mi_scene.bbox()
    xs = np.arange(bb.min.x + step / 2, bb.max.x, step); ys = np.arange(bb.min.y + step / 2, bb.max.y, step)
    X, Y = np.meshgrid(xs, ys); x, y = X.ravel(), Y.ravel(); n = len(x)
    o = mi.Point3f(mi.Float(x.astype(np.float32)), mi.Float(y.astype(np.float32)), mi.Float(np.full(n, top + 10.0, np.float32)))
    si = sc.mi_scene.ray_intersect(mi.Ray3f(o, mi.Vector3f(0.0, 0.0, -1.0)))
    zr = np.where(np.array(si.is_valid()), np.array(si.p.z), -1e9)
    zg = ground_z(gs, x, y, top + 10.0)
    h = zr - zg
    b = h > 4.0
    return np.stack([x[b], y[b]], 1), h[b]


def hull_poly(pts):
    hl = ConvexHull(pts)
    return pts[hl.vertices]


def dist_to_edge(poly, p, az):
    """Distance from p (inside) along each azimuth (deg, from north, clockwise) to the polygon boundary."""
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


def inside_dist(poly, p):
    """Signed distance of p to the polygon boundary, positive inside (convex polygon, counterclockwise)."""
    a = poly; b = np.roll(poly, -1, axis=0); e = b - a
    nrm = np.stack([e[:, 1], -e[:, 0]], 1) / np.linalg.norm(e, axis=1)[:, None]   # outward for ccw
    return float(np.min(np.sum((a - p) * nrm, axis=1)))


def set_facades(sc, wall=None, thick=None, roof=None, ground_thick=None):
    """Facade slab of the scenes: ITU type `wall` (the scenes ship itu_marble) and thickness `thick` in meters (the
    scenes ship 0.1 m, which Sionna RT applies as a single-layer P.2040 slab); a thickness of 5 m acts as a half
    space for every material used here. `roof` replaces the metal roofs; `ground_thick` thickens the ground slab."""
    props = {o.name: o.radio_material.name for o in sc.objects.values()}
    if wall is not None or thick is not None:
        m = rt.ITURadioMaterial("wall_p9", wall or "marble", thickness=0.1 if thick is None else float(thick))
        sc.add(m)
        for nm, mt in props.items():
            if mt == "marble":
                sc.objects[nm].radio_material = "wall_p9"
    if roof is not None:
        m = rt.ITURadioMaterial("roof_p9", roof, thickness=0.1 if thick is None else float(thick))
        sc.add(m)
        for nm, mt in props.items():
            if mt == "metal":
                sc.objects[nm].radio_material = "roof_p9"
    if ground_thick is not None:
        m = rt.ITURadioMaterial("ground_p9", "concrete", thickness=float(ground_thick))
        sc.add(m)
        for nm, mt in props.items():
            if mt in ("concrete", "terrain"):
                sc.objects[nm].radio_material = "ground_p9"


def run(city, n_users, seed=1, depth=3, samples=1e8, tag="", plane_off=30.0, cell=10.0, inner=100.0, pol="V",
        freq=2.0e9, scatter=None, reflect=True, cell_deg=None, cp=False, wall=None, thick=None, roof=None, ground_thick=None):
    out = os.path.join(HERE, "out", f"sky2_{city}{tag}.npz")
    sc = rt.load_scene(SCENES[city]); sc.frequency = freq
    set_facades(sc, wall, thick, roof, ground_thick)
    if scatter is not None:
        for m in sc.radio_materials.values():
            m.scattering_coefficient = scatter
    # the radio-map receiver is ideal isotropic and sums path powers (a local mean), so only the transmit (user)
    # polarization matters; --cp synthesizes circular polarization from a dual V/H element with weights [1, j]/sqrt2
    sc.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="VH" if cp else pol)
    import drjit as dr
    pvec = (dr.cuda.ad.TensorXf(np.array([[0.70710678, 0.0]], np.float32)),
            dr.cuda.ad.TensorXf(np.array([[0.0, 0.70710678]], np.float32))) if cp else None
    sc.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization=pol)
    bb = sc.mi_scene.bbox(); gs = ground_scene(sc); top = float(bb.max.z)
    bpts, bh = built_area(sc, gs, top)
    poly = hull_poly(bpts); H95 = float(np.percentile(bh, 95))
    lam = 299792458.0 / freq
    rng = np.random.default_rng(seed)
    D = dirs(); d = D.reshape(-1, 3).astype(np.float32); n = len(d)
    users, los, exc, dedge, pdist = [], [], [], [], []
    t0 = time.time()
    while len(users) < n_users:
        x0, y0 = rng.uniform(poly[:, 0].min(), poly[:, 0].max()), rng.uniform(poly[:, 1].min(), poly[:, 1].max())
        if inside_dist(poly, np.array([x0, y0])) < inner:
            continue
        p = np.array([x0, y0, float(ground_z(gs, [x0], [y0], top + 10.0)[0]) + 1.5])
        if bool(sc.mi_scene.ray_test(mi.Ray3f(mi.Point3f(*map(float, p)), mi.Vector3f(0.0, 0.0, 1.0)))):
            continue
        o = mi.Point3f(mi.Float(np.full(n, p[0], np.float32)), mi.Float(np.full(n, p[1], np.float32)), mi.Float(np.full(n, p[2], np.float32)))
        blk = np.array(sc.mi_scene.ray_test(mi.Ray3f(o, mi.Vector3f(mi.Float(d[:, 0]), mi.Float(d[:, 1]), mi.Float(d[:, 2])))))
        H = top + plane_off
        cs = (H - p[2]) * np.radians(cell_deg) if cell_deg else cell * (H - p[2]) / (top + 30.0 - p[2])
        L = 2.0 * (H - p[2]) / np.tan(np.radians(E_MIN)) + 4 * cs
        sc.add(rt.Transmitter("ue", position=[float(x) for x in p]))
        rm = rt.RadioMapSolver()(sc, center=[float(p[0]), float(p[1]), float(H)], orientation=[0.0, 0.0, 0.0],
                                 size=[float(L), float(L)], cell_size=[float(cs), float(cs)], samples_per_tx=int(samples), precoding_vec=pvec,
                                 max_depth=depth, los=True, specular_reflection=reflect, diffuse_reflection=scatter is not None,
                                 diffraction=True, edge_diffraction=True, refraction=False)
        sc.remove("ue")
        g = np.array(rm.path_gain).reshape(-1); cc = np.array(rm.cell_centers).reshape(-1, 3)
        ny, nx = np.array(rm.path_gain).shape[-2:]
        G = g.reshape(ny, nx); C = cc.reshape(ny, nx, 3)
        x0c, y0c = C[0, 0, 0], C[0, 0, 1]; dx = C[0, 1, 0] - C[0, 0, 0]; dy = C[1, 0, 1] - C[0, 0, 1]
        s = (H - p[2]) / d[:, 2]; hit = p[None, :] + s[:, None] * d
        ix = np.clip(np.rint((hit[:, 0] - x0c) / dx).astype(int), 0, nx - 1)
        iy = np.clip(np.rint((hit[:, 1] - y0c) / dy).astype(int), 0, ny - 1)
        ck = C[iy, ix]
        assert np.max(np.hypot(ck[:, 0] - hit[:, 0], ck[:, 1] - hit[:, 1])) <= 0.75 * cs, "cell lookup"
        dist = np.linalg.norm(ck - p[None, :], axis=1)
        fs = (lam / (4 * np.pi * dist)) ** 2
        exc.append((10 * np.log10(np.maximum(G[iy, ix], 1e-30) / fs)).reshape(D.shape[:2]))
        los.append((~blk).reshape(D.shape[:2]))
        dedge.append(dist_to_edge(poly, p[:2], AZS)); pdist.append(H - p[2])
        users.append(p)
        if len(users) % 20 == 0 or len(users) == n_users:
            print(f"{city}{tag}: {len(users)}/{n_users} users, {time.time() - t0:.0f} s", flush=True)
            np.savez_compressed(out, users=np.array(users), los=np.array(los), exc=np.array(exc), els=ELS, azs=AZS,
                                d_edge=np.array(dedge), H95=H95, plane_dist=np.array(pdist), hull=poly, built_h=bh,
                                params=json.dumps(dict(version=2, depth=depth, samples=samples, plane_off=plane_off, cell_deg=cell_deg,
                                                       cell=cell, inner=inner, pol=pol, cp=cp, freq=freq, scatter=scatter, reflect=reflect, seed=seed,
                                                       wall=wall, thick=thick, roof=roof, ground_thick=ground_thick)))
    print(city, tag, "done", time.time() - t0, flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("city"); ap.add_argument("n", type=int, nargs="?", default=300)
    ap.add_argument("--depth", type=int, default=3); ap.add_argument("--samples", type=float, default=1e8)
    ap.add_argument("--tag", default=""); ap.add_argument("--planeOff", type=float, default=30.0)
    ap.add_argument("--cell", type=float, default=10.0); ap.add_argument("--inner", type=float, default=100.0)
    ap.add_argument("--pol", default="V"); ap.add_argument("--freq", type=float, default=2.0e9)
    ap.add_argument("--scatter", type=float, default=None); ap.add_argument("--noreflect", action="store_true")
    ap.add_argument("--cellDeg", type=float, default=None); ap.add_argument("--cp", action="store_true")
    ap.add_argument("--wall", default=None); ap.add_argument("--thick", type=float, default=None)
    ap.add_argument("--roof", default=None); ap.add_argument("--groundThick", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    run(a.city, a.n, depth=a.depth, samples=a.samples, tag=a.tag, plane_off=a.planeOff, cell=a.cell, inner=a.inner,
        pol=a.pol, freq=a.freq, scatter=a.scatter, reflect=not a.noreflect, cell_deg=a.cellDeg, cp=a.cp, wall=a.wall,
        thick=a.thick, roof=a.roof, ground_thick=a.groundThick, seed=a.seed)
