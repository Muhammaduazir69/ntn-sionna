#!/usr/bin/env python3
"""C1: does the azimuth swing isolate a street effect? Writes out/street_null.json and out/axis_cf_<city>.npz.

Axes.  own: perpendicular of the user's own narrowest run (the rule of street_geom.py).  cf: cross-fitted, from
K outdoor points in an annulus around the user; their log free runs are averaged by azimuth and the user's own
rays are not used.  nn: own axis of the nearest other user.  Swing: LOS at 30 deg along (R<=15) minus across (R>=75).

Scenes.  real: the Sionna mesh LOS of the stored users, geometry from the 1 m raster.
rot: every connected block of the city rotated about its centroid by a random angle, so footprints, heights,
density and locations stay and street alignment goes.  iso: random rectangles, heights drawn from the city's own
heights, on flat ground, count set so that the mean LOS at 30 deg matches the city.
The same axis rules and statistics run on all scenes.  Limit 2 CPU threads."""
import json, os, sys
import numpy as np
import scipy.ndimage as ni
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
CITIES = ("munich", "etoile", "florence", "san_francisco")
AZ = np.arange(0, 360, 5.0); SA, CA = np.sin(np.radians(AZ)), np.cos(np.radians(AZ))
ELS = np.arange(15, 90, 5.0); TAN = np.tan(np.radians(ELS)); I30 = list(ELS).index(30.0)
RMAX_T, RMAX_S, HU = 500, 1000, 1.5
RT = np.arange(1, RMAX_T + 1, 1.0); RS = np.arange(1, RMAX_S + 1, 1.0)


class Field:
    def __init__(s, z, g, x0, y0, step=1.0):
        s.z, s.g, s.x0, s.y0, s.step = z, g, float(x0), float(y0), float(step); s.ny, s.nx = z.shape

    def at(s, x, y, arr=None):
        arr = s.z if arr is None else arr
        i = np.rint((y - s.y0) / s.step).astype(int); j = np.rint((x - s.x0) / s.step).astype(int)
        ok = (i >= 0) & (i < s.ny) & (j >= 0) & (j < s.nx)
        out = np.full(i.shape, -1e4, np.float32); out[ok] = arr[i[ok], j[ok]]
        return out

    def free(s, x, y, pz):
        """t(a): first distance where the top surface rises above the user's height, 72 azimuths; 500 = open."""
        h = s.at(x + np.outer(SA, RT), y + np.outer(CA, RT)) > pz
        return np.where(h.any(1), RT[np.argmax(h, 1)], float(RMAX_T))

    def skyline(s, x, y, pz):
        h = s.at(x + np.outer(SA, RS), y + np.outer(CA, RS))
        return np.degrees(np.arctan2(h - pz, RS[None, :])).max(1)


def rotated(F, rng, scale=1.0):
    b = (F.z - F.g) > 2.0; lab, n = ni.label(b); hgt = np.where(b, (F.z - F.g) * scale, 0.0)
    out = np.zeros_like(hgt); slices = ni.find_objects(lab)
    for k, sl in enumerate(slices, 1):
        yy, xx = np.nonzero(lab[sl] == k); yy = yy + sl[0].start; xx = xx + sl[1].start
        cy, cx = yy.mean(), xx.mean(); th = rng.uniform(0, 2 * np.pi); r = int(np.hypot(yy - cy, xx - cx).max()) + 2
        y0, y1 = max(int(cy) - r, 0), min(int(cy) + r + 1, F.ny); x0, x1 = max(int(cx) - r, 0), min(int(cx) + r + 1, F.nx)
        Y, X = np.mgrid[y0:y1, x0:x1]; dy, dx = Y - cy, X - cx
        sy = np.rint(cy + np.cos(th) * dy + np.sin(th) * dx).astype(int); sx = np.rint(cx - np.sin(th) * dy + np.cos(th) * dx).astype(int)
        ok = (sy >= 0) & (sy < F.ny) & (sx >= 0) & (sx < F.nx)
        m = np.zeros(Y.shape, bool); m[ok] = lab[sy[ok], sx[ok]] == k
        v = np.zeros(Y.shape, np.float32); v[m] = hgt[sy[m], sx[m]]
        out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], v)
    return Field(F.g + out, F.g, F.x0, F.y0, F.step)


def iso_scene(N, hs, rng, size=1100):
    H = np.zeros((size, size), np.float32); yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(N):
        cx, cy = rng.uniform(0, size, 2); a, b = rng.uniform(15, 45, 2); th = rng.uniform(0, np.pi); h = rng.choice(hs)
        r = int(np.hypot(a, b) / 2) + 1; x0, x1 = max(int(cx) - r, 0), min(int(cx) + r + 1, size); y0, y1 = max(int(cy) - r, 0), min(int(cy) + r + 1, size)
        X, Y = xx[y0:y1, x0:x1] - cx, yy[y0:y1, x0:x1] - cy
        u = X * np.cos(th) + Y * np.sin(th); v = -X * np.sin(th) + Y * np.cos(th)
        H[y0:y1, x0:x1][(np.abs(u) < a / 2) & (np.abs(v) < b / 2)] = h
    return Field(H, np.zeros_like(H), 0.0, 0.0, 1.0)


def inside_dist(poly, p):
    a = poly; e = np.roll(poly, -1, axis=0) - a
    nrm = np.stack([e[:, 1], -e[:, 0]], 1) / np.linalg.norm(e, axis=1)[:, None]
    return float(np.min(np.sum((a - p) * nrm, axis=1)))


def own_axis(t):
    return (AZ[np.argmin(t[:36] + t[36:])] + 90.0) % 180.0


def cf_axis(F, x, y, rng, K=12, r0=10.0, r1=40.0):
    """Cross-fitted axis: outdoor points in an annulus; the user's own rays are not used. None when too few points."""
    prof = np.zeros(36); got = 0; tries = 0
    while got < K and tries < 300:
        tries += 1; a = rng.uniform(0, 2 * np.pi); r = np.sqrt(rng.uniform(r0 ** 2, r1 ** 2)); px, py = x + r * np.sin(a), y + r * np.cos(a)
        g = float(F.at(np.array([px]), np.array([py]), F.g)[0]); z = float(F.at(np.array([px]), np.array([py]))[0])
        if z > g + 0.5 or z < -1e3:
            continue
        t = F.free(px, py, g + HU); prof += np.log(t[:36] + t[36:]); got += 1
    return (AZ[int(np.argmin(prof))] + 90.0) % 180.0 if got >= 4 else np.nan


def rel(ax):
    return np.abs(((AZ[None, :] - ax[:, None] + 90) % 180) - 90)


def swing(L, T, ax):
    R = rel(ax); ok = np.isfinite(ax)[:, None]
    al, ac = T & (R <= 15) & ok, T & (R >= 75) & ok
    return (L * al).sum() / max(al.sum(), 1) - (L * ac).sum() / max(ac.sum(), 1)


def harmonic(L, T):
    """Axis-free anisotropy per user: magnitude of the second azimuth harmonic of the LOS state over trusted azimuths."""
    w = T.astype(float); n = np.maximum(w.sum(1), 1); phi = np.radians(2 * AZ)
    m = (L * w).sum(1) / n; d = (L - m[:, None]) * w
    return np.abs((d * np.exp(1j * phi)[None, :]).sum(1)) / n


def boot(fn, n, rng, B=500):
    return np.array([fn(rng.integers(0, n, n)) for _ in range(B)])


def sample_users(F, poly, H95, rng, n=300, inner=100.0):
    xmin, xmax, ymin, ymax = poly[:, 0].min(), poly[:, 0].max(), poly[:, 1].min(), poly[:, 1].max()
    U, L, Tt, ax_own, ax_cf, ax_cf2, ax_long = [], [], [], [], [], [], []
    from sky_edge import dist_to_edge
    while len(U) < n:
        x, y = rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)
        if inside_dist(poly, np.array([x, y])) < inner:
            continue
        g = float(F.at(np.array([x]), np.array([y]), F.g)[0]); z = float(F.at(np.array([x]), np.array([y]))[0])
        if z > g + 0.5:
            continue
        pz = g + HU; t = F.free(x, y, pz); sky = F.skyline(x, y, pz)
        de = dist_to_edge(poly, np.array([x, y]), AZ)
        L.append(sky[None, :] < ELS[:, None]); Tt.append(de[None, :] * TAN[:, None] >= H95)
        ax_long.append(AZ[np.argmax(t[:36] + t[36:])]); ax_own.append(own_axis(t)); ax_cf.append(cf_axis(F, x, y, rng)); ax_cf2.append(cf_axis(F, x, y, rng, r0=20.0, r1=60.0)); U.append((x, y))
    return dict(U=np.array(U), L=np.array(L), T=np.array(Tt), own=np.array(ax_own), cf=np.array(ax_cf), cf2=np.array(ax_cf2), long=np.array(ax_long))


def curve(L, T, ax):
    """LOS at 30 deg in 6 relative-azimuth bins, for the azimuth figure."""
    R = rel(ax); ok = np.isfinite(ax)[:, None]; out = []
    for lo, hi in ((0, 15.1), (15.1, 30), (30, 45), (45, 60), (60, 74.9), (74.9, 91)):
        m = T & ok & (R >= lo) & (R < hi); out.append(float((L * m).sum() / max(m.sum(), 1)))
    return out


def agree(S):
    """The manuscript's street label: longest-run axis and wall-normal axis within 15 deg."""
    return np.abs(((S["long"] - S["own"] + 90) % 180) - 90) <= 15


def stats(S):
    L, T = S["L"][:, I30, :].astype(float), S["T"][:, I30, :]
    from scipy.spatial import cKDTree
    nn = cKDTree(S["U"]).query(S["U"], k=2)[1][:, 1]
    A2 = harmonic(L, T); ag = agree(S)
    return dict(curve_own=curve(L, T, S["own"]), curve_cf=curve(L, T, S["cf"]), los30=float((L * T).sum() / T.sum()), own=float(swing(L, T, S["own"])), cf=float(swing(L, T, S["cf"])), cf2=float(swing(L, T, S["cf2"])),
                nn=float(swing(L, T, S["own"][nn])), harm=float(A2.mean()), built=None, agree=float(ag.mean()),
                own_agree=float(swing(L[ag], T[ag], S["own"][ag])), own_non=float(swing(L[~ag], T[~ag], S["own"][~ag])),
                cf_agree=float(swing(L[ag], T[ag], S["cf"][ag])), cf_non=float(swing(L[~ag], T[~ag], S["cf"][~ag])))


def main(nrot=8, niso=8):
    res = {}
    for c in CITIES:
        r = np.load(f"{OUT}/raster_{c}.npz"); F = Field(r["z"], r["g"], r["x0"], r["y0"], float(r["step"]))
        z = np.load(f"{OUT}/sky2_{c}.npz"); poly = z["hull"]; H95 = float(z["H95"]); users = z["users"]; rng = np.random.default_rng(100)
        # real users: mesh LOS, raster geometry
        t = np.array([F.free(p[0], p[1], p[2]) for p in users])
        sky = np.array([F.skyline(p[0], p[1], p[2]) for p in users])
        T = z["d_edge"][:, None, :] * TAN[None, :, None] >= H95; Lm = z["los"]
        Lr = sky[:, None, :] < ELS[None, :, None]
        S = dict(U=users[:, :2], L=Lm, T=T, own=np.array([own_axis(x) for x in t]), long=np.array([AZ[np.argmax(x[:36] + x[36:])] for x in t]),
                 cf=np.array([cf_axis(F, p[0], p[1], rng) for p in users]), cf2=np.array([cf_axis(F, p[0], p[1], rng, r0=20.0, r1=60.0) for p in users]))
        st = np.load(f"{OUT}/street2_{c}.npz")
        out = dict(raster_vs_mesh=float((Lr == Lm)[T].mean()), real=stats(S), axis_own_mesh_agree=float((np.abs(((S["own"] - st["axis2_deg"] + 90) % 180) - 90) <= 5).mean()))
        np.savez_compressed(f"{OUT}/axis_cf_{c}.npz", own=S["own"], cf=S["cf"], cf2=S["cf2"], long=S["long"], t_raster=t)
        L30, T30 = Lm[:, I30, :].astype(float), T[:, I30, :]; n = len(users); rb = np.random.default_rng(5)
        from scipy.spatial import cKDTree
        nn = cKDTree(S["U"]).query(S["U"], k=2)[1][:, 1]; A2 = harmonic(L30, T30)
        BR = {k: boot(lambda ii: swing(L30[ii], T30[ii], ax[ii]), n, rb) for k, ax in (("own", S["own"]), ("cf", S["cf"]), ("cf2", S["cf2"]), ("nn", S["own"][nn]))}
        BR["harm"] = boot(lambda ii: A2[ii].mean(), n, rb)
        out["real_ci"] = {k: np.percentile(v, [2.5, 97.5]).tolist() for k, v in BR.items()}
        # rotated-block null: as they stand, and with heights scaled to match the mean LOS at 30 deg
        target = out["real"]["los30"]
        def rot_set(scale, name, nk):
            rs = []
            for k in range(nk):
                rg = np.random.default_rng(1000 + k); Fr = rotated(F, rg, scale); Sr = sample_users(Fr, poly, H95, rg); s = stats(Sr); s["built"] = float(((Fr.z - Fr.g) > 2).mean()); rs.append(s)
                print(c, name, k, round(scale, 3), {a: round(b, 3) for a, b in s.items() if isinstance(b, float)}, flush=True)
            return rs
        out["rot"] = rot_set(1.0, "rot", nrot); out["real_built"] = float(((F.z - F.g) > 2).mean())
        lo_, hi_ = 0.5, 3.0
        for it in range(6):
            sc_ = 0.5 * (lo_ + hi_); rg = np.random.default_rng(70 + it); val = stats(sample_users(rotated(F, rg, sc_), poly, H95, rg, n=100))["los30"]
            lo_, hi_ = (sc_, hi_) if val > target else (lo_, sc_)
        out["rot_scale"] = 0.5 * (lo_ + hi_); out["rotm"] = rot_set(out["rot_scale"], "rotm", nrot)
        # isotropic null matched on mean LOS at 30 deg (bisection on the rectangle count)
        hs = z["built_h"]; lo, hi = 200, 9000; poly2 = np.array([[100., 100.], [1000., 100.], [1000., 1000.], [100., 1000.]])
        for it in range(7):
            N = (lo + hi) // 2; rg = np.random.default_rng(50 + it); Fi = iso_scene(N, hs, rg); Si = sample_users(Fi, poly2, 0.0, rg, n=100)
            val = stats(Si)["los30"]
            lo, hi = (N, hi) if val > target else (lo, N)
        Nm = (lo + hi) // 2; isos = []
        for k in range(niso):
            rg = np.random.default_rng(2000 + k); Fi = iso_scene(Nm, hs, rg); Si = sample_users(Fi, poly2, 0.0, rg); s = stats(Si); s["built"] = float((Fi.z > 2).mean()); isos.append(s)
            print(c, "iso", k, Nm, {a: round(b, 3) for a, b in s.items() if isinstance(b, float)}, flush=True)
        out["iso"] = isos; out["iso_N"] = Nm
        # swing minus null, interval from the user bootstrap of the real scene and the spread of the null realizations
        rd = np.random.default_rng(8); out["diff"] = {}
        for nm in ("rotm", "iso", "rot"):
            for k in ("own", "cf", "cf2", "nn", "harm"):
                nv = np.array([s_[k] for s_ in out[nm]]); dd = BR[k] - rd.choice(nv, len(BR[k]))
                out["diff"][f"{nm}_{k}"] = dict(mean=float(out["real"][k] - nv.mean()), ci=np.percentile(dd, [2.5, 97.5]).tolist(), null=float(nv.mean()), null_sd=float(nv.std()))
        res[c] = out
        json.dump(res, open(f"{OUT}/street_null.json", "w"), indent=1)
        print(c, "real", out["real"], out["real_ci"], flush=True)


if __name__ == "__main__":
    main()
