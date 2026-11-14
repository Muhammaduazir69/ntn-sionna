#!/usr/bin/env python3
"""R1, R5, R6: controls with more axis rules, four elevations, street-labeled users, both user realizations, Holm. Writes out/street_null2.json.
Axes: own, cf (annulus 10-40 m), cf2 (20-60 m), nn, st (structure tensor of the building mask, sigma 20 m, no user ray), road (nearest OSM centerline, any kind; drive = drivable).
Controls: rotm (blocks rotated, heights scaled; road axis is the pre-rotation one), iso (rectangles; no road axis). 8 realizations. At most 2 threads."""
import json, os
import numpy as np
import scipy.ndimage as ni
from scipy.spatial import cKDTree
import street_null as sn
OUT = sn.OUT; ELS = sn.ELS; EL = {20.0: list(ELS).index(20.0), 30.0: list(ELS).index(30.0), 45.0: list(ELS).index(45.0), 60.0: list(ELS).index(60.0)}
N = json.load(open(f"{OUT}/street_null.json")); KEYS = ("own", "cf", "cf2", "nn", "st", "road", "drive"); BOOT = 5000


def st_axis(F, U, sig=20.0):
    b = ((F.z - F.g) > 2).astype(np.float32)
    gy = ni.gaussian_filter(b, 2, order=(1, 0)); gx = ni.gaussian_filter(b, 2, order=(0, 1))
    Jxx = ni.gaussian_filter(gx * gx, sig); Jyy = ni.gaussian_filter(gy * gy, sig); Jxy = ni.gaussian_filter(gx * gy, sig)
    th = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)
    i = np.clip(np.rint((U[:, 1] - F.y0) / F.step).astype(int), 0, F.ny - 1); j = np.clip(np.rint((U[:, 0] - F.x0) / F.step).astype(int), 0, F.nx - 1)
    nrm = th[i, j]; az = np.degrees(np.arctan2(np.cos(nrm), np.sin(nrm)))
    return np.round(((az + 90) % 180) / 5) * 5 % 180


def road_axes(c, U):
    r = np.load(f"{OUT}/osm_roads_{c}.npz"); mid, ang, dr = r["mid"], r["ang"], r["drive"]
    d1, i1 = cKDTree(mid).query(U); d2, i2 = cKDTree(mid[dr]).query(U)
    return np.rint(ang[i1] / 5) * 5 % 180, np.rint(ang[dr][i2] / 5) * 5 % 180, d1, d2


def swings(L, T, axes, sel=None):
    out = {}
    for k, ax in axes.items():
        out[k] = {}
        for e, i in EL.items():
            l_, t_, a_ = L[:, i, :].astype(float), T[:, i, :], ax
            if sel is not None:
                l_, t_, a_ = l_[sel], t_[sel], a_[sel]
            out[k][str(e)] = float(sn.swing(l_, t_, a_))
    return out


def build(F, S, c, road=True):
    U = S["U"]; nn = cKDTree(U).query(U, k=2)[1][:, 1]
    ax = dict(own=S["own"], cf=S["cf"], cf2=S["cf2"], nn=S["own"][nn], st=st_axis(F, U))
    if road:
        ax["road"], ax["drive"], dd, _ = road_axes(c, U); S["dist"] = dd
    return ax


def strata(L, T, road, own, dist):
    """Road-axis swing inside strata of own-wall agreement and of distance to the centerline, for the reweighted control (m4)."""
    w = np.abs(((road - own + 90) % 180) - 90); o = {}
    for nm, v, bins in (("wall", w, ((-1, 15), (15, 30), (30, 91))), ("dist", dist, ((-1, 5), (5, 15), (15, 1e9)))):
        o[nm] = []
        for lo, hi in bins:
            m = (v > lo) & (v <= hi)
            o[nm].append(dict(share=float(m.mean()), swing=float(sn.swing(L[m], T[m], road[m])) if m.sum() >= 5 else None))
    return o


def boot_swing(l_, t_, a_, idx, B, rb):
    """sn.swing over B resamples of users, vectorized: same estimator as the point value."""
    R = sn.rel(a_); ok = np.isfinite(a_)[:, None]; al, ac = t_ & (R <= 15) & ok, t_ & (R >= 75) & ok
    Sa, Na, Sc, Nc = (l_ * al).sum(1), al.sum(1), (l_ * ac).sum(1), ac.sum(1); ii = rb.choice(idx, (B, len(idx)))
    return Sa[ii].sum(1) / np.maximum(Na[ii].sum(1), 1) - Sc[ii].sum(1) / np.maximum(Nc[ii].sum(1), 1)


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0.0
    for r_, i in enumerate(o):
        run = max(run, (m - r_) * ps[i]); adj[i] = min(run, 1.0)
    return adj.tolist()


def real_set(F, c, tag, H95, seed):
    z = np.load(f"{OUT}/sky2_{c}{tag}.npz"); U = z["users"]; rng = np.random.default_rng(seed)
    T = z["d_edge"][:, None, :] * sn.TAN[None, :, None] >= H95; t = np.array([F.free(p[0], p[1], p[2]) for p in U])
    S = dict(U=U[:, :2], L=z["los"], T=T, own=np.array([sn.own_axis(x) for x in t]), long=np.array([sn.AZ[np.argmax(x[:36] + x[36:])] for x in t]),
             cf=np.array([sn.cf_axis(F, p[0], p[1], rng) for p in U]), cf2=np.array([sn.cf_axis(F, p[0], p[1], rng, r0=20.0, r1=60.0) for p in U]))
    return S


def main(nk=8):
    res = {}
    for c in sn.CITIES:
        r = np.load(f"{OUT}/raster_{c}.npz"); F = sn.Field(r["z"], r["g"], r["x0"], r["y0"], float(r["step"])); z = np.load(f"{OUT}/sky2_{c}.npz"); poly = z["hull"]; H95 = float(z["H95"]); o = {}
        ctl = dict(rotm=[], iso=[])
        for k in range(nk):
            rg = np.random.default_rng(3000 + k); Fr = sn.rotated(F, rg, N[c]["rot_scale"]); Sr = sn.sample_users(Fr, poly, H95, rg); ax = build(F if False else Fr, Sr, c)
            ag = sn.agree(Sr); ctl["rotm"].append(dict(all=swings(Sr["L"], Sr["T"], ax), street=swings(Sr["L"], Sr["T"], ax, ag), agree=float(ag.mean()), strata=strata(Sr["L"][:, EL[30.0], :].astype(float), Sr["T"][:, EL[30.0], :], ax["road"], Sr["own"], Sr["dist"]), curve=dict(own=sn.curve(Sr["L"][:, EL[30.0], :].astype(float), Sr["T"][:, EL[30.0], :], Sr["own"]), road=sn.curve(Sr["L"][:, EL[30.0], :].astype(float), Sr["T"][:, EL[30.0], :], ax["road"]), st=sn.curve(Sr["L"][:, EL[30.0], :].astype(float), Sr["T"][:, EL[30.0], :], ax["st"]))))
            rg = np.random.default_rng(4000 + k); poly2 = np.array([[100., 100.], [1000., 100.], [1000., 1000.], [100., 1000.]]); Fi = sn.iso_scene(N[c]["iso_N"], z["built_h"], rg); Si = sn.sample_users(Fi, poly2, 0.0, rg); ax = build(Fi, Si, c, road=False)
            ag = sn.agree(Si); ctl["iso"].append(dict(all=swings(Si["L"], Si["T"], ax), street=swings(Si["L"], Si["T"], ax, ag), agree=float(ag.mean()), curve=dict(own=sn.curve(Si["L"][:, EL[30.0], :].astype(float), Si["T"][:, EL[30.0], :], Si["own"]), st=sn.curve(Si["L"][:, EL[30.0], :].astype(float), Si["T"][:, EL[30.0], :], ax["st"]))))
            print(c, "control", k, flush=True)
        o["ctl"] = ctl
        for tag, name, sd in (("", "seed1", 5), ("_c03sd2", "seed2", 6)):
            S = real_set(F, c, tag, H95, sd); ax = build(F, S, c); ag = sn.agree(S); rb = np.random.default_rng(sd); n = len(S["U"]); B = BOOT; rr = dict(agree=float(ag.mean()), road_dist_med=float(np.median(S["dist"])))
            S_ = {}
            for pop, sel in (("all", None), ("street", ag), ("near", S["dist"] <= 8.0)):
                pts_ = swings(S["L"], S["T"], ax, sel); idx = np.arange(n) if sel is None else np.flatnonzero(sel); bo = {}
                for k_, a_ in ax.items():
                    bo[k_] = {}
                    for e, i in EL.items():
                        l_, t_ = S["L"][:, i, :].astype(float), S["T"][:, i, :]
                        bo[k_][str(e)] = boot_swing(l_, t_, a_, idx, B, rb)
                ex = {}
                for cn, cl in ctl.items():
                    pk = "all" if pop in ("all", "near") else "street"
                    for k_ in ax:
                        if k_ not in cl[0][pk]:
                            continue
                        for e in bo[k_]:
                            nv = np.array([x[pk][k_][e] for x in cl]); dd = bo[k_][e] - rb.choice(nv, B)
                            p = min(1.0, 2 * min((1 + (dd <= 0).sum()) / (B + 1), (1 + (dd >= 0).sum()) / (B + 1)))  # add-one, two-sided
                            ex[f"{cn}|{k_}|{e}"] = dict(mean=float(pts_[k_][e] - nv.mean()), ci=np.percentile(dd, [2.5, 97.5]).tolist(), p=float(p), null=float(nv.mean()), sd=float(nv.std()))
                S_[pop] = dict(swing=pts_, ci={k_: {e: np.percentile(v_, [2.5, 97.5]).tolist() for e, v_ in bo[k_].items()} for k_ in bo}, excess=ex, n=len(idx))
            L30, T30 = S["L"][:, EL[30.0], :].astype(float), S["T"][:, EL[30.0], :]
            rr["curve"] = {k_: sn.curve(L30, T30, a_) for k_, a_ in ax.items()}
            rr["strata"] = strata(L30, T30, ax["road"], S["own"], S["dist"]); rr["boot_B"] = B
            rr.update(S_); o[name] = rr
            np.savez_compressed(f"{OUT}/axes2_{c}{tag}.npz", **{k_: v_ for k_, v_ in ax.items()}, long=S["long"], dist=S["dist"])
            print(c, name, "all", {k_: round(rr["all"]["swing"][k_]["30.0"], 3) for k_ in ax}, "street", {k_: round(rr["street"]["swing"][k_]["30.0"], 3) for k_ in ax}, flush=True)
        res[c] = o
        # Holm over the 30 deg comparisons of each axis-population family, per realization
        for name in ("seed1", "seed2"):
            for pop in ("all", "street", "near"):
                for k_ in KEYS:
                    keys = [(c_, q) for c_ in res for q in res[c_][name][pop]["excess"] if q.split("|")[1] == k_ and q.endswith("|30.0")]
                    if keys:
                        adj = holm([res[c_][name][pop]["excess"][q]["p"] for c_, q in keys])
                        for (c_, q), a_ in zip(keys, adj):
                            res[c_][name][pop]["excess"][q]["p_holm"] = a_
        json.dump(res, open(f"{OUT}/street_null2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
