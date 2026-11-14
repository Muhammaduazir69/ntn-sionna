#!/usr/bin/env python3
"""M8: consequence on a LEO shell, from the stored sky maps (no new field solve). Static users of each city see a Walker
delta shell (two-body circular orbits, Earth rotation); each snapshot looks up LOS state and excess loss in the user's
sky map at the nearest grid direction. Directions above 85 deg use the 85 deg row; below 15 deg are not visible.
Scene y is taken as north. Writes out/sat_select<tag>.json.

Policies: HE highest elevation; HE-S sticky, keep until set below 15 deg; AZ-own, AZ-cf highest LOS probability from a
table of (elevation, azimuth from the axis) built on the other three cities, with the user's own or cross-fitted axis;
FOOT highest F_H(h + t(a) tan e), the user's own facade distances with heights from the other three cities;
MAP highest elevation above the skyline that the user's height map gives (not the mesh test), else HE; MAP-S same, sticky while clear; ORA smallest total ray-traced
path loss. TR: LOS drawn independently per satellite from the TR 38.811 table."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
CITIES = {"munich": 48.14, "etoile": 48.87, "florence": 43.77, "san_francisco": 37.77}
RE, H, MU, WE = 6371e3, 550e3, 3.986004418e14, 7.2921159e-5
TR_EL = np.arange(10, 91, 10.0)
TR_LOS = {"dense_urban": np.array([28.2, 33.1, 39.8, 46.8, 53.7, 61.2, 73.8, 82.0, 98.1]) / 100,
          "urban": np.array([24.6, 38.6, 49.3, 61.3, 72.6, 80.5, 91.9, 96.8, 99.2]) / 100}
FREQ = 2e9; LAM = 299792458.0 / FREQ; UNR = 150.0


def shell(planes, per, inc=53.0, F=1, dt=10.0, steps=360, t0=0.0, alt=H):
    a = RE + alt; n = np.sqrt(MU / a ** 3); i = np.radians(inc); T = t0 + dt * np.arange(steps)
    raan = 2 * np.pi * np.arange(planes) / planes; out = []
    for p in range(planes):
        for s in range(per):
            u = n * T + 2 * np.pi * (s / per + F * p / (planes * per))
            x, y = np.cos(u), np.sin(u) * np.cos(i); z = np.sin(u) * np.sin(i)
            X = a * (x * np.cos(raan[p]) - y * np.sin(raan[p])); Y = a * (x * np.sin(raan[p]) + y * np.cos(raan[p])); out.append(np.stack([X, Y, a * z], 1))
    return np.array(out), T                 # [sat, time, 3], inertial


def topo(S, T, lat):
    """Elevation (deg), azimuth (deg from north, clockwise), range (m) from a ground point; Earth rotation at WE."""
    la = np.radians(lat); th = WE * T
    g = RE * np.stack([np.cos(la) * np.cos(th), np.cos(la) * np.sin(th), np.full_like(th, np.sin(la))], 1)
    r = S - g[None, :, :]; rng_ = np.linalg.norm(r, axis=2)
    east = np.stack([-np.sin(th), np.cos(th), np.zeros_like(th)], 1); up = g / RE; north = np.cross(up, east)
    E, N, U = (r * east[None]).sum(2), (r * north[None]).sum(2), (r * up[None]).sum(2)
    return np.degrees(np.arcsin(U / rng_)), np.degrees(np.arctan2(E, N)) % 360, rng_


def run(tag="", planes=72, per=22, label="dense"):
    S, T = shell(planes, per); data = {}
    for c in CITIES:
        z = np.load(f"{OUT}/sky2_{c}{tag}.npz"); st = np.load(f"{OUT}/street2_{c}.npz"); cf = np.load(f"{OUT}/axis_cf_{c}.npz")
        t_, H_ = st["free"], st["hfac"]; e1 = np.where(t_ < 500, np.degrees(np.arctan2(H_ - 1.5, t_)), -90.0); eps = np.maximum(e1, st["sky"])
        data[c] = dict(los=z["los"], L=-z["exc"], own=st["axis2_deg"], cf=cf["cf"], eps=eps, t=t_, H=H_)
    # P(LOS | elevation row, relative azimuth bin), each from all cities; applied leave one city out
    def table(cs, key):
        num = np.zeros((15, 19)); den = np.zeros((15, 19))
        for c in cs:
            D = data[c]; ax = D[key]; ok = np.isfinite(ax)
            R = np.abs(((np.arange(0, 360, 5.0)[None, :] - ax[ok][:, None] + 90) % 180) - 90)
            k = np.rint(R / 5).astype(int)
            for e in range(15):
                np.add.at(num[e], k.ravel(), D["los"][ok][:, e, :].ravel().astype(float)); np.add.at(den[e], k.ravel(), 1.0)
        return num / np.maximum(den, 1)
    res = {}
    for c, lat in CITIES.items():
        D = data[c]; el, az, rg = topo(S, T, lat); los, L = D["los"], D["L"]; nu = los.shape[0]
        vis = el >= 15.0
        ie = np.clip(np.rint((el - 15) / 5).astype(int), 0, 14); ia = np.rint(az / 5).astype(int) % 72
        fs = 20 * np.log10(4 * np.pi * rg / LAM)
        tabs = {k: table([x for x in CITIES if x != c], k) for k in ("own", "cf")}
        rngtr = np.random.default_rng(3)
        hs_o = np.sort(np.concatenate([data[x]["H"][data[x]["t"] < 500] for x in CITIES if x != c]))
        rows = {k: np.zeros((nu, len(T))) for k in ("HE", "HES", "AZown", "AZcf", "FOOT", "MAP", "MAPS", "ORA")}
        lossr = {k: np.zeros((nu, len(T))) for k in rows}; ids = {k: -np.ones((nu, len(T)), int) for k in rows}
        any_los = np.zeros((nu, len(T)), bool); nvis = np.zeros(len(T)); trnone = {k: np.zeros((nu, len(T)), bool) for k in TR_LOS}
        for t in range(len(T)):
            v = np.flatnonzero(vis[:, t]); nvis[t] = len(v)
            if len(v) == 0:
                for k in rows:
                    lossr[k][:, t] = np.nan
                continue
            E, A, F_ = ie[v, t], ia[v, t], fs[v, t]; he = v[np.argmax(el[v, t])]
            LL = los[:, E, A]; XL = np.where(LL, 0.0, np.minimum(L[:, E, A], UNR))    # [user, vis]
            tot = F_[None, :] + XL; any_los[:, t] = LL.any(1)
            for k, tb in TR_LOS.items():
                trnone[k][:, t] = ~(rngtr.random((nu, len(v))) < np.interp(el[v, t], TR_EL, tb)).any(1)
            order = np.argsort(-el[v, t]); MPL = D["eps"][:, A] < el[v, t][None, :]; first = np.argmax(MPL[:, order], 1); clear = MPL.any(1)
            mp = np.where(clear, order[first], order[0])
            ttv = D["t"][:, A]; pf = np.where(ttv >= 500, 1.0, np.searchsorted(hs_o, 1.5 + ttv * np.tan(np.radians(el[v, t]))[None, :]) / len(hs_o))
            sel = dict(HE=np.full(nu, np.argmax(el[v, t])), MAP=mp, ORA=np.argmin(tot, 1), FOOT=np.argmax(pf + 1e-6 * el[v, t][None, :], 1))
            for kk, tb in (("AZown", tabs["own"]), ("AZcf", tabs["cf"])):
                ax = D["own" if kk == "AZown" else "cf"]; R = np.abs(((az[v, t][None, :] - np.nan_to_num(ax)[:, None] + 90) % 180) - 90)
                pr = tb[E[None, :].repeat(nu, 0), np.rint(R / 5).astype(int)]
                sel[kk] = np.argmax(pr + 1e-6 * el[v, t][None, :], 1)
            # sticky policies keep the previous satellite while allowed
            for sk, base, keep_clear in (("HES", "HE", False), ("MAPS", "MAP", True)):
                prev = ids[sk][:, t - 1] if t else -np.ones(nu, int); s_ = sel[base].copy(); pos = {sat: i for i, sat in enumerate(v)}
                for u in range(nu):
                    q = pos.get(prev[u], -1)
                    if q >= 0 and (not keep_clear or MPL[u, q] or not clear[u]):
                        s_[u] = q
                sel[sk] = s_
            for k, s_ in sel.items():
                rows[k][:, t] = LL[np.arange(nu), s_]; lossr[k][:, t] = tot[np.arange(nu), s_]; ids[k][:, t] = v[s_]
        ok = nvis > 0; dt_min = (T[1] - T[0]) / 60.0
        boot = np.random.default_rng(9)
        def ci(fn):
            return np.percentile([fn(boot.integers(0, nu, nu)) for _ in range(400)], [2.5, 97.5]).tolist()
        r = dict(mean_visible=float(nvis.mean()), any_los=float(any_los[:, ok].mean()), any_los_ci=ci(lambda i: any_los[i][:, ok].mean()),
                 tr_any={k: float(1 - v[:, ok].mean()) for k, v in trnone.items()}, policies={})
        base = lossr["HE"][:, ok]
        for k in rows:
            lo = lossr[k][:, ok]; hos = (np.diff(ids[k][:, ok], axis=1) != 0).sum(1) / (ok.sum() * dt_min)
            r["policies"][k] = dict(los=float(rows[k][:, ok].mean()), los_ci=ci(lambda i: rows[k][i][:, ok].mean()), p50=float(np.median(lo)), p90=float(np.percentile(lo, 90)),
                                    mean_gain=float((base - lo).mean()), gain_ci=ci(lambda i: (base[i] - lo[i]).mean()), ho_per_min=float(hos.mean()))
        res[c] = r
        print(c, label, "vis %.1f any %.3f" % (r["mean_visible"], r["any_los"]), "TR", {k: round(v, 3) for k, v in r["tr_any"].items()},
              {k: (round(v["los"], 3), round(v["p90"], 1), round(v["ho_per_min"], 2)) for k, v in r["policies"].items()}, flush=True)
    return res


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    out = dict(dense=run(tag), sparse=run(tag, 24, 11, "sparse"))
    json.dump(out, open(f"{OUT}/sat_select{tag}.json", "w"), indent=1)
