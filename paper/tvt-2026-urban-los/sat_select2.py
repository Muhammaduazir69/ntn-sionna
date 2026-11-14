#!/usr/bin/env python3
"""R2: selection under realistic knowledge. Adds position error and height error of the map, a measured-RSRP rule (no map) with L3 filter and hysteresis,
the map used only to choose which three satellites to measure, and (R3) a marginal map rule that reads P(LOS) over the position posterior instead of the skyline at the reported position. Truth: LOS and loss from the sky maps. Reported position: true + N(0, sigma^2) per axis,
redrawn until outdoors; skyline from the 1 m height map. Writes out/sat_select2<tag>.json. Tags: _c03 (seed 1), _c03sd2 (seed 2)."""
import json, sys
import numpy as np
import street_null as sn, sat_select as ss
OUT = ss.OUT; SIG = (0, 3, 5, 10, 20); KPOST = 64; HPOST = 0.1


def field(c):
    r = np.load(f"{OUT}/raster_{c}.npz"); return sn.Field(r["z"], r["g"], r["x0"], r["y0"], float(r["step"]))


def eps_at(F, U, sig, rng, hscale=1.0, rep=None):
    Fz = F
    if hscale != 1.0:
        h = F.z - F.g; b = h > 2; Fz = sn.Field(F.g + np.where(b, h * hscale, 0), F.g, F.x0, F.y0, F.step)
    out = []
    for p in U:
        x, y = p[0], p[1]
        if sig > 0:
            for _ in range(200):
                dx, dy = rng.normal(0, sig, 2); x, y = p[0] + dx, p[1] + dy
                if float(F.at(np.array([x]), np.array([y]))[0]) <= float(F.at(np.array([x]), np.array([y]), F.g)[0]) + 0.5:
                    break
        if rep is not None:
            rep.append((x, y))
        out.append(Fz.skyline(x, y, float(F.at(np.array([x]), np.array([y]), F.g)[0]) + 1.5))
    return np.array(out)


def posterior(F, rep, sig, rng, K=KPOST):
    """Skylines of K outdoor draws from N(reported, sig^2) per axis: the position posterior of a rule that knows its error scale. Shape users x K x azimuths."""
    out = np.zeros((len(rep), K, 72), np.float32)
    for i, (rx, ry) in enumerate(rep):
        n = 0
        for _ in range(100 * K):
            x, y = rx + rng.normal(0, sig), ry + rng.normal(0, sig); g = float(F.at(np.array([x]), np.array([y]), F.g)[0]); z = float(F.at(np.array([x]), np.array([y]))[0])
            if z <= g + 0.5 and z > -1e3:
                out[i, n] = F.skyline(x, y, g + 1.5); n += 1
                if n == K:
                    break
        if n < K:
            out[i, n:] = out[i, :K - n]
    return out


def pos_err(F, U, sig, rng):
    out = []
    for p in U:
        x, y = p[0], p[1]
        if sig > 0:
            for _ in range(200):
                dx, dy = rng.normal(0, sig, 2); x, y = p[0] + dx, p[1] + dy
                if float(F.at(np.array([x]), np.array([y]))[0]) <= float(F.at(np.array([x]), np.array([y]), F.g)[0]) + 0.5:
                    break
        out.append((x, y))
    return np.array(out)


def road_table(c):
    """P(LOS | elevation row, azimuth from road axis), first realization of the other three cities."""
    num = np.zeros((15, 19)); den = np.zeros((15, 19))
    for o in ss.CITIES:
        if o == c:
            continue
        z = np.load(f"{OUT}/sky2_{o}.npz"); ax = np.load(f"{OUT}/axes2_{o}.npz")["road"]; k = np.rint(sn.rel(ax) / 5).astype(int)
        for i in range(15):
            np.add.at(num[i], k.ravel(), z["los"][:, i, :].ravel().astype(float)); np.add.at(den[i], k.ravel(), 1.0)
    return num / np.maximum(den, 1)


def run_shell(c, lat, planes, per, tag, EP, road, MP, keep=None):
    # one direction in sky2_florence_c03s is NaN: count it as unreached, since np.argmin and np.percentile would otherwise take it up. keep: optional dict that receives per-user arrays for paired statistics.
    Sat, T = ss.shell(planes, per); z = np.load(f"{OUT}/sky2_{c}{tag}.npz"); los, L = z["los"], np.nan_to_num(-z["exc"], nan=ss.UNR, posinf=np.inf, neginf=-np.inf); nu = los.shape[0]; ns = Sat.shape[0]
    el, az, rg = ss.topo(Sat, T, lat); vis = el >= 15; ie = np.clip(np.rint((el - 15) / 5).astype(int), 0, 14); ia = np.rint(az / 5).astype(int) % 72; fs = 20 * np.log10(4 * np.pi * rg / ss.LAM)
    kinds = ["HE", "ORA"] + list(EP) + list(road["ax"]) + ["RSRP6h3", "RSRP6h6", "RSRP3h3", "K3elev", "K3map10"] + list(MP) + [k + "h" for k in MP] + ["RSRP6h0", "K3marg10"]
    LOS = {k: np.zeros((nu, len(T))) for k in kinds}; TOT = {k: np.zeros((nu, len(T))) for k in kinds}; ID = {k: -np.ones((nu, len(T)), int) for k in kinds}
    rn2 = np.random.default_rng(14); rn3 = np.random.default_rng(15); S0 = dict(f=np.full((nu, ns), np.nan)); SK = dict(f=np.full((nu, ns), np.nan), s=-np.ones(nu, int)); SM = {k: -np.ones(nu, int) for k in MP}; rn = np.random.default_rng(13); ST = {k: dict(f=np.full((nu, ns), np.nan), s=-np.ones(nu, int)) for k in ("RSRP6h3", "RSRP6h6", "RSRP3h3", "K3elev", "K3map10")}; PAR = dict(RSRP6h3=(6., 3.), RSRP6h6=(6., 6.), RSRP3h3=(3., 3.), K3elev=(6., 3.), K3map10=(6., 3.)); good = np.zeros(len(T), bool)
    for t in range(len(T)):
        v = np.flatnonzero(vis[:, t])
        if len(v) == 0:
            continue
        good[t] = True; E, A = ie[v, t], ia[v, t]; e_ = el[v, t]; LL = los[:, E, A]; XL = np.where(LL, 0.0, np.minimum(L[:, E, A], ss.UNR)); tot = fs[v, t][None, :] + XL; order = np.argsort(-e_)
        def rec(k, s):
            LOS[k][:, t] = LL[np.arange(nu), s]; TOT[k][:, t] = tot[np.arange(nu), s]; ID[k][:, t] = v[s]
        rec("HE", np.full(nu, int(np.argmax(e_)))); rec("ORA", np.argmin(tot, 1))
        for k, eps in EP.items():
            M = eps[:, A] < e_[None, :]; first = np.argmax(M[:, order], 1); rec(k, np.where(M.any(1), order[first], order[0]))
        for k, ax in road["ax"].items():
            R = np.abs(((az[v, t][None, :] - ax[:, None] + 90) % 180) - 90); pr = road["tab"][E[None, :].repeat(nu, 0), np.rint(R / 5).astype(int)]; rec(k, np.argmax(pr + 1e-6 * e_[None, :], 1))
        for k, st in ST.items():
            sg, hy = PAR[k]; m = tot + rn.normal(0, sg, tot.shape); old = st["f"][:, v]; new = np.full((nu, ns), np.nan); new[:, v] = np.where(np.isnan(old), m, 0.5 * old + 0.5 * m); st["f"] = new; f = new[:, v].copy()
            if k.startswith("K3"):
                # only three candidates are measured for each user: highest three, or highest three clear on the 10 m map
                allow = np.zeros_like(f, bool); pri = np.tile(-e_, (nu, 1))
                if k == "K3map10":
                    pri = pri - 1000.0 * (EP["sig10"][:, A] < e_[None, :])
                idx = np.argsort(pri, 1)[:, :3]; allow[np.arange(nu)[:, None], idx] = True; pos0 = -np.ones(ns, int); pos0[v] = np.arange(len(v)); q0 = np.where(st["s"] >= 0, pos0[np.maximum(st["s"], 0)], -1)
                allow[q0 >= 0, q0[q0 >= 0]] = True; f[~allow] = np.inf
            best = np.argmin(f, 1); pos = -np.ones(ns, int); pos[v] = np.arange(len(v)); q = np.where(st["s"] >= 0, pos[np.maximum(st["s"], 0)], -1); sv = f[np.arange(nu), np.maximum(q, 0)]; sw = (q < 0) | (f[np.arange(nu), best] < sv - hy); ch = np.where(sw, best, q); rec(k, ch); st["s"] = v[ch]
        pos_ = -np.ones(ns, int); pos_[v] = np.arange(len(v)); rw = np.arange(nu); PR = {}
        for k, S_ in MP.items():
            Pl = (S_[:, :, A] < e_[None, None, :]).mean(1); PR[k] = Pl; rec(k, np.argmax(Pl + 1e-6 * e_[None, :], 1))
            # same rule with a margin: stay on the serving satellite unless another is more likely clear by HPOST
            best = np.argmax(Pl + 1e-6 * e_[None, :], 1); q = np.where(SM[k] >= 0, pos_[np.maximum(SM[k], 0)], -1); sv = Pl[rw, np.maximum(q, 0)]; ch = np.where((q < 0) | (Pl[rw, best] > sv + HPOST), best, q); rec(k + "h", ch); SM[k] = v[ch]
        # measured rule with no margin, so handovers compare like with like; its noise stream is separate so the earlier rows do not move
        m = tot + rn2.normal(0, 6.0, tot.shape); old = S0["f"][:, v]; new = np.full((nu, ns), np.nan); new[:, v] = np.where(np.isnan(old), m, 0.5 * old + 0.5 * m); S0["f"] = new; rec("RSRP6h0", np.argmin(new[:, v], 1))
        # three candidates to measure chosen by the marginal rule at 10 m
        if "marg10" in PR:
            m = tot + rn3.normal(0, 6.0, tot.shape); old = SK["f"][:, v]; new = np.full((nu, ns), np.nan); new[:, v] = np.where(np.isnan(old), m, 0.5 * old + 0.5 * m); SK["f"] = new; f = new[:, v].copy()
            allow = np.zeros_like(f, bool); idx = np.argsort(-(PR["marg10"] + 1e-6 * e_[None, :]), 1)[:, :3]; allow[rw[:, None], idx] = True; q = np.where(SK["s"] >= 0, pos_[np.maximum(SK["s"], 0)], -1); allow[q >= 0, q[q >= 0]] = True; f[~allow] = np.inf
            best = np.argmin(f, 1); sv = f[rw, np.maximum(q, 0)]; ch = np.where((q < 0) | (f[rw, best] < sv - 6.0), best, q); rec("K3marg10", ch); SK["s"] = v[ch]
    b = np.random.default_rng(9); r = {}; dtm = (T[1] - T[0]) / 60.0
    for k in kinds:
        l = LOS[k][:, good]; to = TOT[k][:, good]; ho = (np.diff(ID[k][:, good], axis=1) != 0).sum(1) / (good.sum() * dtm)
        r[k] = dict(los=float(l.mean()), ci=np.percentile([l[b.integers(0, nu, nu)].mean() for _ in range(300)], [2.5, 97.5]).tolist(), p50=float(np.median(to)), p90=float(np.percentile(to, 90)), ho=float(ho.mean()))
        if keep is not None:
            keep[k] = dict(los=l.mean(1), tot=to, id=ID[k][:, good], fs=fs[:, good], T=T)
    return r


def main(tag):
    res = {}
    for c, lat in ss.CITIES.items():
        F = field(c); U = np.load(f"{OUT}/sky2_{c}{tag}.npz")["users"]; rng = np.random.default_rng(42); REP = {s: [] for s in SIG}; EP = {f"sig{s}": eps_at(F, U, s, rng, rep=REP[s]) for s in SIG}
        EP["sig0_hx07"] = eps_at(F, U, 0, rng, 0.7); EP["sig0_hx13"] = eps_at(F, U, 0, rng, 1.3)
        from street_null2 import road_axes
        road = dict(tab=road_table(c), ax={f"AZroad{s}": road_axes(c, pos_err(F, U, s, rng))[0] for s in (0, 10, 20)})
        rp = np.random.default_rng(4242); MP = {f"marg{s}": posterior(F, REP[s], s, rp) for s in SIG if s > 0}
        # wrong error scale: reported positions at 10 m, rule believes 5 m or 20 m
        MP["marg10as5"] = posterior(F, REP[10], 5, rp); MP["marg10as20"] = posterior(F, REP[10], 20, rp)
        res[c] = dict(dense=run_shell(c, lat, 72, 22, tag, EP, road, MP), sparse=run_shell(c, lat, 24, 11, tag, EP, road, MP))
        print(c, tag, {k: (round(v["los"], 3), round(v["p90"], 1), round(v["ho"], 2)) for k, v in res[c]["dense"].items()}, flush=True)
        print("   sparse", {k: round(v["los"], 3) for k, v in res[c]["sparse"].items()}, flush=True)
        json.dump(res, open(f"{OUT}/sat_select2{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
