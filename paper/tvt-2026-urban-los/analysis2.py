#!/usr/bin/env python3
"""Analysis of the version-2 sky maps (sky_maps2.py, street_geom.py ... sky2 street2); writes out/analysis2.json.

Trusted pairs: a (user, elevation, azimuth) is used when d_edge * tan(e) >= H95, so the ray clears almost all
building heights before it leaves the modeled city. Users are the independent units; intervals are cluster
bootstrap over users. Unreached blocked directions are censored (loss treated as +inf for medians and quantiles).
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
CITIES = ("munich", "etoile", "florence", "san_francisco")
H_U, LAM = 1.5, 299792458.0 / 2e9
TR_EL = np.arange(10, 91, 10.0)
TR_LOS = {"dense_urban": np.array([28.2, 33.1, 39.8, 46.8, 53.7, 61.2, 73.8, 82.0, 98.1]) / 100,
          "urban": np.array([24.6, 38.6, 49.3, 61.3, 72.6, 80.5, 91.9, 96.8, 99.2]) / 100}
TR_CL_S = np.array([34.3, 30.9, 29.0, 27.7, 26.8, 26.2, 25.8, 25.5, 25.5])   # S band, NLOS, both tables
TR_SF_S = {"dense_urban": np.array([15.5, 13.9, 12.4, 11.7, 10.6, 10.5, 10.1, 9.2, 9.2]),   # NLOS sigma (dB)
           "urban": np.full(9, 6.0)}
STATS = json.load(open(os.path.join(OUT, "city_stats.json")))


def p681_los(t_perp, e, rel, hb, hm=1.5):
    """ITU-R P.681-11 Sec. 4.2 roadside building model with zero Fresnel clearance (Cf = 0):
    P(block) = exp(-h1^2 / (2 hb^2)), h1 = hm + dm tan(e) / sin(phi); hb the modal building height."""
    # the Recommendation excludes the limiting azimuths; clamp to half the 5-degree grid step
    s = np.maximum(np.abs(np.sin(np.radians(rel))), np.sin(np.radians(2.5)))
    h1 = hm + t_perp * np.tan(np.radians(e)) / s
    return 1 - np.exp(-np.maximum(h1, 0) ** 2 / (2 * hb ** 2))


def itu_stats_los(e, alpha, beta, gamma, h_r=1.5, n_max=200):
    """Building-statistics LOS model (built-up fraction alpha, density beta per km^2, Rayleigh scale gamma)."""
    s = 1000.0 / np.sqrt(alpha * beta)
    h = h_r + (np.arange(n_max) + 0.5) * s * np.tan(np.radians(e))
    return float(np.prod(1 - np.exp(-h ** 2 / (2 * gamma ** 2))))
REF = np.arange(20, 81, 10.0)        # reference elevations inside the traced grid
UNR = 150.0                           # loss above this: no ray reached the cell
B = 500


def load(c, tag=""):
    z = np.load(os.path.join(OUT, f"sky2_{c}{tag}.npz")); s = np.load(os.path.join(OUT, f"street2_{c}{tag}.npz"))
    els = z["els"]
    trust = z["d_edge"][:, None, :] * np.tan(np.radians(els))[None, :, None] >= float(z["H95"])
    return dict(los=z["los"], L=-z["exc"], els=els, azs=z["azs"], trust=trust, d_edge=z["d_edge"], built_h=z["built_h"],
                t=s["free"], H=s["hfac"], sky=s["sky"], ax1=s["axis_deg"], ax2=s["axis2_deg"], users=z["users"])


def wmean(x, w):
    w = w.astype(float); return float((x * w).sum() / max(w.sum(), 1))


def los_curve(D, idx=None):
    los, tr = D["los"], D["trust"]
    if idx is not None:
        los, tr = los[idx], tr[idx]
    return np.array([wmean(los[:, i, :], tr[:, i, :]) for i in range(len(D["els"]))])


def boot(D, fn, rng):
    n = D["los"].shape[0]
    return np.array([fn(rng.integers(0, n, n)) for _ in range(B)])


def p1_indep(t, H, mask, e):
    """Eq. (plos): open share plus E_t[F_H(h + t tan e)] with t and H taken independently over masked pairs."""
    tt, hh = t[mask], H[mask]
    if len(tt) == 0:
        return np.nan
    op = tt >= 500
    hs = np.sort(hh[~op]); z = H_U + tt[~op] * np.tan(np.radians(e))
    fr = np.searchsorted(hs, z, side="left") / max(len(hs), 1)
    return float(op.mean() + (1 - op.mean()) * fr.mean()) if len(hs) else float(op.mean())


def stat_curve(Dsrc, Dtgt_els, idx=None):
    t, H = Dsrc["t"], Dsrc["H"]
    if idx is not None:
        t, H = t[idx], H[idx]
    out = []
    for i, e in enumerate(Dtgt_els):
        k = list(Dsrc["els"]).index(e)
        m = (Dsrc["trust"][idx][:, k, :] if idx is not None else Dsrc["trust"][:, k, :])
        out.append(p1_indep(t, H, m, e))
    return np.array(out)


def nearest(els, table):
    idx = np.clip(np.rint(np.asarray(els, float) / 10.0).astype(int) - 1, 0, 8)
    return np.asarray(table)[idx]


def at_ref(els, curve):
    return np.array([curve[list(els).index(e)] for e in REF])


def rel_az(azs, ax):
    return np.abs(((azs[None, :] - ax[:, None] + 90) % 180) - 90)


def deficit(D):
    e = np.radians(D["els"])[None, :, None]
    t = D["t"][:, None, :]; H = D["H"][:, None, :]
    return H - (H_U + t * np.tan(e))


def blocked_pts(D):
    """Blocked, trusted pairs with a positive first-facade deficit: (deficit, loss, elevation, t, reached)."""
    dh = deficit(D)
    m = (~D["los"]) & D["trust"] & (D["t"][:, None, :] < 500) & (dh > 0)
    E = np.broadcast_to(D["els"][None, :, None], dh.shape)
    T = np.broadcast_to(D["t"][:, None, :], dh.shape)
    U = np.broadcast_to(np.arange(dh.shape[0])[:, None, None], dh.shape)
    return dh[m], D["L"][m], E[m], T[m], U[m]


def pinball(y, q, tau=0.5):
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def qreg(x, y, tau, rng=np.random.default_rng(3), nmax=60000):
    """Two-parameter quantile regression y ~ k0 + k1 x by direct minimization of the pinball loss (convex)."""
    from scipy.optimize import minimize
    if len(x) > nmax:
        i = rng.choice(len(x), nmax, replace=False); x, y = x[i], y[i]
    b0 = np.polyfit(x, y, 1)[::-1]
    f = lambda b: pinball(y, b[0] + b[1] * x, tau)
    r = minimize(f, b0, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-5, maxiter=4000))
    return r.x


def knife(nu):
    return np.where(nu > -0.78, 6.9 + 20 * np.log10(np.sqrt((nu - 0.1) ** 2 + 1) + nu - 0.1), 0.0)


def main(tag=""):
    rng = np.random.default_rng(11)
    Ds = {c: load(c, tag) for c in CITIES if os.path.exists(os.path.join(OUT, f"sky2_{c}{tag}.npz"))}
    res = {}
    for c, D in Ds.items():
        els, n = D["els"], D["los"].shape[0]
        r = dict(n_users=n, els=els.tolist(), trusted_share=float(D["trust"].mean()),
                 trusted_by_el=[float(D["trust"][:, i, :].mean()) for i in range(len(els))])
        true = los_curve(D); r["los"] = true.tolist()
        bt = boot(D, lambda ii: los_curve(D, ii), rng)
        r["los_lo"], r["los_hi"] = np.percentile(bt, 2.5, 0).tolist(), np.percentile(bt, 97.5, 0).tolist()
        r["los_ci_hw_max"] = float(np.nanmax((np.array(r["los_hi"]) - np.array(r["los_lo"])) / 2))
        # LOS at 30 deg against the user's distance to the hull edge (check of the edge treatment)
        dmin = D["d_edge"].min(axis=1); i30 = list(els).index(30.0)
        bins = [100, 150, 200, 300, 1e9]
        r["los30_by_edge"] = [[lo, hi, int(((dmin >= lo) & (dmin < hi)).sum()),
                               wmean(D["los"][(dmin >= lo) & (dmin < hi), i30, :], D["trust"][(dmin >= lo) & (dmin < hi), i30, :])]
                              for lo, hi in zip(bins[:-1], bins[1:]) if ((dmin >= lo) & (dmin < hi)).sum() > 5]
        # TR 38.811 at its reference elevations
        tr = {k: at_ref(REF, np.interp(REF, TR_EL, v)) for k, v in TR_LOS.items()}
        tref = at_ref(els, true)
        for k, v in tr.items():
            r[f"tr_{k}_max"] = float(np.max(np.abs(v - tref))); r[f"tr_{k}_mean"] = float(np.mean(np.abs(v - tref)))
        # eq. (plos): in sample (a decomposition), split half, and from each other city
        ins = stat_curve(D, els)
        rule1 = np.stack([(H_U + D["t"] * np.tan(np.radians(e)) > D["H"]) | (D["t"] >= 500) for e in els], 1)
        p1_emp = np.array([wmean(rule1[:, i, :], D["trust"][:, i, :]) for i in range(len(els))])
        r["stat_in"] = ins.tolist()
        r["skyline_share"] = (p1_emp - true).tolist()                 # delta(e)
        r["dependence"] = (ins - p1_emp).tolist()                     # independence of t and H
        r["stat_in_max"] = float(np.max(np.abs(at_ref(els, ins) - tref)))
        xu, yu = D["users"][:, 0], D["users"][:, 1]
        sh = []
        for xs_ in (xu, yu):                              # two half-plane splits, each predicting both ways
            west = xs_ < np.median(xs_); a_, b_ = np.flatnonzero(west), np.flatnonzero(~west)
            for p_, q_ in ((a_, b_), (b_, a_)):
                sh.append(np.max(np.abs(at_ref(els, stat_curve(D, els, p_)) - at_ref(els, los_curve(D, q_)))))
        r["stat_split_max"] = float(np.mean(sh)); r["stat_split_worst"] = float(np.max(sh))
        r["stat_cross"] = {o: float(np.max(np.abs(at_ref(els, stat_curve(Ds[o], els)) - tref))) for o in Ds if o != c}
        r["stat_cross_mean"] = {o: float(np.mean(np.abs(at_ref(els, stat_curve(Ds[o], els)) - tref))) for o in Ds if o != c}
        # first-facade and skyline rules; accuracy overall, near the boundary, on blocked pairs
        eps1 = np.where(D["t"] < 500, np.degrees(np.arctan2(D["H"] - H_U, D["t"])), -90.0)
        eps = np.maximum(eps1, D["sky"])
        rule2 = np.stack([eps < e for e in els], 1)
        T = D["trust"]
        near = np.abs(np.stack([e - eps for e in els], 1)) <= 5
        r["rule1_acc"] = wmean(rule1 == D["los"], T); r["rule2_acc"] = wmean(rule2 == D["los"], T)
        r["rule2_acc_near"] = wmean(rule2 == D["los"], T & near)
        r["rule2_acc_nlos"] = wmean(rule2 == D["los"], T & ~D["los"])
        r["rule1_false_blocked"] = wmean(~rule1 & D["los"], T)
        # idealized canyon from each user's street (in sample, like eq. plos)
        rl = np.radians(D["azs"][None, :] - D["ax1"][:, None])
        kl = ((D["ax1"] + 90) / 5).round().astype(int) % 72; kr = ((D["ax1"] - 90) / 5).round().astype(int) % 72
        dl = np.minimum(D["t"][np.arange(n), kl], 500.0); dr = np.minimum(D["t"][np.arange(n), kr], 500.0)
        tc = np.minimum(np.where(np.sin(rl) > 0, dl[:, None], dr[:, None]) / np.maximum(np.abs(np.sin(rl)), 1e-6), 1e4)
        hs = np.sort(D["H"][D["t"] < 500])
        can = [wmean(np.where(tc >= 500, 1.0, np.searchsorted(hs, H_U + tc * np.tan(np.radians(e)), side="left") / len(hs)),
                     T[:, i, :]) for i, e in enumerate(els)]
        r["canyon"] = can; r["canyon_max"] = float(np.max(np.abs(at_ref(els, np.array(can)) - tref)))
        # paired bootstrap of max error: canyon minus eq. (plos), in sample
        def diff(ii):
            tt = los_curve(D, ii)
            can_i = np.array([wmean(np.where(tc[ii] >= 500, 1.0, np.searchsorted(hs, H_U + tc[ii] * np.tan(np.radians(e)), side="left") / len(hs)),
                                    T[ii][:, i, :]) for i, e in enumerate(els)])
            return float(np.max(np.abs(at_ref(els, can_i) - at_ref(els, tt))) - np.max(np.abs(at_ref(els, stat_curve(D, els, ii)) - at_ref(els, tt))))
        dd = [diff(rng.integers(0, n, n)) for _ in range(100)]
        r["canyon_minus_stat_ci"] = np.percentile(dd, [2.5, 97.5]).tolist(); r["canyon_minus_stat"] = float(np.median(dd))
        # along / across at 30 deg for two axis definitions, and for users whose two axes agree (in a street)
        kn = ((D["ax2"] + 90) / 5).round().astype(int) % 72; kn2 = (kn + 36) % 72
        agree = np.abs(((D["ax1"] - D["ax2"] + 90) % 180) - 90) <= 15     # the manuscript's street label
        r["street_user_share"] = float(agree.mean())
        # elevation swing on the same users (15 -> 60 deg), for comparison with the azimuth swing
        i15, i60 = list(els).index(15.0), list(els).index(60.0)
        r["swing_el_street"] = wmean(D["los"][agree][:, i60, :], T[agree][:, i60, :]) - wmean(D["los"][agree][:, i15, :], T[agree][:, i15, :])
        r["swing_el_all"] = wmean(D["los"][:, i60, :], T[:, i60, :]) - wmean(D["los"][:, i15, :], T[:, i15, :])
        # P.681 roadside model: distance to the wall on the satellite's side along the wall normal, hb = Rayleigh mode
        Hf = D["H"][D["t"] < 500]; hb_u, hb_c = np.unique(np.round(D["built_h"]), return_counts=True); hb = float(hb_u[np.argmax(hb_c)])   # true mode, as P.681 defines hb
        r["p681_hb_rayleigh"] = float(np.sqrt(np.mean(Hf ** 2) / 2))
        relr = np.radians(D["azs"][None, :] - D["ax2"][:, None])
        dm = np.where(np.sin(relr) > 0, D["t"][np.arange(n), kn][:, None], D["t"][np.arange(n), kn2][:, None])
        p681 = [wmean(p681_los(np.minimum(dm, 500), e, np.degrees(relr), hb), T[:, i, :]) for i, e in enumerate(els)]
        r["p681"] = p681; r["p681_hb"] = hb
        r["p681_max"] = float(np.max(np.abs(at_ref(els, np.array(p681)) - tref)))
        r["p681_mean"] = float(np.mean(np.abs(at_ref(els, np.array(p681)) - tref)))
        # along / across at 30 deg: ray traced (wall-normal axis, all users), P.681, and the first-facade model
        # conditioned on the relative azimuth (in sample, and from one half-plane of users to the other)
        R2 = rel_az(D["azs"], D["ax2"]); P30 = p681_los(np.minimum(dm, 500), 30.0, np.degrees(relr), hb)
        def cond(sel, lo_, hi_):
            m_ = (R2 >= lo_) & (R2 <= hi_) & sel[:, None]
            tt_ = D["t"][m_]; hh_ = np.sort(D["H"][(D["t"] < 500) & sel[:, None]])
            op_ = tt_ >= 500; z_ = H_U + tt_[~op_] * np.tan(np.radians(30.0))
            return float(op_.mean() + (1 - op_.mean()) * np.mean(np.searchsorted(hh_, z_) / len(hh_)))
        allu = np.ones(n, bool); west = D["users"][:, 0] < np.median(D["users"][:, 0])
        r["p681_along"], r["p681_across"] = wmean(P30, T[:, i30, :] & (R2 <= 15)), wmean(P30, T[:, i30, :] & (R2 >= 75))
        r["ff_along"], r["ff_across"] = cond(allu, 0, 15), cond(allu, 75, 90)
        tw = lambda sel, lo_, hi_: wmean(D["los"][sel][:, i30, :], (T[:, i30, :] & (R2 >= lo_) & (R2 <= hi_))[sel])
        errs = []
        for a_, b_ in ((west, ~west), (~west, west)):
            errs += [abs(cond(a_, 0, 15) - tw(b_, 0, 15)), abs(cond(a_, 75, 90) - tw(b_, 75, 90))]
        r["ff_split_err_max"] = float(max(errs))
        st_ = STATS[c]; itu = np.array([itu_stats_los(e, st_["alpha"], st_["beta"], st_["gamma"]) for e in els])
        r["itu_stats"] = itu.tolist(); r["itu_max"] = float(np.max(np.abs(at_ref(els, itu) - tref)))
        r["itu_mean"] = float(np.mean(np.abs(at_ref(els, itu) - tref)))
        for nm, ax, sel in (("ax1", D["ax1"], np.ones(n, bool)), ("ax2", D["ax2"], np.ones(n, bool)), ("street", D["ax2"], agree)):
            R_ = rel_az(D["azs"], ax)
            L30, T30 = D["los"][:, i30, :], T[:, i30, :]
            al = (R_ <= 15) & sel[:, None]; ac = (R_ >= 75) & sel[:, None]
            r[f"along_{nm}"] = wmean(L30, T30 & al); r[f"across_{nm}"] = wmean(L30, T30 & ac)
            ba = []
            for _ in range(B):
                ii = rng.integers(0, n, n)
                ba.append((wmean(L30[ii], (T30 & al)[ii]), wmean(L30[ii], (T30 & ac)[ii])))
            ba = np.array(ba)
            r[f"along_{nm}_ci"] = np.percentile(ba[:, 0], [2.5, 97.5]).tolist(); r[f"across_{nm}_ci"] = np.percentile(ba[:, 1], [2.5, 97.5]).tolist()
        # loss of blocked trusted directions: uncensored (reached only) and censored medians
        med, med_c, p10, p90, unr = [], [], [], [], []
        for i in range(len(els)):
            m = (~D["los"][:, i, :]) & T[:, i, :]
            x = D["L"][:, i, :][m]
            if len(x) < 50:
                med.append(None); med_c.append(None); p10.append(None); p90.append(None); unr.append(None); continue
            xr = x[x < UNR]; xc = np.where(x < UNR, x, np.inf)
            med.append(float(np.median(xr))); med_c.append(float(np.median(xc)))
            p10.append(float(np.percentile(xc, 10))); p90.append(float(np.percentile(xc, 90))); unr.append(float(np.mean(x >= UNR)))
        r.update(nlos_med=med, nlos_med_cens=med_c, nlos_p10=p10, nlos_p90=p90, unreached=unr)
        r["tr_cl"] = np.interp(els, TR_EL, TR_CL_S).tolist()
        for k_ in ("dense_urban", "urban"):
            cl = np.interp(els, TR_EL, TR_CL_S)[None, :, None]; sg = np.interp(els, TR_EL, TR_SF_S[k_])[None, :, None]
            mb_ = (~D["los"]) & T; Lx = np.where(D["L"] < UNR, D["L"], np.inf)
            r[f"tr_cover_{k_}"] = wmean((Lx >= cl - 1.2816 * sg) & (Lx <= cl + 1.2816 * sg), mb_)
            r[f"tr_below_{k_}"] = wmean(Lx < cl - 1.2816 * sg, mb_)
        # coverage of the deficit law among blocked trusted pairs
        dh = deficit(D); mb = (~D["los"]) & T
        r["law_coverage"] = wmean((dh > 0) & (D["t"][:, None, :] < 500) & (D["L"] < UNR), mb)
        r["skyline_blocked_share"] = wmean((dh <= 0) | (D["t"][:, None, :] >= 500), mb)
        res[c] = r
    # loss law: leave one city out, per link. Unreached links are censored: set to CEN, above every prediction,
    # which is exact for the 0.1 and 0.5 quantile fits; the 0.9 band is fitted on reached links and its coverage
    # counts unreached links as above it. MAE is over reached links; intervals are paired cluster bootstraps over
    # users of the held-out city.
    CEN = 150.0
    P = {c: blocked_pts(D) for c, D in Ds.items()}
    bins = np.array([0, 1, 2, 4, 7, 10, 15, 20, 30, 50, 200.0])
    loo = {}

    def nu_of(d_, e_, t_):
        er = np.radians(e_); d1 = np.sqrt(t_ ** 2 + (d_ + t_ * np.tan(er)) ** 2)
        return d_ * np.cos(er) * np.sqrt(2.0 / (LAM * d1))

    for held in (P if len(P) > 1 else []):
        tr_ = [P[c] for c in P if c != held]
        d, L, E, T_ = (np.concatenate([x[k] for x in tr_]) for k in range(4))
        Lc = np.minimum(L, CEN); ok = L < UNR
        lx = np.log10(1 + d)
        beta = qreg(lx, Lc, 0.5)
        qfit = {0.1: qreg(lx, Lc, 0.1).tolist(), 0.9: qreg(lx[ok], L[ok], 0.9).tolist()}
        const = float(np.median(Lc))
        el_med = {e: float(np.median(Lc[E == e])) for e in np.unique(E)}
        off = float(np.median(L[ok] - knife(nu_of(d[ok], E[ok], T_[ok]))))
        dh, Lh, Eh, Th, Uh = P[held]
        reached = Lh < UNR; Lhc = np.minimum(Lh, CEN)
        pred = beta[0] + beta[1] * np.log10(1 + dh)
        pe = np.array([el_med.get(e, const) for e in Eh])
        J = knife(nu_of(dh, Eh, Th))
        trc = np.interp(Eh, TR_EL, TR_CL_S)
        preds = dict(law=pred, const=np.full_like(pred, const), el=pe, knife=J, knife_off=J + off, tr=trc)
        mae = {k: float(np.mean(np.abs(Lh[reached] - v[reached]))) for k, v in preds.items()}
        pin = {k: pinball(Lhc, v) for k, v in preds.items()}
        lo10 = qfit[0.1][0] + qfit[0.1][1] * np.log10(1 + dh); hi90 = qfit[0.9][0] + qfit[0.9][1] * np.log10(1 + dh)
        cover = float(np.mean((Lhc >= lo10) & (Lhc <= hi90) & reached))
        # paired bootstrap over held-out users of MAE differences against the law
        users = np.unique(Uh); idx = {u: np.flatnonzero(Uh == u) for u in users}
        boots = {k: [] for k in ("const", "el", "knife_off", "tr")}
        for _ in range(300):
            sel = np.concatenate([idx[u] for u in rng.choice(users, len(users))])
            sr = sel[reached[sel]]
            base = np.mean(np.abs(Lh[sr] - pred[sr]))
            for k in boots:
                boots[k].append(np.mean(np.abs(Lh[sr] - preds[k][sr])) - base)
        loo[held] = dict(beta=beta.tolist(), q=qfit, knife_off=off, cover90=cover,
                         unreached=float(np.mean(~reached)),
                         **{f"mae_{k}": v for k, v in mae.items()}, **{f"pin_{k}": v for k, v in pin.items()},
                         **{f"gain_{k}_ci": np.percentile(v, [2.5, 97.5]).tolist() for k, v in boots.items()},
                         **{f"gain_{k}": float(np.median(v)) for k, v in boots.items()},
                         knife_bias=float(np.median(J[reached] - Lh[reached])))
        xs, ys, ps = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (dh >= lo) & (dh < hi)
            if m.sum() > 50:
                xs.append(float(np.median(dh[m]))); ys.append(float(np.median(Lhc[m]))); ps.append(float(beta[0] + beta[1] * np.log10(1 + np.median(dh[m]))))
        loo[held].update(x=xs, y=ys, pred=ps, binned_mean_err=float(np.mean(np.abs(np.array(ys) - np.array(ps)))))
    res["loss_loo"] = loo
    if not P:
        return
    d = np.concatenate([P[c][0] for c in P]); L = np.concatenate([P[c][1] for c in P]); ok = L < UNR
    lxa = np.log10(1 + d); beta = qreg(lxa, np.minimum(L, CEN), 0.5); q10 = qreg(lxa, np.minimum(L, CEN), 0.1)
    lx = lxa[ok]; X = np.stack([np.ones(ok.sum()), lx], 1); q90 = qreg(lx, L[ok], 0.9)
    res["loss_fit_all"] = dict(k0=float(beta[0]), k1=float(beta[1]), q10=q10.tolist(), q90=q90.tolist(),
                               r2=float(1 - np.var(L[ok] - X @ beta) / np.var(L[ok])))
    json.dump(res, open(os.path.join(OUT, f"analysis2{tag}.json"), "w"), indent=1)
    for c in Ds:
        r = res[c]
        print(c, "n=%d trusted %.2f | LOS15/30/60 %.2f/%.2f/%.2f | TR max u/d %.2f/%.2f mean %.2f/%.2f | stat in %.2f split %.2f | canyon %.2f | rule2 %.3f near %.3f | along/across ax1 %.2f/%.2f ax2 %.2f/%.2f street %.2f/%.2f (%.0f%%)"
              % (r["n_users"], r["trusted_share"], r["los"][0], r["los"][3], r["los"][9], r["tr_urban_max"], r["tr_dense_urban_max"],
                 r["tr_urban_mean"], r["tr_dense_urban_mean"], r["stat_in_max"], r["stat_split_max"], r["canyon_max"], r["rule2_acc"],
                 r["rule2_acc_near"], r["along_ax1"], r["across_ax1"], r["along_ax2"], r["across_ax2"], r["along_street"], r["across_street"],
                 100 * r["street_user_share"]))
        print("   cross-city stat max:", {k: round(v, 2) for k, v in r["stat_cross"].items()}, "edge check:", [(a, b, n_, round(v, 2)) for a, b, n_, v in r["los30_by_edge"]])
        print("   NLOS med", np.round([x if x is not None else np.nan for x in r["nlos_med"]], 1), "censored", np.round([x if x is not None else np.nan for x in r["nlos_med_cens"]], 1))
    for c, v in loo.items():
        print("LOO", c, "MAE law %.1f const %.1f elev %.1f knife+off %.1f TR %.1f | gain vs const %.1f %s, vs knife+off %.1f %s, vs TR %.1f | cover %.2f | unr %.3f"
              % (v["mae_law"], v["mae_const"], v["mae_el"], v["mae_knife_off"], v["mae_tr"], v["gain_const"], np.round(v["gain_const_ci"], 1),
                 v["gain_knife_off"], np.round(v["gain_knife_off_ci"], 1), v["gain_tr"], v["cover90"], v["unreached"]))
    for c in Ds:
        r = res[c]
        print(c, "p681 max %.2f mean %.2f (hb %.1f) | itu max %.2f | TR cover dense %.2f below %.2f, urban %.2f | street share %.2f along/across ax2 %.2f/%.2f street %.2f/%.2f | swing el street %.2f all %.2f | split %.2f worst %.2f | canyon-stat %.2f %s"
              % (r["p681_max"], r["p681_mean"], r["p681_hb"], r["itu_max"], r["tr_cover_dense_urban"], r["tr_below_dense_urban"], r["tr_cover_urban"],
                 r["street_user_share"], r["along_ax2"], r["across_ax2"], r["along_street"], r["across_street"], r["swing_el_street"], r["swing_el_all"],
                 r["stat_split_max"], r["stat_split_worst"], r["canyon_minus_stat"], np.round(r["canyon_minus_stat_ci"], 2)))
    print("fit all", res["loss_fit_all"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
