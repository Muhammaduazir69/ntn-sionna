#!/usr/bin/env python3
"""Line-of-sight and excess-loss models against the ray-traced sky maps; writes out/models.json.

LOS models, for each city and elevation e:
  first-facade rule   the exact rule per user and azimuth: clear iff h + t(a) tan e > H(a)
  statistical model   P_LOS(e) = E[ F_H(h + t tan e) ], t and H the city's facade-distance and facade-height
                      distributions taken independently (inputs from building data, not fitted to LOS)
  TR 38.811           urban and dense-urban tables;   P.1410: geometric model with mesh parameters
Excess loss: the median loss of blocked directions against the deficit d = H - (h + t tan e) of the first facade,
L(d) = a + b log10(1 + d), fitted on three cities and tested on the held-out fourth (leave-one-city-out)."""
import json, os
import numpy as np
from p9_analysis import CITIES, TR_EL, TR_LOS, p1410, nearest

HERE = os.path.dirname(os.path.abspath(__file__))
H_U = 1.5


def load(c):
    z = np.load(os.path.join(HERE, "out", f"sky_{c}.npz")); s = np.load(os.path.join(HERE, "out", f"street_{c}.npz"))
    return z["los"], z["exc"], z["els"], s["free"], s["hfac"]


def main():
    S = json.load(open(os.path.join(HERE, "out", "city_stats.json")))
    out, pts = {}, {}
    for c in CITIES:
        los, exc, els, t, H = load(c)
        rule = np.stack([(H_U + t * np.tan(np.radians(e)) > H) | (t >= 500) for e in els], 1)
        acc = float((rule == los).mean())
        sky = np.load(os.path.join(HERE, "out", f"street_{c}.npz"))["sky"]
        comb = np.maximum(np.where(t < 500, np.degrees(np.arctan2(H - H_U, t)), -90.0), sky)
        rule2 = np.stack([comb < e for e in els], 1)
        acc2 = float((rule2 == los).mean())
        false_los = float((rule & ~los).mean())
        # LOS against the satellite azimuth relative to the street axis, at 30 deg elevation
        ax = np.load(os.path.join(HERE, "out", f"street_{c}.npz"))["axis_deg"]
        azs = np.arange(0, 360, 5.0)
        rel = np.abs(((azs[None, :] - ax[:, None] + 90) % 180) - 90)
        i30 = list(els).index(30.0)
        along = float(los[:, i30, :][rel < 15].mean()); across = float(los[:, i30, :][rel >= 75].mean())
        tt, hh = t[t < 500], H[t < 500]
        stat = []
        for e in els:
            # E over independent t and H: P(H < h + t tan e) averaged over t; open directions (t >= 500) are clear
            pt_open = np.mean(t >= 500)
            zr = np.sort(H_U + tt * np.tan(np.radians(e)))
            hs = np.sort(hh)
            frac = np.searchsorted(hs, zr, side="left") / len(hs)      # F_H at each ray height
            stat.append(pt_open + (1 - pt_open) * frac.mean())
        true = los.mean(axis=(0, 2))
        tr_u = nearest(els, TR_LOS["urban"]); tr_d = nearest(els, TR_LOS["dense_urban"])
        st = S[c]; pi = np.array([p1410(e, st["alpha"], st["beta"], st["gamma"]) for e in els])
        out[c] = dict(els=els.tolist(), true=true.tolist(), rule_acc=acc, rule2_acc=acc2, rule_false_los=false_los,
                      los30_along=along, los30_across=across, stat=stat,
                      err_stat=float(np.max(np.abs(np.array(stat) - true))), err_tr_urban=float(np.max(np.abs(tr_u - true))),
                      err_tr_dense=float(np.max(np.abs(tr_d - true))), err_p1410=float(np.max(np.abs(pi - true))),
                      unreached=float(np.mean(-exc[~los] >= 150)))
        # blocked directions: deficit and loss
        d_all, L_all = [], []
        for i, e in enumerate(els):
            dfc = H - (H_U + t * np.tan(np.radians(e)))
            m = (~los[:, i, :]) & (t < 500) & (-exc[:, i, :] < 150) & (dfc > 0)
            d_all.append(dfc[m]); L_all.append(-exc[:, i, :][m])
        pts[c] = (np.concatenate(d_all), np.concatenate(L_all))
        # baseline: single knife edge at the first facade (ITU-R P.526, J(nu)), nu from the deficit and the
        # distance to the edge along the ray; the far side of the edge is at infinity
        lam = 3e8 / 2e9; nu_all = []
        for i, e in enumerate(els):
            er = np.radians(e); dfc = H - (H_U + t * np.tan(er))
            m = (~los[:, i, :]) & (t < 500) & (-exc[:, i, :] < 150) & (dfc > 0)
            nu_all.append(dfc[m] * np.cos(er) * np.sqrt(2.0 / (lam * t[m] / np.cos(er))))
        nu = np.concatenate(nu_all)
        J = 6.9 + 20 * np.log10(np.sqrt((nu - 0.1) ** 2 + 1) + nu - 0.1)
        out[c]["knife_bias"] = float(np.median(J - pts[c][1]))
        # idealized canyon baseline (Saunders-Evans / Maccari type): the user's own street, walls parallel to
        # its axis at the measured lateral distances, infinitely long; facade heights from the city law F_H
        s = np.load(os.path.join(HERE, "out", f"street_{c}.npz")); ax_deg = s["axis_deg"]; azs = np.arange(0, 360, 5.0)
        k_l = ((ax_deg + 90) / 5).round().astype(int) % 72; k_r = ((ax_deg - 90) / 5).round().astype(int) % 72
        dl = np.minimum(t[np.arange(len(t)), k_l], 500.0); dr = np.minimum(t[np.arange(len(t)), k_r], 500.0)
        rel = np.radians(azs[None, :] - ax_deg[:, None]); sr = np.sin(rel)
        dperp = np.where(sr > 0, dl[:, None], dr[:, None])
        tc = np.minimum(dperp / np.maximum(np.abs(sr), 1e-6), 1e4)
        hsort = np.sort(hh); canyon = []
        for e in els:
            z = H_U + tc * np.tan(np.radians(e))
            canyon.append(float(np.mean(np.where(tc >= 500, 1.0, np.searchsorted(hsort, z, side="left") / len(hsort)))))
        out[c]["canyon"] = canyon
        out[c]["err_canyon"] = float(np.max(np.abs(np.array(canyon) - true)))
        # 95 % cluster-bootstrap intervals over users (users are the independent units)
        rng = np.random.default_rng(7); nU = los.shape[0]; B = 1000
        bt = np.array([los[rng.integers(0, nU, nU)].mean(axis=(0, 2)) for _ in range(B)])
        lo_, hi_ = np.percentile(bt, [2.5, 97.5], axis=0)
        out[c]["los_ci_halfwidth_max"] = float(np.max((hi_ - lo_) / 2))
        bs_al, bs_ac = [], []
        for _ in range(B):
            ii = rng.integers(0, nU, nU)
            L30 = los[ii, i30, :]; R = rel_ = np.abs(((azs[None, :] - ax_deg[ii][:, None] + 90) % 180) - 90)
            bs_al.append(L30[R < 15].mean()); bs_ac.append(L30[R >= 75].mean())
        out[c]["los30_along_ci"] = np.percentile(bs_al, [2.5, 97.5]).tolist()
        out[c]["los30_across_ci"] = np.percentile(bs_ac, [2.5, 97.5]).tolist()
        print(c, "canyon err %.3f, LOS CI hw %.3f, along CI %s across CI %s" % (out[c]["err_canyon"], out[c]["los_ci_halfwidth_max"],
              np.round(out[c]["los30_along_ci"], 2), np.round(out[c]["los30_across_ci"], 2)))
        print(c, "rule acc %.3f | max |LOS err|: statistical %.3f, TR urban %.3f, TR dense %.3f, P.1410 %.3f | unreached %.3f"
              % (acc, out[c]["err_stat"], out[c]["err_tr_urban"], out[c]["err_tr_dense"], out[c]["err_p1410"], out[c]["unreached"]))
    # leave-one-city-out fit of the median loss against deficit
    bins = np.array([0, 1, 2, 4, 7, 10, 15, 20, 30, 50, 200.0])
    def binned(d, L):
        x, y = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (d >= lo) & (d < hi)
            if m.sum() > 50:
                x.append(np.median(d[m])); y.append(np.median(L[m]))
        return np.array(x), np.array(y)
    loo = {}
    for held in CITIES:
        d = np.concatenate([pts[c][0] for c in CITIES if c != held]); L = np.concatenate([pts[c][1] for c in CITIES if c != held])
        x, y = binned(d, L)
        b, a = np.polyfit(np.log10(1 + x), y, 1)
        xh, yh = binned(*pts[held])
        pred = a + b * np.log10(1 + xh)
        loo[held] = dict(a=float(a), b=float(b), max_err=float(np.max(np.abs(pred - yh))), mean_err=float(np.mean(np.abs(pred - yh))),
                         x=xh.tolist(), y=yh.tolist(), pred=pred.tolist())
        print("held out", held, "L = %.1f + %.1f log10(1+d); median-loss error on held-out bins: mean %.1f, max %.1f dB"
              % (a, b, loo[held]["mean_err"], loo[held]["max_err"]))
    d = np.concatenate([pts[c][0] for c in CITIES]); L = np.concatenate([pts[c][1] for c in CITIES])
    x, y = binned(d, L); b, a = np.polyfit(np.log10(1 + x), y, 1)
    out["loss_fit_all"] = dict(a=float(a), b=float(b))
    out["loss_loo"] = loo
    json.dump(out, open(os.path.join(HERE, "out", "models.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
