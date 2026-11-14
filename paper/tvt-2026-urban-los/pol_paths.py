#!/usr/bin/env python3
"""M9, R7: polarization of the paths that reach a blocked user. Path solver (specular paths and, since round 5, diffracted paths too;
an earlier run with diffraction off wrongly suggested the solver returns none) with dual V/H arrays at both ends gives a 2x2 Jones matrix per path, from the user (transmitter) to a
receiver on the 1000 m plane in a blocked direction. Path powers are summed, as in the radio map. Polarizations at the
satellite are defined as matched to the user's polarization on the free-space path (J_LOS), so a co-polar circular
link has no loss in LOS and a one-bounce path is cross-polar. Excess loss is relative to LOS of the same pairing.
Writes out/pol_<city><tag>.npz."""
import json, os, sys, time
import numpy as np
import mitsuba as mi
import sionna.rt as rt
from sky_maps import SCENES, ELS, AZS, dirs
from sky_maps2 import set_facades
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out"); FREQ = 2e9
DIFF = True
UV = np.array([1.0, 0.0], complex); UR = np.array([1.0, -1j], complex) / np.sqrt(2)


def jones(paths):
    re, im = paths.a
    a = np.array(re) + 1j * np.array(im)      # [num_rx, rx_ant, num_tx, tx_ant, num_paths]
    v = np.array(paths.valid)
    return a[:, :, 0, :, :], v.reshape(a.shape[0], -1) if v.ndim else v


def solve(sc, solver, p, targets, depth, samples):
    sc.add(rt.Transmitter("ue", position=[float(x) for x in p]))
    for k, q in enumerate(targets):
        sc.add(rt.Receiver(f"r{k}", position=[float(x) for x in q]))
    P = solver(sc, max_depth=depth, max_num_paths_per_src=2000000, samples_per_src=int(samples), synthetic_array=True, los=True,
               specular_reflection=True, diffuse_reflection=False, refraction=False, diffraction=DIFF, edge_diffraction=DIFF)
    a = P.a; A = np.array(a[0]) + 1j * np.array(a[1])
    inter = np.array(P.interactions); vd = np.array(P.valid)
    sc.remove("ue")
    for k in range(len(targets)):
        sc.remove(f"r{k}")
    return A, vd, inter


def run(city, n_users=60, n_dir=24, depth=3, samples=1e7, tag="", wall=None, thick=None, seed=7, src=""):
    z = np.load(f"{OUT}/sky2_{city}{src}.npz"); users = z["users"][:n_users]; los = z["los"]; L = -z["exc"]
    els = z["els"]; d = dirs(); top_off = 1000.0
    sc = rt.load_scene(SCENES[city]); sc.frequency = FREQ; set_facades(sc, wall, thick)
    top = float(sc.mi_scene.bbox().max.z); H = top + top_off
    for s, in [(sc,)]:
        s.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="VH")
        s.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="VH")
    em = rt.load_scene(); em.frequency = FREQ
    em.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="VH")
    em.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="VH")
    solver = rt.PathSolver(); rng = np.random.default_rng(seed); out = []
    t0 = time.time()
    for u, p in enumerate(users):
        cand = np.argwhere(~los[u])
        if len(cand) == 0:
            continue
        pick = cand[rng.choice(len(cand), min(n_dir, len(cand)), replace=False)]
        tg = [p + (H - p[2]) / d[i, j, 2] * d[i, j] for i, j in pick]
        A, vd, inter = solve(sc, solver, p, tg, depth, samples)
        A0, _, _ = solve(em, solver, p, tg, 0, 1e4)
        for k, (i, j) in enumerate(pick):
            Jp = A[k, :, 0, :, :]; valid = vd[k, 0, :] if vd.ndim == 3 else vd[k]
            Jp = Jp[:, :, valid] if valid.ndim == 1 else Jp
            J0 = A0[k, :, 0, :, 0]; nb = inter[:, k, 0, :][:, valid] if inter.ndim == 4 else None
            if Jp.shape[-1] == 0:
                out.append((u, els[i], AZS[j], L[u, i, j], *[np.nan] * 4, 0, 0, np.nan, np.nan, np.nan)); continue
            nrm = np.sum(np.abs(J0 @ UV) ** 2)                      # free-space path gain
            ideal = np.sum(np.abs(np.einsum("abp,b->ap", Jp, UV)) ** 2) / nrm
            wV = J0 @ UV; wV /= np.linalg.norm(wV); wR = J0 @ UR; wR /= np.linalg.norm(wR)
            pw = lambda w, uu: np.sum(np.abs(w.conj() @ np.einsum("abp,b->ap", Jp, uu)) ** 2) / nrm
            parity = (np.sum(nb == 1, axis=0) if nb is not None else np.zeros(Jp.shape[-1], int))
            pp = np.sum(np.abs(np.einsum("abp,b->ap", Jp, UV)) ** 2, axis=0) / nrm
            hasd = np.any(nb == 8, axis=0); hasr = np.any(nb == 1, axis=0)
            pd_, pr_, pm_ = pp[hasd & ~hasr].sum(), pp[hasr & ~hasd].sum(), pp[hasd & hasr].sum()
            out.append((u, els[i], AZS[j], L[u, i, j], ideal, pw(wV, UV), pw(wR, UV) / 0.5, pw(wR, UR), Jp.shape[-1], float(np.mean(parity % 2)), pd_, pr_, pm_))
        if (u + 1) % 10 == 0:
            print(city, tag, u + 1, f"{time.time() - t0:.0f} s", flush=True)
    np.savez_compressed(f"{OUT}/pol_{city}{tag}.npz", rows=np.array(out, float), cols="user,el,az,L_rm,ideal,VV,V_R_rel_LOS,RR,npaths,odd_share,P_diff,P_spec,P_mixed",
                        params=json.dumps(dict(depth=depth, samples=samples, wall=wall, thick=thick, n_users=n_users, n_dir=n_dir)))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("city"); ap.add_argument("--users", type=int, default=60); ap.add_argument("--dirs", type=int, default=24)
    ap.add_argument("--depth", type=int, default=3); ap.add_argument("--samples", type=float, default=1e7); ap.add_argument("--tag", default="")
    ap.add_argument("--wall", default=None); ap.add_argument("--thick", type=float, default=None); ap.add_argument("--src", default="")
    a = ap.parse_args(); run(a.city, a.users, a.dirs, a.depth, a.samples, a.tag, a.wall, a.thick, src=a.src)
