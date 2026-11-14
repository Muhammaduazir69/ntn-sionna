#!/usr/bin/env python3
"""Canonical check of the rough-facade setting: a single rough wall lit by a transmitter, radio map against the numerical
Lambertian integral P/Pt = (lam/4pi)^2 Int S^2 |R|^2 cos(ti) cos(ts) / (pi r1^2 r2^2) dA over the wall. Reflection off.
Writes out/validate_diffuse.json. The difference is the error of the excess loss with rough facades."""
import json, os
import numpy as np
import sionna.rt as rt
HERE = os.path.dirname(os.path.abspath(__file__)); F = 2e9; LAM = 299792458.0 / F; S = 0.4
XML = """<scene version="2.1.0">
  <bsdf type="itu-radio-material" id="m"><string name="type" value="concrete"/><float name="thickness" value="5.0"/><float name="scattering_coefficient" value="0.4"/></bsdf>
  <shape type="rectangle" id="wall"><transform name="to_world"><scale x="150" y="50"/><rotate x="1" y="0" z="0" angle="90"/><translate x="0" y="0" z="50"/></transform><ref id="m" name="bsdf"/></shape>
</scene>"""
sc = rt.load_scene_from_string(XML); sc.frequency = F
sc.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V"); sc.rx_array = sc.tx_array
tx = np.array([0.0, -10.0, 1.5]); sc.add(rt.Transmitter("t", position=tx.tolist()))
H = 1000.0; cs = 40.0; L = 1200.0
rm = rt.RadioMapSolver()(sc, center=[0.0, -300.0, H], orientation=[0.0, 0.0, 0.0], size=[L, 800.0], cell_size=[cs, cs], samples_per_tx=10 ** 9, max_depth=1, los=False,
                         specular_reflection=False, diffuse_reflection=True, diffraction=False, refraction=False)
g = np.array(rm.path_gain)[0]; cc = np.array(rm.cell_centers)
# power reflection of the 5 m concrete slab averaged over V/H at the local incidence
eta = 5.24 - 1j * 17.98 * 0.0462 * 2 ** 0.7822 / 2
xs = np.arange(-150, 150, 2.0) + 1.0; zs = np.arange(0, 100, 2.0) + 1.0; X, Z = np.meshgrid(xs, zs); dA = 4.0
r1v = np.stack([X - tx[0], np.zeros_like(X) - tx[1], Z - tx[2]], -1); r1 = np.linalg.norm(r1v, axis=-1); ci = np.abs(r1v[..., 1]) / r1
s2 = 1 - ci ** 2; rt_ = np.sqrt(eta - s2); te = (ci - rt_) / (ci + rt_); tm = (eta * ci - rt_) / (eta * ci + rt_)
R2 = 0.5 * (np.abs(te) ** 2 + np.abs(tm) ** 2)
ana = np.zeros(g.shape)
pts = cc.reshape(-1, 3)
for k, p in enumerate(pts):
    if p[1] >= 0:
        continue
    r2v = np.stack([p[0] - X, p[1] - np.zeros_like(X), p[2] - Z], -1); r2 = np.linalg.norm(r2v, axis=-1); cs_ = np.abs(r2v[..., 1]) / r2
    ana.flat[k] = (LAM / (4 * np.pi)) ** 2 * np.sum(S ** 2 * R2 * ci * cs_ / (np.pi * r1 ** 2 * r2 ** 2)) * dA
m = (ana > 0) & (g > 0)
d = 10 * np.log10(g[m] / ana[m]); res = dict(n=int(m.sum()), median_db=float(np.median(d)), p10=float(np.percentile(d, 10)), p90=float(np.percentile(d, 90)))
json.dump(res, open(os.path.join(HERE, "out", "validate_diffuse.json"), "w"), indent=1); print(res)
