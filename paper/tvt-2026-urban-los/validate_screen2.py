#!/usr/bin/env python3
"""SUPERSEDED by validate_screen3.py (R3): the cube scene here is not a half plane; its offsets were an artifact and are withdrawn. R7: solver diffraction against the exact half-plane in deep shadow. PEC half plane, hard polarization (E normal to the edge), Keller coefficient:
D = exp(-j pi/4)/(2 sqrt(2 pi k)) * [ 1/cos((phi-phi0)/2) + 1/cos((phi+phi0)/2) ], excess over free space |D|^2 (s + s')/(s s').
Runs thin and thick screens, V and H. Writes out/validate_screen2.json; also records path-solver diffracted path counts for the claim that none appear."""
import json, os
import numpy as np
import sionna.rt as rt
HERE = os.path.dirname(os.path.abspath(__file__)); F = 2e9; LAM = 299792458.0 / F; K = 2 * np.pi / LAM
XML = """<scene version="2.1.0"><bsdf type="itu-radio-material" id="m"><string name="type" value="metal"/><float name="thickness" value="0.01"/></bsdf>
<shape type="cube" id="s"><transform name="to_world"><scale x="%g" y="400" z="200"/><translate x="0" y="0" z="-200"/></transform><ref id="m" name="bsdf"/></shape></scene>"""
d1, d2, ht = 300.0, 50.0, 20.0; res = {}
for hw, pol in ((0.1, "V"), (0.1, "H"), (0.005, "V"), (0.005, "H")):
    sc = rt.load_scene_from_string(XML % hw); sc.frequency = F
    sc.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization=pol); sc.rx_array = sc.tx_array
    sc.add(rt.Transmitter("tx", position=[-d1, 0.0, ht]))
    rm = rt.RadioMapSolver()(sc, center=[d2, 0.0, -15.0], orientation=[0.0, float(np.pi / 2), 0.0], size=[60.0, 20.0], cell_size=[2.0, 2.0], samples_per_tx=10 ** 8, max_depth=1, los=True, specular_reflection=False, diffraction=True, edge_diffraction=True, refraction=False)
    g = np.array(rm.path_gain).reshape(-1); cc = np.array(rm.cell_centers).reshape(-1, 3); sel = np.abs(cc[:, 1]) < 1.5
    z = cc[sel, 2]; gz = g[sel]; o = np.argsort(-z); z, gz = z[o], gz[o]
    d = np.hypot(d1 + d2, z - ht); ex = -10 * np.log10(np.maximum(gz, 1e-30) / (LAM / (4 * np.pi * d)) ** 2)
    h = -(ht + (z - ht) * d1 / (d1 + d2)); nu = h * np.sqrt(2 * (d1 + d2) / (LAM * d1 * d2))
    sp = np.hypot(d1, ht); s = np.hypot(d2, z)
    a_in = np.arctan2(ht, d1); a_out = np.arctan2(-z, d2)      # angles above and below the edge plane
    # wedge angles from the source-side screen face: incidence phi0 = pi/2 + a_in (a_in above the edge plane), observation phi = 3 pi/2 + a_out, a_out>0 below the edge on the far side
    phi0 = np.pi / 2 + a_in; phi = 1.5 * np.pi + a_out
    D = lambda sg: np.exp(-1j * np.pi / 4) / (2 * np.sqrt(2 * np.pi * K)) * (1 / np.cos((phi - phi0) / 2) + sg / np.cos((phi + phi0) / 2))
    ex_h = -10 * np.log10(np.abs(D(1)) ** 2 * (s + sp) / (s * sp)); ex_s = -10 * np.log10(np.abs(D(-1)) ** 2 * (s + sp) / (s * sp))
    J = 6.9 + 20 * np.log10(np.sqrt((nu - 0.1) ** 2 + 1) + nu - 0.1)
    m = nu > 2
    res[f"hw{hw}_{pol}"] = dict(nu=nu[m].tolist(), solver=ex[m].tolist(), hard=ex_h[m].tolist(), soft=ex_s[m].tolist(), knife=J[m].tolist(), minus_hard=[float(x) for x in np.percentile(ex[m] - ex_h[m], [0, 50, 100])], minus_soft=[float(x) for x in np.percentile(ex[m] - ex_s[m], [0, 50, 100])], minus_knife=[float(x) for x in np.percentile(ex[m] - J[m], [0, 50, 100])])
    print(hw, pol, "solver-hard", np.round(res[f"hw{hw}_{pol}"]["minus_hard"], 2), "solver-soft", np.round(res[f"hw{hw}_{pol}"]["minus_soft"], 2), "solver-knife", np.round(res[f"hw{hw}_{pol}"]["minus_knife"], 2), flush=True)
    sc.remove("tx")
# path solver: any diffracted path into the shadow?
sc = rt.load_scene_from_string(XML % 0.1); sc.frequency = F; sc.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V"); sc.rx_array = sc.tx_array
sc.add(rt.Transmitter("tx", position=[-d1, 0.0, ht])); sc.add(rt.Receiver("rx", position=[d2, 0.0, -15.0]))
P = rt.PathSolver()(sc, max_depth=1, samples_per_src=10 ** 7, los=True, specular_reflection=False, diffuse_reflection=False, refraction=False, diffraction=True, edge_diffraction=True)
res["path_solver_valid_paths"] = int(np.array(P.valid).sum()); print("path solver paths", res["path_solver_valid_paths"])
json.dump(res, open(os.path.join(HERE, "out", "validate_screen2.json"), "w"), indent=1)
A = np.array(P.a[0]) + 1j * np.array(P.a[1]); print("types", np.array(P.interactions).ravel(), "PG path dB", 10 * np.log10((np.abs(A) ** 2).sum() + 1e-30))
d_ = np.hypot(d1 + d2, -15 - ht); print("excess loss at (d2, -15) path solver", -10 * np.log10((np.abs(A) ** 2).sum() / (LAM / (4 * np.pi * d_)) ** 2))
res["path_solver_excess"] = float(-10 * np.log10((np.abs(A) ** 2).sum() / (LAM / (4 * np.pi * d_)) ** 2)); json.dump(res, open(os.path.join(HERE, "out", "validate_screen2.json"), "w"), indent=1)
