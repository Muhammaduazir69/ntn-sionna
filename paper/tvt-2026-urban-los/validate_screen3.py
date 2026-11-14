#!/usr/bin/env python3
"""R3 (C1, m1): canonical solver checks on geometry the solver actually sees.
(a) A vertical half plane is one two-triangle PLY sheet, metal, its free top edge at z = 0. Reference: exact Keller half plane, phi0 = pi/2 + a_in,
    phi = 3 pi/2 + a_out, measured from the source-side face. Path solver at five points and radio map down a vertical cut, V and H.
(b) A Mitsuba `cube`, the earlier "screen": 24 vertices, so every edge is free and the solver diffracts at the rear edge of its top face, a horizontal plate
    reaching toward the source. Scored against both the vertical and the horizontal half plane (phi0 = a_in, phi = pi + a_out).
(c) Roof rim: canonical block with its walls and roof as separate meshes, as in the city scenes, against one welded mesh, path solver and radio map.
Writes out/validate_screen3.json."""
import json, os
import numpy as np
import mitsuba as mi
import sionna.rt as rt
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out"); PLY = os.path.join(OUT, "canon"); os.makedirs(PLY, exist_ok=True)
F = 2e9; LAM = 299792458.0 / F; K = 2 * np.pi / LAM; d1, d2, ht = 300.0, 50.0, 20.0; ZS = (-10.0, -15.0, -20.0, -30.0, -40.0)


def ply(name, V, T):
    p = os.path.join(PLY, name + ".ply")
    with open(p, "w") as fo:
        fo.write(f"ply\nformat ascii 1.0\nelement vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\nelement face {len(T)}\nproperty list uchar int vertex_indices\nend_header\n")
        for v in V:
            fo.write("%g %g %g\n" % tuple(v))
        for t in T:
            fo.write("3 %d %d %d\n" % tuple(t))
    return p




def scene(shapes, mats):
    x = '<scene version="2.1.0">'
    for k, (ty, th) in mats.items():
        x += f'<bsdf type="itu-radio-material" id="{k}"><string name="type" value="{ty}"/><float name="thickness" value="{th}"/></bsdf>'
    for i, (body, m) in enumerate(shapes):
        x += f'<shape id="s{i}" {body}<boolean name="face_normals" value="true"/><ref id="{m}" name="bsdf"/></shape>'
    return rt.load_scene_from_string(x + "</scene>")


def plyshape(p):
    return f'type="ply"><string name="filename" value="{p}"/>'


def cubeshape(hw):
    return f'type="cube"><transform name="to_world"><scale x="{hw}" y="400" z="200"/><translate x="0" y="0" z="-200"/></transform>'


def keller(phi, phi0, s, sp, d, sg):
    D = np.exp(-1j * np.pi / 4) / (2 * np.sqrt(2 * np.pi * K)) * (1 / np.cos((phi - phi0) / 2) + sg / np.cos((phi + phi0) / 2))
    return -10 * np.log10(np.abs(D) ** 2 * d ** 2 / (s * sp * (s + sp)))


def exact(z):
    """excess loss of the vertical and horizontal half planes, hard (+) and soft (-), at (d2, z)"""
    z = np.asarray(z, float); sp = np.hypot(d1, ht); s = np.hypot(d2, z); d = np.hypot(d1 + d2, z - ht); ai = np.arctan2(ht, d1); ao = np.arctan2(-z, d2)
    v0, v = np.pi / 2 + ai, 1.5 * np.pi + ao; h0, h = ai, np.pi + ao
    return dict(vhard=keller(v, v0, s, sp, d, 1), vsoft=keller(v, v0, s, sp, d, -1), hhard=keller(h, h0, s, sp, d, 1), hsoft=keller(h, h0, s, sp, d, -1))


def arrays(sc, pol):
    sc.frequency = F; sc.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization=pol); sc.rx_array = sc.tx_array


def path_gain(sc, tx, rx, depth=1, samples=10 ** 7):
    sc.add(rt.Transmitter("tx", position=[float(v) for v in tx])); sc.add(rt.Receiver("rx", position=[float(v) for v in rx]))
    P = rt.PathSolver()(sc, max_depth=depth, samples_per_src=samples, los=True, specular_reflection=False, diffuse_reflection=False, refraction=False, diffraction=True, edge_diffraction=True)
    A = np.array(P.a[0]) + 1j * np.array(P.a[1]); p = np.abs(A) ** 2; sc.remove("tx"); sc.remove("rx")
    return float(p.sum()), int((p > 0).sum()), float(p.max()) if p.size else 0.0


def run_screen(sc, pol, out):
    arrays(sc, pol); rows = []
    for z in ZS:
        g, n, g1 = path_gain(sc, (-d1, 0.0, ht), (d2, 0.0, z)); d = np.hypot(d1 + d2, z - ht); fs = (LAM / (4 * np.pi * d)) ** 2; rows.append(dict(z=z, solver=-10 * np.log10(max(g, 1e-30) / fs), top=-10 * np.log10(max(g1, 1e-30) / fs), paths=n, **{k: float(v) for k, v in exact(z).items()}))
    sc.add(rt.Transmitter("tx", position=[-d1, 0.0, ht]))
    rm = rt.RadioMapSolver()(sc, center=[d2, 0.0, -25.0], orientation=[0.0, float(np.pi / 2), 0.0], size=[40.0, 12.0], cell_size=[2.0, 2.0], samples_per_tx=10 ** 8, max_depth=1, los=True, specular_reflection=False, diffraction=True, edge_diffraction=True, refraction=False)
    g = np.array(rm.path_gain).reshape(-1); cc = np.array(rm.cell_centers).reshape(-1, 3); sc.remove("tx")
    sel = (np.abs(cc[:, 1]) < 1.5) & (g > 0); z = cc[sel, 2]; d = np.hypot(d1 + d2, z - ht); ex = -10 * np.log10(g[sel] / (LAM / (4 * np.pi * d)) ** 2); o = np.argsort(-z)
    return dict(path=rows, rm_z=z[o].tolist(), rm_solver=ex[o].tolist(), rm_exact={k: v[o].tolist() for k, v in exact(z).items()})


res = {}
V = [(0, -400, 0), (0, 400, 0), (0, 400, -400), (0, -400, -400)]; sheet = ply("sheet", V, [(0, 1, 2), (0, 2, 3)])
print("cube vertices", mi.load_dict({"type": "cube"}).vertex_count(), "sheet vertices", len(V), flush=True); res["cube_vertex_count"] = int(mi.load_dict({"type": "cube"}).vertex_count())
for name, body in (("sheet", plyshape(sheet)), ("cube20cm", cubeshape(0.1)), ("cube1cm", cubeshape(0.005))):
    res[name] = {}
    for pol in ("V", "H"):
        sc = scene([(body, "m")], dict(m=("metal", 0.01))); res[name][pol] = run_screen(sc, pol, res)
        r = res[name][pol]; print(name, pol, [(x["z"], round(x["solver"], 2), round(x["vhard" if pol == "V" else "vsoft"], 2), round(x["hhard" if pol == "V" else "hsoft"], 2), x["paths"]) for x in r["path"]], flush=True)

# roof rim. Block x in [-60, 0], y in [-300, 300], z in [0, 20]; user 7 m from the wall at 1.5 m; source 1000 m above the roof at elevation e over the block.
X0, Y0, H = -60.0, 300.0, 20.0; v = [(0, -Y0, 0), (0, Y0, 0), (0, Y0, H), (0, -Y0, H), (X0, -Y0, 0), (X0, Y0, 0), (X0, Y0, H), (X0, -Y0, H)]
wq = [(0, 1, 2, 3), (1, 5, 6, 2), (5, 4, 7, 6), (4, 0, 3, 7)]; rq = [(3, 2, 6, 7)]
tri = lambda Q: [t for a, b, c, d in Q for t in ((a, b, c), (a, c, d))]
weld = ply("block_welded", v, tri(wq + rq)); wall = ply("block_walls", v, tri(wq)); roof = ply("block_roof", v, tri(rq))
U = np.array([7.0, 0.0, 1.5]); ELV = np.array([15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 48.0]); PL = H + 1000.0
CFG = dict(welded_concrete=[(plyshape(weld), "c")], split_concrete=[(plyshape(wall), "c"), (plyshape(roof), "c")], split_concrete_metalroof=[(plyshape(wall), "c"), (plyshape(roof), "m")])
res["rim"] = dict(els=ELV.tolist())
for nm, sh in CFG.items():
    res["rim"][nm] = {}
    for pol in ("V", "H"):
        sc = scene(sh, dict(c=("concrete", 0.3), m=("metal", 0.01))); arrays(sc, pol); ex, np_ = [], []
        for e in ELV:
            dz = PL - U[2]; rx = U + np.array([-dz / np.tan(np.radians(e)), 0.0, dz]); g, n, _ = path_gain(sc, U, rx); d = np.linalg.norm(rx - U); ex.append(-10 * np.log10(max(g, 1e-30) / (LAM / (4 * np.pi * d)) ** 2)); np_.append(n)
        sc.add(rt.Transmitter("tx", position=[float(x) for x in U]))
        W = 7000.0; cell = 20.0; rm = rt.RadioMapSolver()(sc, center=[-W / 2, 0.0, PL], orientation=[0.0, 0.0, 0.0], size=[W, 200.0], cell_size=[cell, cell], samples_per_tx=2 * 10 ** 8, max_depth=1, los=True, specular_reflection=False, diffraction=True, edge_diffraction=True, refraction=False)
        g = np.array(rm.path_gain).reshape(-1); cc = np.array(rm.cell_centers).reshape(-1, 3); sc.remove("tx"); ex_rm = []
        for e in ELV:
            dz = PL - U[2]; rx = U + np.array([-dz / np.tan(np.radians(e)), 0.0, dz]); dd = np.abs(cc[:, 0] - rx[0]) + np.abs(cc[:, 1]); near = np.argsort(dd)[:9]; d = np.linalg.norm(rx - U); gv = g[near]; ex_rm.append(float(-10 * np.log10(max(gv.mean(), 1e-30) / (LAM / (4 * np.pi * d)) ** 2)))
        res["rim"][nm][pol] = dict(path=ex, paths=np_, rm=ex_rm); print(nm, pol, np.round(ex, 2), np_, "rm", np.round(ex_rm, 2), flush=True)
json.dump(res, open(os.path.join(OUT, "validate_screen3.json"), "w"), indent=1)
print("VS3DONE", flush=True)
