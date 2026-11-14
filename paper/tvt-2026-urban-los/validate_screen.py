"""Canonical check of Sionna RT edge diffraction: a thin metal screen (a 0.2 m thick box, so that its top edge is a wedge) around x = 0 with its
edge at z = 0. A transmitter at (-d1, 0, h_t) and receivers at (d2, 0, z) from the lit region into deep shadow;
the excess loss over free space is compared with the knife-edge J(nu) of ITU-R P.526.
Writes out/validate_screen.json."""
import json, os
import numpy as np
import sionna.rt as rt

HERE = os.path.dirname(os.path.abspath(__file__))
XML = """<scene version="2.1.0">
  <bsdf type="itu-radio-material" id="mat-metal"><string name="type" value="metal"/><float name="thickness" value="0.01"/></bsdf>
  <shape type="cube" id="screen">
    <transform name="to_world"><scale x="0.1" y="400" z="200"/><translate x="0" y="0" z="-200"/></transform>
    <ref id="mat-metal" name="bsdf"/>
  </shape>
</scene>"""


def J(nu):
    return np.where(nu > -0.78, 6.9 + 20 * np.log10(np.sqrt((nu - 0.1) ** 2 + 1) + nu - 0.1), 0.0)


def main(f=2.0e9, d1=300.0, d2=50.0, ht=20.0):
    """Radio map on the vertical plane x = d2 behind the screen (the path solver returned no diffracted paths
    into the shadow in Sionna RT 2.0.1, the radio map solver does once the edge is illuminated)."""
    lam = 299792458.0 / f
    sc = rt.load_scene_from_string(XML)
    sc.frequency = f
    sc.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    sc.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    sc.add(rt.Transmitter("tx", position=[-d1, 0.0, ht]))
    rm = rt.RadioMapSolver()(sc, center=[d2, 0.0, -15.0], orientation=[0.0, float(np.pi / 2), 0.0], size=[60.0, 20.0],
                             cell_size=[2.0, 2.0], samples_per_tx=10 ** 8, max_depth=1, los=True, specular_reflection=False,
                             diffraction=True, edge_diffraction=True, refraction=False)
    g = np.array(rm.path_gain).reshape(-1); cc = np.array(rm.cell_centers).reshape(-1, 3)
    sel = np.abs(cc[:, 1]) < 1.5
    z = cc[sel, 2]; gz = g[sel]; o = np.argsort(-z); z, gz = z[o], gz[o]
    d = np.hypot(d1 + d2, z - ht); fs = (lam / (4 * np.pi * d)) ** 2
    ex = -10 * np.log10(np.maximum(gz, 1e-30) / fs)
    h = -(ht + (z - ht) * d1 / (d1 + d2)); nu = h * np.sqrt(2 * (d1 + d2) / (lam * d1 * d2))
    res = dict(z=z.tolist(), nu=nu.tolist(), sionna_db=ex.tolist(), knife_db=J(nu).tolist())
    json.dump(res, open(os.path.join(HERE, "out", "validate_screen.json"), "w"), indent=1)
    shadow = nu > 1
    print("deep shadow (nu > 1): Sionna minus knife edge, median %.1f dB, range %.1f to %.1f"
          % (np.median(ex[shadow] - J(nu)[shadow]), np.min(ex[shadow] - J(nu)[shadow]), np.max(ex[shadow] - J(nu)[shadow])))


if __name__ == "__main__":
    main()
