"""Sky maps by reciprocity: a transmitter at the user, a hemispherical measurement mesh of radius R around
it; the path gain on each triangle, divided by the free-space gain over R, is the excess loss toward that
direction (elevation, azimuth)."""
import numpy as np, mitsuba as mi, drjit as dr
import sionna.rt as rt

INWARD = False


def hemisphere(center, R, n_el=18, n_az=72, el_min=5.0):
    """Triangle mesh of the upper hemisphere (elevation el_min..90 deg) as a Mitsuba mesh, plus the
    elevation and azimuth (deg) of each triangle's centroid."""
    els = np.radians(np.linspace(el_min, 90.0, n_el + 1))
    azs = np.radians(np.linspace(0.0, 360.0, n_az + 1))
    V, idx = [], {}
    for i, e in enumerate(els):
        for j, a in enumerate(azs[:-1]):
            idx[(i, j)] = len(V)
            V.append([center[0] + R * np.cos(e) * np.sin(a), center[1] + R * np.cos(e) * np.cos(a), center[2] + R * np.sin(e)])
    F = []
    for i in range(n_el):
        for j in range(n_az):
            j2 = (j + 1) % n_az
            a_, b_, c_, d_ = idx[(i, j)], idx[(i, j2)], idx[(i + 1, j)], idx[(i + 1, j2)]
            F += ([[a_, c_, b_], [b_, c_, d_]] if INWARD else [[a_, b_, c_], [b_, d_, c_]])
    V, F = np.array(V, np.float32), np.array(F, np.int32)
    import os, tempfile
    fd, path = tempfile.mkstemp(suffix=".ply"); os.close(fd)
    with open(path, "w") as fh:   # ASCII PLY, loaded through Sionna so the mesh is complete
        fh.write(f"ply\nformat ascii 1.0\nelement vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\n"
                 f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n")
        for v in V:
            fh.write(f"{v[0]} {v[1]} {v[2]}\n")
        for f in F:
            fh.write(f"3 {f[0]} {f[1]} {f[2]}\n")
    m = rt.load_mesh(path)
    os.unlink(path)
    cen = V[F].mean(axis=1) - np.array(center, np.float32)
    el = np.degrees(np.arcsin(cen[:, 2] / np.linalg.norm(cen, axis=1)))
    az = np.degrees(np.arctan2(cen[:, 0], cen[:, 1])) % 360
    return m, el, az
