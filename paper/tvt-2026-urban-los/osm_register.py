#!/usr/bin/env python3
"""R5: road axes from OpenStreetMap, registered to the Sionna scene by correlating the OSM building mask with the scene's building
mask (translation, small rotation and scale scan). Reads the plain ways that osm_extract.py (python -I) wrote.
Writes out/osm_<city>.npz: per-user road axis (bearing of the nearest drivable or pedestrian centerline, deg 0..180), distance to it, and registration.
    python3 osm_register.py <out_dir>"""
import json, sys
import numpy as np
import scipy.ndimage as ni
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
CITIES = {"munich": (48.137, 11.578), "etoile": (48.874, 2.295), "florence": (43.773, 11.256), "san_francisco": (37.8, -122.41)}
ROADS = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "living_street", "pedestrian", "service", "footway", "path", "motorway_link", "primary_link", "secondary_link", "tertiary_link"}
DRIVE = ROADS - {"footway", "path", "service"}


def main(out):
    res = {}
    for c, (la0, lo0) in CITIES.items():
        D = json.load(open(f"{out}/osm_ways_{c}.json")); kx = 111320.0 * np.cos(np.radians(la0)); ky = 110540.0
        r = np.load(f"{out}/raster_{c}.npz"); z, g = r["z"], r["g"]; x0, y0 = float(r["x0"]), float(r["y0"]); ny, nx = z.shape; S = (z - g) > 2
        S2 = ni.zoom(S.astype(np.float32), 0.5, order=1); blds, rds = [], []
        for e in D:
            P = np.array([[kx * (p[0] - lo0), ky * (p[1] - la0)] for p in e["p"]])
            if e["b"] and len(P) >= 3:
                blds.append(P)
            elif e["h"] in ROADS and len(P) >= 2 and e["u"] == "no" and e["l"] in ("0", ""):
                rds.append((P, e["h"]))
        from PIL import Image, ImageDraw
        best = None
        ext = 3000
        for rot, sc in [(r_, 1.0) for r_ in np.arange(-2.0, 2.01, 0.5)] + [(0.0, s_) for s_ in (0.98, 0.99, 1.01, 1.02)]:
            if True:
                cs, sn_ = np.cos(np.radians(rot)), np.sin(np.radians(rot))
                im = Image.new("L", (ext, ext)); dr = ImageDraw.Draw(im)
                for P in blds:
                    Q = sc * np.c_[cs * P[:, 0] - sn_ * P[:, 1], sn_ * P[:, 0] + cs * P[:, 1]]
                    dr.polygon([(float(a) / 2 + ext / 2, float(b) / 2 + ext / 2) for a, b in Q], fill=1)     # 2 m cells
                M = np.asarray(im, np.float32)
                cc = fftconvolve(M - M.mean(), (S2 - S2.mean())[::-1, ::-1], mode="same")
                pk = np.unravel_index(np.argmax(cc), cc.shape); v = cc[pk] / (M.std() * S2.std() * S2.size)
                if best is None or v > best[0]:
                    best = (float(v), rot, sc, pk)
        v, rot, sc, pk = best
        # peak gives the offset of the scene's center in the OSM raster (row = y, col = x), 2 m cells
        cy = (pk[0] - ext / 2 - 0.5) * 2; cx = (pk[1] - ext / 2 - 0.5) * 2
        ox = cx - (x0 + (nx - 1) / 2); oy = cy - (y0 + (ny - 1) / 2)     # osm = R(sc) lonlat; scene = osm - offset
        cs, sn_ = np.cos(np.radians(rot)), np.sin(np.radians(rot))
        segs = []; mid = []; ang = []; drive = []
        for P, h in rds:
            Q = sc * np.c_[cs * P[:, 0] - sn_ * P[:, 1], sn_ * P[:, 0] + cs * P[:, 1]] - [ox, oy]
            for a, b in zip(Q[:-1], Q[1:]):
                L = np.hypot(*(b - a)); n = max(int(L / 2), 1)
                for t_ in (np.arange(n) + 0.5) / n:
                    mid.append(a + t_ * (b - a)); ang.append(np.degrees(np.arctan2(b[0] - a[0], b[1] - a[1])) % 180); drive.append(h in DRIVE)
        mid = np.array(mid); ang = np.array(ang); drive = np.array(drive)
        T_all = cKDTree(mid); T_d = cKDTree(mid[drive])
        out_ = dict(corr=v, rot=rot, scale=sc, offset=[ox, oy])
        for tag in ("", "_c03sd2"):
            u = np.load(f"{out}/sky2_{c}{tag}.npz")["users"][:, :2]
            d1, i1 = T_all.query(u); d2, i2 = T_d.query(u)
            np.savez_compressed(f"{out}/osm_{c}{tag}.npz", road_any=np.rint(ang[i1] / 5) * 5 % 180, dist_any=d1, road_drive=np.rint(ang[drive][i2] / 5) * 5 % 180, dist_drive=d2)
            out_[f"dist_med{tag}"] = [float(np.median(d1)), float(np.median(d2))]
        np.savez_compressed(f"{out}/osm_roads_{c}.npz", mid=mid, ang=ang, drive=drive)
        res[c] = out_; print(c, out_, "buildings", len(blds), "roads", len(rds), flush=True)
    json.dump(res, open(f"{out}/osm_register.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
