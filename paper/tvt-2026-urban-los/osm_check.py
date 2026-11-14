#!/usr/bin/env python3
"""Registration check: IoU of the scene's building mask and the registered OSM building mask; share of drivable centerline points that fall inside scene buildings."""
import json
import numpy as np
from PIL import Image, ImageDraw
CITIES = {"munich": (48.137, 11.578), "etoile": (48.874, 2.295), "florence": (43.773, 11.256), "san_francisco": (37.8, -122.41)}
R = json.load(open("out/osm_register.json")); res = {}
for c, (la0, lo0) in CITIES.items():
    r = np.load(f"out/raster_{c}.npz"); S = (r["z"] - r["g"]) > 2; x0, y0 = float(r["x0"]), float(r["y0"]); ox, oy = R[c]["offset"]; kx = 111320.0 * np.cos(np.radians(la0)); ky = 110540.0
    im = Image.new("L", (S.shape[1], S.shape[0])); dr = ImageDraw.Draw(im)
    for w in json.load(open(f"out/osm_ways_{c}.json")):
        if w["b"] and len(w["p"]) >= 3:
            dr.polygon([(kx * (p[0] - lo0) - ox - x0, ky * (p[1] - la0) - oy - y0) for p in w["p"]], fill=1)
    M = np.asarray(im, bool); iou = float((M & S).sum() / max((M | S).sum(), 1))
    q = np.load(f"out/osm_roads_{c}.npz"); m = q["mid"][q["drive"]]; i = np.rint((m[:, 1] - y0)).astype(int); j = np.rint((m[:, 0] - x0)).astype(int); ok = (i >= 0) & (i < S.shape[0]) & (j >= 0) & (j < S.shape[1])
    res[c] = dict(iou=iou, drive_in_building=float(S[i[ok], j[ok]].mean()), built_share=float(S.mean())); print(c, res[c])
json.dump(res, open("out/osm_check.json", "w"), indent=1)
