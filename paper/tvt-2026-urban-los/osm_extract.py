#!/usr/bin/env python3
"""Stdlib only, run with python3 -I: reduce a downloaded Overpass JSON to plain ways (lon, lat, kind), writes <out>/osm_ways_<city>.json."""
import json, sys
dl, out = sys.argv[1], sys.argv[2]
for c in ("munich", "etoile", "florence", "san_francisco"):
    E = json.load(open(f"{dl}/{c}.json"))["elements"]; W = []
    for e in E:
        g = e.get("geometry"); t = e.get("tags", {})
        if not g:
            continue
        W.append(dict(p=[[float(q["lon"]), float(q["lat"])] for q in g], b=1 if "building" in t else 0, h=str(t.get("highway", "")), u=str(t.get("tunnel", "no")), l=str(t.get("layer", "0"))))
    json.dump(W, open(f"{out}/osm_ways_{c}.json", "w")); print(c, len(W))
