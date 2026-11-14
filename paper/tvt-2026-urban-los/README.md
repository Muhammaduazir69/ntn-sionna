# Code and data: ray-traced line of sight and excess loss for direct-to-cell satellites in cities

Companion material for M. Uzair, "What Street Geometry Adds to Elevation Tables: Ray-Traced Line of Sight and Excess Loss for Direct-to-Cell Satellites in Cities," 2026.

The study traces sky maps with Sionna RT over OpenStreetMap city scenes, compares the line-of-sight share and excess loss with elevation-only tables, and tests satellite-selection rules that use street geometry or measured signal.

| Path | Contents |
|---|---|
| `osm_*.py`, `raster_scene.py`, `street_geom.py` | City scenes from OpenStreetMap and street geometry |
| `sky_maps*.py`, `pol_paths.py`, `run_pol.sh`, `run_round4.sh`, `sweep_c2.sh` | Sionna RT sky-map tracing |
| `sat_select*.py`, `selection_floor.py` | Satellite-selection rules |
| `analysis*.py`, `model_fit.py`, `brier_ladder.py`, `validate_*.py`, `knife_edge.py`, `canyon_bound.py` | Statistics, model fits and validation |
| `out.tar.xz.part-*` | All result files, including the sky maps |
| `make_macros*.py`, `make_figs*.py`, `README_runs.md` | Numbers, figures and run notes |

Large result folders are stored as xz archives. Where an archive is split into `*.part-NN` files, join and check it first: `cat out.tar.xz.part-* > out.tar.xz && sha256sum -c out.tar.xz.sha256 && tar -xJf out.tar.xz`.

Contact: Muhammad Uzair, muhammaduzairr69@gmail.com, ORCID 0009-0002-4104-2680. License: as the repository (GPL-2.0-only for code).
