# Reproducing the data of the TVT manuscript (Sionna RT 2.0.1, RTX 3080)

Main maps: `sky_maps2.py <city> 300 --planeOff 1000 --cell 2.5 --samples 1e9` (tag empty, marble as shipped, 2 GHz, seed 1);
rough: `--scatter 0.4 --tag _sc4`. Geometry: `street_geom.py <city> sky2 street2`. Round 4: `run_round4.sh` (concrete, rough concrete,
second seed), `sweep_c2.sh` (Munich 60-user facade sweep), `run_pol.sh` (per-path polarization), `raster_scene.py`, `street_null.py` (controls),
`analysis2.py`, `analysis2.py _sc4`, `analysis2.py _c03`, `analysis3.py`, `selection_floor.py`, `sat_select.py _c03`, `loss_summary.py`, `conv_summary.py`, `conv_bands.py`,
`validate_screen3.py` (one-sheet half plane, cube, split and welded rim; replaces validate_screen.py and validate_screen2.py, whose cube is not a half plane), `validate_diffuse.py`, `canyon_bound.py`, `make_macros2.py`, `make_figs2.py`, `../figure_src/make_fig1.py`.
File manifest: out/*.npz carry their parameters in the `params` entry (scripts after 30 Oct); earlier main maps were produced by the commands above and their first 60 users equal `_sbase`.
The Schreiber et al. coefficients in analysis3.py are Tables II and VIII of arXiv:2608.28331, S band; the P.681 measured parameters in loss_summary.py are from Annex 2, 2.2 GHz urban, of Rec. ITU-R P.681-11 (research/p681-11_full_text.txt).

Round R3: `street_null2.py` (5000 bootstrap draws, add-one p, strata), `sat_select2.py _c03` and `_c03sd2` (adds the marginal map rule, eq. marg; earlier rows are unchanged because new rules use separate random streams), `validate_screen3.py`. OSM provenance: out/osm_query.md. Previous outputs: out/pre_r3/.

Round R4: `sat_select_facade.py first` and `second` (same streams as sat_select2.main; runs the rules on _c03, _c03s, marble, _sc4, and _c03sd2; make_macros2.py asserts that its _c03 and _c03sd2 rows equal the stored sat_select2 outputs; paired user-bootstrap intervals; own-curve draws for the sparse shell), `osm_shift.py` (registration sensitivity). OMP_NUM_THREADS=1, about 50 min each.
