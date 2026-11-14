#!/usr/bin/env python3
"""Write ../macros_p9.tex from the version-2 outputs: out/analysis2.json, out/conv2.json, out/runs_<city>.json,
out/validate_screen.json. Every number of the manuscript comes from here."""
import json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
TAG = {"munich": "Mun", "etoile": "Par", "florence": "Flo", "san_francisco": "Sfo"}
FLAT = ("munich", "etoile", "florence")
A = json.load(open(os.path.join(OUT, "analysis2.json")))
C = json.load(open(os.path.join(OUT, "conv2.json")))
M = {}


def f(x, d=1):
    return f"{x:.{d}f}"


def pts(x):
    return f(100 * x, 0)


els = np.array(A["munich"]["els"])
idx = {e: list(els).index(e) for e in (15.0, 20.0, 30.0, 45.0, 60.0)}
M["NUsers"] = str(A["munich"]["n_users"])
for c, t in TAG.items():
    r = A[c]
    for e, nm in ((15.0, "Fifteen"), (30.0, "Thirty"), (60.0, "Sixty")):
        M[f"Los{t}E{nm}"] = pts(r["los"][idx[e]])
    M[f"TrU{t}"], M[f"TrD{t}"] = pts(r["tr_urban_max"]), pts(r["tr_dense_urban_max"])
    M[f"TrUMean{t}"], M[f"TrDMean{t}"] = pts(r["tr_urban_mean"]), pts(r["tr_dense_urban_mean"])
    M[f"StatIn{t}"], M[f"StatSplit{t}"] = pts(r["stat_in_max"]), pts(r["stat_split_max"])
    M[f"StatCrossLo{t}"], M[f"StatCrossHi{t}"] = pts(min(r["stat_cross"].values())), pts(max(r["stat_cross"].values()))
    M[f"Canyon{t}"] = pts(r["canyon_max"])
    M[f"RuleOne{t}"], M[f"RuleTwo{t}"] = f(100 * r["rule1_acc"], 1), f(100 * r["rule2_acc"], 1)
    M[f"RuleNear{t}"], M[f"RuleNlos{t}"] = f(100 * r["rule2_acc_near"], 1), f(100 * r["rule2_acc_nlos"], 1)
    M[f"Sky{t}"] = pts(max(r["skyline_share"])); M[f"Dep{t}"] = pts(max(np.abs(r["dependence"])))
    for nm in ("ax1", "ax2", "street"):
        k = {"ax1": "One", "ax2": "Two", "street": "Street"}[nm]
        M[f"Along{k}{t}"], M[f"Across{k}{t}"] = pts(r[f"along_{nm}"]), pts(r[f"across_{nm}"])
        M[f"AlongCi{k}{t}"] = f"{100 * r[f'along_{nm}_ci'][0]:.0f}--{100 * r[f'along_{nm}_ci'][1]:.0f}"
        M[f"AcrossCi{k}{t}"] = f"{100 * r[f'across_{nm}_ci'][0]:.0f}--{100 * r[f'across_{nm}_ci'][1]:.0f}"
    M[f"StreetShare{t}"] = pts(r["street_user_share"])
    nl = np.array([np.nan if x is None else x for x in r["nlos_med"]])
    nc = np.array([np.nan if x is None else x for x in r["nlos_med_cens"]])
    M[f"NlosLo{t}"], M[f"NlosHi{t}"] = f(np.nanmin(nl), 0), f(np.nanmax(nl), 0)
    fin = nc[np.isfinite(nc)]
    M[f"NlosCensLo{t}"], M[f"NlosCensHi{t}"] = f(np.min(fin), 0), f(np.max(fin), 0)
    u = np.array([np.nan if x is None else x for x in r["unreached"]])
    M[f"Unr{t}"] = f(100 * np.nanmean(u), 1)
    M[f"Coverage{t}"] = pts(r["law_coverage"]); M[f"SkyBlk{t}"] = pts(r["skyline_blocked_share"])
    M[f"CiHw{t}"] = f(100 * r["los_ci_hw_max"], 1)
    lo = A["loss_loo"][c]
    for k in ("mae_law", "mae_const", "mae_el", "mae_knife", "mae_knife_off"):
        M[f"{''.join(w.capitalize() for w in k.split('_'))}{t}"] = f(lo[k], 1)
    M[f"PinLaw{t}"], M[f"PinConst{t}"] = f(lo["pin_law"], 2), f(lo["pin_const"], 2)
    M[f"Cover{t}"] = pts(lo["cover90"]); M[f"KnifeBias{t}"] = f(lo["knife_bias"], 0)
    M[f"Binned{t}"] = f(lo["binned_mean_err"], 1)
    # TR clutter loss comparison: elevations where the censored median exceeds the TR value
    tr = np.array(r["tr_cl"]); M[f"CensAboveTr{t}"] = str(int(np.sum(np.isfinite(nc) & (nc > tr)) + np.sum(~np.isfinite(nc) & ~np.isnan(nc))))
    rp = os.path.join(OUT, f"runs_{c}.json")
    if os.path.exists(rp):
        R = json.load(open(rp)); M[f"Walks{t}"] = str(R["n_walks"])
        for rel, nm in ((0, "Zero"), (30, "Thirty"), (60, "Sixty"), (90, "Ninety")):
            v = R[f"30_{rel}"]
            M[f"RunShare{nm}{t}"] = pts(v["los_share"]) if v["los_share"] is not None else "--"
            M[f"RunLos{nm}{t}"] = f(v["mean_los"], 0) if v["mean_los"] else "--"
            M[f"RunNlos{nm}{t}"] = f(v["mean_nlos"], 0) if v["mean_nlos"] else "--"
allc = list(TAG)
M["TrBestMeanLo"] = pts(min(min(A[c]["tr_urban_mean"], A[c]["tr_dense_urban_mean"]) for c in allc))
M["TrBestMeanHi"] = pts(max(min(A[c]["tr_urban_mean"], A[c]["tr_dense_urban_mean"]) for c in allc))
M["TrBestMaxLo"] = pts(min(min(A[c]["tr_urban_max"], A[c]["tr_dense_urban_max"]) for c in allc))
M["TrBestMaxHi"] = pts(max(min(A[c]["tr_urban_max"], A[c]["tr_dense_urban_max"]) for c in allc))
M["RuleTwoMin"] = f(100 * min(A[c]["rule2_acc"] for c in allc), 1)
M["RuleNearMin"] = f(100 * min(A[c]["rule2_acc_near"] for c in allc), 1)
M["StatSplitFlatHi"] = pts(max(A[c]["stat_split_max"] for c in FLAT))
M["StatInFlatHi"] = pts(max(A[c]["stat_in_max"] for c in FLAT))
M["CiHwMax"] = f(100 * max(A[c]["los_ci_hw_max"] for c in allc), 1)
M["LossKzero"], M["LossKone"] = f(A["loss_fit_all"]["k0"], 1), f(A["loss_fit_all"]["k1"], 1)
M["LossRsq"] = f(A["loss_fit_all"]["r2"], 2)
q10, q90 = A["loss_fit_all"]["q10"], A["loss_fit_all"]["q90"]
M["LossQtenKzero"], M["LossQtenKone"], M["LossQninetyKzero"], M["LossQninetyKone"] = (f(x, 1) for x in (*q10, *q90))
for k in ("mae_law", "mae_const", "mae_el"):
    v = [A["loss_loo"][c][k] for c in allc]
    nm = "".join(w.capitalize() for w in k.split("_"))
    M[f"{nm}Lo"], M[f"{nm}Hi"] = f(min(v), 1), f(max(v), 1)
kb = [A["loss_loo"][c]["knife_bias"] for c in allc]
M["KnifeBiasLo"], M["KnifeBiasHi"] = f(min(kb), 0), f(max(kb), 0)
allnl = [x for c in allc for x in A[c]["nlos_med"] if x is not None]
M["NlosAllLo"], M["NlosAllHi"] = f(min(allnl), 0), f(max(allnl), 0)
M["UnrLo"], M["UnrHi"] = f(min(float(M[f"Unr{t}"]) for t in TAG.values()), 1), f(max(float(M[f"Unr{t}"]) for t in TAG.values()), 1)
# convergence and sensitivity (Munich, 60 users)
for tag, nm in (("_po30", "PoThirty"), ("_po100", "PoHundred"), ("_po300", "PoThreeH"), ("_po1000", "PoThousand"),
                ("_po3000", "PoThreeK"), ("_po1000c5", "CellFive"), ("_po1000c25", "CellTwoFive"), ("_po1000c25s10", "CellTwoFiveRays"),
                ("_po1000c125", "CellOneTwoFive"), ("_sbase", "Base"), ("_sd6", "DepthSix"), ("_ssc", "Scatter"),
                ("_snr", "NoReflect"), ("_sph", "PolH"), ("_sf16", "FreqL"), ("_sf35", "FreqC")):
    if tag in C:
        v = C[tag]
        M[f"Cv{nm}Med"], M[f"Cv{nm}Cens"], M[f"Cv{nm}Unr"] = f(v["med"], 1), f(v["med_cens"], 1), f(100 * v["unreached"], 1)
        M[f"Cv{nm}Dist"] = f(v["plane_dist"], 0)
M["LosNormMed"] = f(C["_sbase"]["los_med"], 1)
M["LosNormPone"], M["LosNormPnn"] = f(C["_sbase"]["los_p1"], 1), f(C["_sbase"]["los_p99"], 1)
with open(os.path.join(HERE, "..", "macros_p9.tex"), "w") as fh:
    fh.write("% generated by sim/make_macros2.py; do not edit\n")
    for k, v in sorted(M.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(M), "macros")

# ---- round 3: baselines, azimuth model, loss in both facade settings, convergence by elevation band ----
S2 = json.load(open(os.path.join(OUT, "analysis2_sc4.json")))
CB = json.load(open(os.path.join(OUT, "conv_bands.json")))
M3 = {}
for c, t in TAG.items():
    r, rs = A[c], S2[c]
    M3[f"PsixMax{t}"], M3[f"PsixMean{t}"] = pts(r["p681_max"]), pts(r["p681_mean"])
    M3[f"PsixAlong{t}"], M3[f"PsixAcross{t}"] = pts(r["p681_along"]), pts(r["p681_across"])
    M3[f"FfAlong{t}"], M3[f"FfAcross{t}"] = pts(r["ff_along"]), pts(r["ff_across"])
    M3[f"FfSplit{t}"] = pts(r["ff_split_err_max"])
    M3[f"ItuMax{t}"] = pts(r["itu_max"])
    M3[f"SwingAz{t}"] = pts(r["along_ax2"] - r["across_ax2"]); M3[f"SwingEl{t}"] = pts(r["swing_el_all"])
    M3[f"TrCovSmooth{t}"], M3[f"TrCovDiff{t}"] = pts(r["tr_cover_dense_urban"]), pts(rs["tr_cover_dense_urban"])
    M3[f"TrBelowSmooth{t}"], M3[f"TrBelowDiff{t}"] = pts(r["tr_below_dense_urban"]), pts(rs["tr_below_dense_urban"])
    md = [x for x in rs["nlos_med"] if x is not None]
    M3[f"NlosDiffLo{t}"], M3[f"NlosDiffHi{t}"] = f(min(md), 0), f(max(md), 0)
    lo, ls = A["loss_loo"][c], S2["loss_loo"][c]
    M3[f"GainTr{t}"], M3[f"GainTrDiff{t}"] = f(lo["gain_tr"], 1), f(ls["gain_tr"], 1)
    M3[f"GainConst{t}"], M3[f"GainConstDiff{t}"] = f(lo["gain_const"], 1), f(ls["gain_const"], 1)
    M3[f"GainKnife{t}"] = f(lo["gain_knife_off"], 1)
    M3[f"CoverCens{t}"], M3[f"CoverDiff{t}"] = pts(lo["cover90"]), pts(ls["cover90"])
    M3[f"StreetW{t}"] = pts(r["street_user_share"])
allc = list(TAG)
M3["SwingAzLo"], M3["SwingAzHi"] = pts(min(A[c]["along_ax2"] - A[c]["across_ax2"] for c in allc)), pts(max(A[c]["along_ax2"] - A[c]["across_ax2"] for c in allc))
M3["SwingElLo"], M3["SwingElHi"] = pts(min(A[c]["swing_el_all"] for c in allc)), pts(max(A[c]["swing_el_all"] for c in allc))
M3["PsixMaxLo"], M3["PsixMaxHi"] = pts(min(A[c]["p681_max"] for c in allc)), pts(max(A[c]["p681_max"] for c in allc))
M3["PsixAlongOverLo"] = pts(min(A[c]["p681_along"] - A[c]["along_ax2"] for c in allc))
M3["PsixAlongOverHi"] = pts(max(A[c]["p681_along"] - A[c]["along_ax2"] for c in allc))
M3["FfErrMaxFlat"] = pts(max(max(abs(A[c]["ff_along"] - A[c]["along_ax2"]), abs(A[c]["ff_across"] - A[c]["across_ax2"])) for c in FLAT))
M3["FfSplitLo"], M3["FfSplitHi"] = pts(min(A[c]["ff_split_err_max"] for c in allc)), pts(max(A[c]["ff_split_err_max"] for c in allc))
M3["ItuMaxLo"], M3["ItuMaxHi"] = pts(min(A[c]["itu_max"] for c in allc)), pts(max(A[c]["itu_max"] for c in allc))
M3["TrBelowLo"] = pts(min(min(A[c]["tr_below_dense_urban"], S2[c]["tr_below_dense_urban"]) for c in allc))
M3["TrBelowHi"] = pts(max(max(A[c]["tr_below_dense_urban"], S2[c]["tr_below_dense_urban"]) for c in allc))
M3["TrCovLo"] = pts(min(min(A[c]["tr_cover_dense_urban"], S2[c]["tr_cover_dense_urban"]) for c in allc))
M3["TrCovHi"] = pts(max(max(A[c]["tr_cover_dense_urban"], S2[c]["tr_cover_dense_urban"]) for c in allc))
mdd = [x for c in allc for x in S2[c]["nlos_med"] if x is not None]
M3["NlosDiffAllLo"], M3["NlosDiffAllHi"] = f(min(mdd), 0), f(max(mdd), 0)
p90d = [x for c in allc for x in S2[c]["nlos_p90"] if x is not None and np.isfinite(x)]
M3["PninetyDiffLo"], M3["PninetyDiffHi"] = f(min(p90d), 0), f(max(p90d), 0)
M3["LossKoneDiff"], M3["LossKzeroDiff"] = f(S2["loss_fit_all"]["k1"], 1), f(S2["loss_fit_all"]["k0"], 1)
M3["GainTrLo"] = f(min(min(A["loss_loo"][c]["gain_tr"], S2["loss_loo"][c]["gain_tr"]) for c in allc), 1)
M3["GainTrHi"] = f(max(max(A["loss_loo"][c]["gain_tr"], S2["loss_loo"][c]["gain_tr"]) for c in allc), 1)
for tag, nm in (("_sbase", "Base"), ("_po3000u", "PlaneK"), ("_c056r4e9", "CellHalf"), ("_scp", "Circ")):
    for k, b in enumerate(("Low", "Mid", "High")):
        M3[f"Band{nm}{b}"] = f(CB[tag]["bands"][k][0], 1)
    M3[f"BandUnr{nm}"] = f(100 * CB[tag]["bands"][0][1], 0)
with open(os.path.join(HERE, "..", "macros_p9.tex"), "a") as fh:
    for k, v in sorted(M3.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(M3), "round-3 macros")
M4 = {}
M4["GainTrSmoothLo"] = f(min(A["loss_loo"][c]["gain_tr"] for c in allc), 1)
M4["GainTrSmoothHi"] = f(max(A["loss_loo"][c]["gain_tr"] for c in allc), 1)
M4["GainTrRoughLo"] = f(min(S2["loss_loo"][c]["gain_tr"] for c in allc), 1)
M4["GainTrRoughHi"] = f(max(S2["loss_loo"][c]["gain_tr"] for c in allc), 1)
M4["GainConstLo"] = f(min(A["loss_loo"][c]["gain_const"] for c in allc), 1)
M4["GainConstHi"] = f(max(A["loss_loo"][c]["gain_const"] for c in allc), 1)
M4["FfSplitFlatHi"] = pts(max(A[c]["ff_split_err_max"] for c in FLAT))
with open(os.path.join(HERE, "..", "macros_p9.tex"), "a") as fh:
    for k, v in sorted(M4.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---- round 4 (Nov 2026): street-free controls, map levels, shell, facade sweep, polarization ----
import glob
SN = json.load(open(os.path.join(OUT, "street_null.json"))); A3 = json.load(open(os.path.join(OUT, "analysis3.json")))
FL = json.load(open(os.path.join(OUT, "selection_floor.json"))); SS = json.load(open(os.path.join(OUT, "sat_select_c03.json")))
LS = json.load(open(os.path.join(OUT, "loss_summary.json"))); CB2 = json.load(open(os.path.join(OUT, "conv_bands.json")))
CD = json.load(open(os.path.join(OUT, "validate_diffuse.json"))); CY = json.load(open(os.path.join(OUT, "canyon_bound.json")))
M5 = {}
def nullmean(c, ctl, key):
    return float(np.mean([s[key] for s in SN[c][ctl]]))
def rng2(name, vals, fmt=pts):
    M5[f"{name}Lo"], M5[f"{name}Hi"] = fmt(min(vals)), fmt(max(vals))
for c, t in TAG.items():
    s = SN[c]; r3 = A3[c]
    M5[f"SwingOwn{t}"] = pts(s["real"]["own"]); M5[f"SwingCf{t}"] = pts(s["real"]["cf"])
    M5[f"SwingOwnCi{t}"] = "%s--%s" % (pts(s["real_ci"]["own"][0]), pts(s["real_ci"]["own"][1]))
    M5[f"NullRotOwn{t}"] = pts(nullmean(c, "rotm", "own")); M5[f"NullIsoOwn{t}"] = pts(nullmean(c, "iso", "own"))
    M5[f"FloorReal{t}"] = pts(FL["table"][c]["real"]["model"])
    M5[f"ExcessRotCf{t}"] = f"{100 * s['diff']['rotm_cf']['mean']:+.0f}".replace("-", "$-$"); M5[f"ExcessIsoCf{t}"] = f"{100 * s['diff']['iso_cf']['mean']:+.0f}".replace("-", "$-$")
    ci = s["diff"]["rotm_cf"]["ci"]; M5[f"CiRotCf{t}"] = (f"{100 * ci[0]:.0f} to {100 * ci[1]:.0f}").replace("-", "$-$")
    M5[f"Agree{t}"] = pts(r3["agree_share"]); M5[f"StreetOwn{t}"] = pts(s["real"]["own_agree"]); M5[f"NonStreetOwn{t}"] = pts(s["real"]["own_non"])
    M5[f"RotStreetOwn{t}"] = pts(nullmean(c, "rotm", "own_agree")); M5[f"RotNonOwn{t}"] = pts(nullmean(c, "rotm", "own_non"))
    M5[f"Harm{t}"] = f(s["real"]["harm"], 2); M5[f"HarmRot{t}"] = f(nullmean(c, "rotm", "harm"), 2); M5[f"HarmIso{t}"] = f(nullmean(c, "iso", "harm"), 2)
    M5[f"Aspect{t}"] = f(r3["aspect"], 2)
    cu = r3["curves"]
    M5[f"SchMean{t}"], M5[f"SchMax{t}"] = pts(cu["schreiber"]["mean"]), pts(cu["schreiber"]["max"])
    M5[f"PsixMean{t}"], M5[f"PsixMax{t}"] = pts(cu["p681"]["mean"]), pts(cu["p681"]["max"])
    M5[f"CapMean{t}"], M5[f"CapMax{t}"] = pts(cu["p681_cap"]["mean"]), pts(cu["p681_cap"]["max"])
    b = r3["brier"]
    for k, v in (("Tr", "tr_best"), ("Sch", "schreiber"), ("Oth", "other_curve"), ("Cf", "axis_cf"), ("Own", "axis_own"), ("Psix", "p681"), ("Foot", "footprint"), ("RuleOne", "rule1"), ("RuleTwo", "rule2")):
        M5[f"Br{k}{t}"] = f(b[v], 3)
    d = SS["dense"][c]; p = d["policies"]; sp = SS["sparse"][c]; q = sp["policies"]
    M5[f"SelHe{t}"], M5[f"SelAzOwn{t}"], M5[f"SelAzCf{t}"], M5[f"SelFoot{t}"], M5[f"SelMap{t}"] = (pts(p[k]["los"]) for k in ("HE", "AZown", "AZcf", "FOOT", "MAP"))
    M5[f"SelMapCi{t}"] = "%s--%s" % (pts(p["MAP"]["los_ci"][0]), pts(p["MAP"]["los_ci"][1]))
    M5[f"PnHe{t}"], M5[f"PnMap{t}"] = f(p["HE"]["p90"], 0), f(p["MAP"]["p90"], 0); M5[f"HoHe{t}"], M5[f"HoMap{t}"] = f(p["HE"]["ho_per_min"], 1), f(p["MAP"]["ho_per_min"], 1)
    M5[f"Any{t}"] = pts(d["any_los"])
    M5[f"SpHe{t}"], M5[f"SpAzOwn{t}"], M5[f"SpMap{t}"] = (pts(q[k]["los"]) for k in ("HE", "AZown", "MAP"))
    M5[f"SpAny{t}"], M5[f"SpTr{t}"] = pts(sp["any_los"]), pts(sp["tr_any"]["dense_urban"])
    for pre, st in (("BMar", "marble"), ("BCon", "concrete"), ("BRou", "concrete_rough")):
        for i_, x in enumerate("ABC"):
            M5[f"{pre}{t}{x}"] = f(LS[st][c]["bands"][i_]["cens"], 1)
allc = list(TAG)
ho_foot = [SS["dense"][c]["policies"]["FOOT"]["ho_per_min"] for c in allc]
rng2("SwingOwn", [SN[c]["real"]["own"] for c in allc]); rng2("SwingCf", [SN[c]["real"]["cf"] for c in allc])
rng2("SwingRotOwn", [nullmean(c, "rotm", "own") for c in allc]); rng2("SwingIsoOwn", [nullmean(c, "iso", "own") for c in allc])
ex = [SN[c]["diff"][f"{k}_own"]["mean"] for c in allc for k in ("rotm", "iso")]; rng2("SwingExcess", ex)
rng2("ExcessIsoCf", [SN[c]["diff"]["iso_cf"]["mean"] for c in allc]); M5["SwingStreetBound"] = pts(max(max(SN[c]["diff"]["iso_cf"]["mean"], SN[c]["diff"]["rotm_own"]["mean"]) for c in allc))
M5["ExcessFloIsoOwn"] = pts(SN["florence"]["diff"]["iso_own"]["mean"]); rng2("SelFootHo", ho_foot, lambda x: f(x, 1))
M5["ExcessSfoRotOwn"] = pts(SN["san_francisco"]["diff"]["rotm_own"]["mean"])
rng2("Agree", [A3[c]["agree_share"] for c in allc]); rng2("StreetShare", [A3[c]["agree_share"] for c in allc]); rng2("StreetOwn", [SN[c]["real"]["own_agree"] for c in allc]); rng2("NonStreetOwn", [SN[c]["real"]["own_non"] for c in allc]); rng2("RotStreetOwn", [nullmean(c, "rotm", "own_agree") for c in allc])
rng2("Raster", [SN[c]["raster_vs_mesh"] for c in allc]); M5["RotScaleLo"], M5["RotScaleHi"] = f(min(SN[c]["rot_scale"] for c in allc), 2), f(max(SN[c]["rot_scale"] for c in allc), 2)
M5["IsoBuiltHi"] = pts(max(nullmean(c, "iso", "built") for c in allc))
M5["FloorK"] = str(FL["K_eff"])
fe = [abs(FL["table"][c][k]["sim"] - FL["table"][c][k]["model"]) for c in allc for k in ("iso", "rotm") if not (c == "munich" and k == "iso")]
M5["FloorCtrlErr"] = pts(max(fe)); M5["FloorErrHi"] = pts(max(abs(FL["table"][c]["real"]["sim"] - FL["table"][c]["real"]["model"]) for c in allc))
br = lambda k: [A3[c]["brier"][k] for c in allc]
f3 = lambda x: f(x, 3); f2 = lambda x: f(x, 2)
rng2("BrierOther", br("other_curve"), f2); rng2("BrierOwn", br("own_curve"), f2); rng2("BrierFoot", br("footprint"), f2); rng2("BrierSky", br("rule2"), f3); rng2("BrierAxis", br("axis_own"), f2)
gain = [A3[c]["brier"]["other_curve"] - A3[c]["brier"]["axis_own"] for c in allc]; rng2("BrierAxisGain", gain, f3)
M5["BrierCfGainHi"] = f3(max(A3[c]["brier"]["other_curve"] - A3[c]["brier"]["axis_cf"] for c in allc))
rng2("Sch", [A3[c]["curves"]["schreiber"]["mean"] for c in allc]); M5["SchMeanLo"], M5["SchMeanHi"] = M5.pop("SchLo"), M5.pop("SchHi")
hb = [A3[c]["hb_mode"] for c in allc]; M5["HbLo"], M5["HbHi"] = f(min(hb), 0), f(max(hb), 0)
oa = [A3[c]["az_own"] for c in allc]
M5["PsixAlongOverLo"], M5["PsixAlongOverHi"] = pts(min(x["p681_along"] - x["along"] for x in oa)), pts(max(x["p681_along"] - x["along"] for x in oa))
M5["PsixAcrossErrHi"] = pts(max(abs(x["p681_across"] - x["across"]) for x in oa))
rng2("CapAlong", [abs(x["cap_along"] - x["along"]) for x in oa])
M5["FfErrHi"] = pts(max(max(abs(x["ff_along"] - x["along"]), abs(x["ff_across"] - x["across"])) for x in oa))
rng2("SwingMesh", [x["along"] - x["across"] for x in oa])
rng2("FfSplit", [A3[c]["split"]["e30_mean_max"] for c in allc]); M5["FfSplitWorst"] = pts(max(A3[c]["split"]["e30_worst"] for c in allc)); rng2("FfSplitAll", [A3[c]["split"]["all_mean"] for c in allc])
sc_ = [v["max"] for c in allc for v in A3[c]["curves"]["stat_cross"].values()]; M5["StatCrossMinAll"], M5["StatCrossMaxAll"] = pts(min(sc_)), pts(max(sc_))
for c, t in TAG.items():
    v_ = [v["max"] for v in A3[c]["curves"]["stat_cross"].values()]; M5[f"StatCrossLo{t}"], M5[f"StatCrossHi{t}"] = pts(min(v_)), pts(max(v_))
nn = [A3[c]["nn_med"] for c in allc if c != "etoile"]; M5["NnLo"], M5["NnHi"] = f(min(nn), 0), f(max(nn), 0); M5["NnPar"] = f(A3["etoile"]["nn_med"], 0)
D_ = SS["dense"]; S_ = SS["sparse"]; pol = lambda k, D=D_: [D[c]["policies"][k]["los"] for c in allc]
rng2("SelHe", pol("HE")); rng2("SelMap", pol("MAP")); rng2("SelFoot", pol("FOOT"))
rng2("SelMapGain", [a - b for a, b in zip(pol("MAP"), pol("HE"))])
M5["SelAxisGainHi"] = pts(max(max(a - b for a, b in zip(pol("AZown", DD), pol("HE", DD))) for DD in (D_, S_)))
M5["SelCfGainHi"] = pts(max(a - b for a, b in zip(pol("AZcf"), pol("HE"))))
pg = [D_[c]["policies"]["HE"]["p90"] - D_[c]["policies"]["MAP"]["p90"] for c in allc]; rng2("SelMapPninety", pg, lambda x: f(x, 0))
ho = lambda k: [D_[c]["policies"][k]["ho_per_min"] for c in allc]; rng2("SelMapHo", ho("MAP"), lambda x: f(x, 1)); rng2("SelHeHo", ho("HE"), lambda x: f(x, 1)); rng2("SelMapsHo", ho("MAPS"), lambda x: f(x, 1))
M5["SelMapsLoss"] = str(int(np.ceil(100 * max(a - b for a, b in zip(pol("MAP"), pol("MAPS"))))))
rng2("SpAny", [S_[c]["any_los"] for c in allc]); rng2("SpTr", [S_[c]["tr_any"]["dense_urban"] for c in allc]); M5["SpTrGapHi"] = pts(max(S_[c]["tr_any"]["dense_urban"] - S_[c]["any_los"] for c in allc))
sv = [S_[c]["mean_visible"] for c in allc]; M5["SpVisLo"], M5["SpVisHi"] = f(min(sv), 0), f(max(sv), 0)
# loss: margins below 60 deg
def margins(st):
    return [LS[st][c]["bands"][i_]["margin"] for c in allc for i_ in (0, 1)]
fd = lambda x: f(x, 0)
rng2("MarginMarble", margins("marble"), fd); rng2("MarginConc", margins("concrete"), fd); rng2("MarginRough", margins("concrete_rough"), fd)
M5["MarginCiHw"] = fd(max((LS[st][c]["bands"][i_]["margin_ci"][1] - LS[st][c]["bands"][i_]["margin_ci"][0]) / 2 for st in ("marble", "concrete", "concrete_rough") for c in allc for i_ in (0, 1)))
M5["MunHighOver"] = f(-LS["concrete"]["munich"]["bands"][2]["margin"], 1)
ur = [LS["concrete"][c]["bands"][i_]["unreached"] for c in allc for i_ in range(3)]; rng2("UnrConc", ur)
cv = [LS["concrete"][c]["bands"][i_]["cover"] for c in allc for i_ in (0, 1)]; rng2("TrCovConc", cv)
bl = [LS["concrete"][c]["bands"][i_]["below_band"] for c in allc for i_ in (0, 1)]; rng2("TrBelowConc", bl)
cu_ = [LS["concrete"][c]["bands"][i_]["cover_urban"] for c in allc for i_ in (0, 1)]; rng2("TrCovUrb", cu_)
R_ = LS["refs"]
for i_, x in enumerate("ABC"):
    M5[f"RefTr{x}"], M5[f"RefSch{x}"], M5[f"RefPsix{x}"], M5[f"RefPsixDir{x}"] = f(R_["tr"][i_], 1), f(R_["schreiber"][i_], 1), f(R_["p681"][i_], 1), f(R_["p681_direct"][i_], 1)
    M5[f"RefTrMun{x}"] = f(CB2["_sbase"]["bands"][i_][3], 1)
gp = [R_["tr"][i_] - R_["schreiber"][i_] for i_ in range(3)]; M5["RefSchGapLo"], M5["RefSchGapHi"] = fd(min(gp)), fd(max(gp))
CENS = lambda t_, b: CB2[t_]["bands"][b][2]
row = lambda lab, t_: lab + " & " + " & ".join(f(CENS(t_, b), 1) for b in range(3)) + " & " + f(100 * CB2[t_]["bands"][0][1], 0) + r"\\"
ROWS = [("Marble 0.1~m (shipped)", "_sbase"), ("Marble 0.05~m", "_wm005"), ("Marble 0.15~m", "_wm015"), ("Marble 0.2~m", "_wm02"), ("Marble 0.3~m", "_wm03"), ("Marble 0.5~m", "_wm05"), ("Marble half space", "_wmhs"),
        ("Concrete 0.1~m", "_wc01"), ("Concrete 0.2~m", "_wc02"), ("Concrete 0.3~m", "_wc03"), ("Concrete half space", "_wchs"), ("Brick 0.3~m", "_wb03"), ("Glass 0.01~m", "_wg03"),
        ("Rough marble 0.1~m", "_ssc"), ("Rough concrete 0.3~m", "_wc03s"), ("Rough concrete half space", "_wchss"),
        ("Depth 6, marble 0.1~m", "_sd6"), ("Depth 9, marble 0.1~m", "_sd9"), ("Depth 6, concrete 0.3~m", "_wc03d6"), ("Depth 6, concrete half space", "_wchsd6"), ("Depth 9, concrete half space", "_wchsd9"),
        ("Plane 3000~m, marble 0.1~m", "_po3000u"), ("Cells $0.56^\\circ$, $4\\times10^9$ rays", "_c056r4e9"), ("Horizontal pol., marble 0.1~m", "_sph"), ("Circular pol.\\ at user, marble 0.1~m", "_scp"),
        ("1.6~GHz, marble 0.1~m", "_sf16"), ("3.5~GHz, marble 0.1~m", "_sf35"),
        ("1.6~GHz, concrete 0.3~m", "_wc03f1.6"), ("3.5~GHz, concrete 0.3~m", "_wc03f3.5"),
        ("No specular reflection, marble 0.1~m", "_snr")]
M5["SensRows"] = "\n".join(row(a_, t_) for a_, t_ in ROWS)
mar = ["_wm005", "_sbase", "_wm015", "_wm02", "_wm03", "_wm05"]; low = [CENS(t_, 0) for t_ in mar]
M5["SensMarLoA"], M5["SensMarHiA"] = f(min(low), 1), f(max(low), 1); M5["ThickSwing"] = f(max(max(CENS(t_, b) for t_ in mar) - min(CENS(t_, b) for t_ in mar) for b in (0, 1)), 0)
conc_ = ["_wc01", "_wc02", "_wc03", "_wchs"]; M5["ThickSwingConc"] = f(max(max(CENS(t_, b) for t_ in conc_) - min(CENS(t_, b) for t_ in conc_) for b in (0, 1)), 1); M5["BaseConcA"], M5["HsMarA"], M5["HsConA"] = f(CENS("_wc03", 0), 0), f(CENS("_wmhs", 0), 1), f(CENS("_wchs", 0), 1)
sm2 = mar + ["_wmhs", "_wc01", "_wc02", "_wc03", "_wchs", "_wb03", "_wg03"]
M5["CloseMarginTr"] = f(min(CB2[t_]["bands"][b][3] - CENS(t_, b) for t_ in sm2 for b in (0, 1)), 1)
M5["FreqMarRise"] = f(CENS("_sf35", 0) - CENS("_sbase", 0), 0)
fs_ = lambda tags: max(max(CENS(t_, b) for t_ in tags) - min(CENS(t_, b) for t_ in tags) for b in range(3))
M5["FreqConcSwing"] = f(fs_(["_wc03f1.6", "_wc03", "_wc03f2.5", "_wc03f3.0", "_wc03f3.5"]), 1); M5["FreqHsSwing"] = f(fs_(["_wchsf1.6", "_wchs", "_wchsf2.5", "_wchsf3.0", "_wchsf3.5"]), 1)
M5["DepthMarA"], M5["DepthMarSixA"], M5["DepthMarNineA"] = f(CENS("_sbase", 0), 1), f(CENS("_sd6", 0), 1), f(CENS("_sd9", 0), 1)
M5["DepthConA"], M5["DepthConSixA"], M5["DepthHsNineA"] = f(CENS("_wc03", 0), 1), f(CENS("_wc03d6", 0), 1), f(CENS("_wchsd9", 0), 1)
M5["UnrThreeLow"], M5["UnrSixLow"] = f(100 * CB2["_wc03"]["bands"][0][1], 0), f(100 * CB2["_wc03d6"]["bands"][0][1], 0)
M5["DepthMidHigh"] = f(max(abs(CENS(a_, b) - CENS(c_, b)) for a_, c_ in (("_sbase", "_sd6"), ("_wc03", "_wc03d6"), ("_wchs", "_wchsd6")) for b in (1, 2)), 1)
M5["PlaneCellMax"] = f(max(abs(CENS(t_, b) - CENS("_sbase", b)) for t_ in ("_po3000u", "_c056r4e9") for b in range(3)), 1)
M5["RoofGroundMax"] = f(max(abs(CENS(t_, b) - CENS("_wc03", b)) for t_ in ("_wc03r", "_wc03g") for b in range(3)), 1)
M5["NoReflA"], M5["NoReflB"], M5["NoReflC"] = (f(CENS("_snr", b), 1) for b in range(3))
M5["DiffuseCheck"] = f(CD["median_db"], 1); M5["CanyonMed"], M5["CanyonPninety"] = f(CY["drop_median"], 0), f(CY["drop_p90"], 0)
# polarization
rows_ = {c: np.load(os.path.join(OUT, f"pol_{c}_c03.npz"))["rows"] for c in allc}; X = np.vstack(list(rows_.values())); P_ = X[np.isfinite(X[:, 4])]
db = lambda v: -10 * np.log10(v)
M5["PolN"] = str(len(P_)); M5["PolShare"] = pts(len(P_) / len(X)); M5["PathRmDiff"] = f(np.median(np.abs(P_[:, 3] - db(P_[:, 4]))), 1)
M5["PolLinCirc"] = f(np.median(np.abs(db(P_[:, 6]) - db(P_[:, 4]))), 1); M5["PolLinLin"] = f(np.median(np.abs(db(P_[:, 5]) - db(P_[:, 4]))), 1); M5["PolCircCirc"] = f(np.median(db(P_[:, 7]) - db(P_[:, 4])), 1)
rises = []
for c in allc:
    r_ = rows_[c]; base = np.where(r_[:, 3] < 150, r_[:, 3], np.inf); d_ = np.where(np.isfinite(r_[:, 4]), db(np.maximum(r_[:, 7], 1e-30)) - db(r_[:, 4]), 0.0)
    for lo_, hi_ in ((15, 30), (30, 60)):
        k = (r_[:, 1] >= lo_) & (r_[:, 1] < hi_); rises.append(float(np.median(base[k] + d_[k]) - np.median(base[k])))
M5["PolRiseLo"], M5["PolRiseHi"] = f(min(rises), 0), f(max(rises), 0)
# misc
neg = []
for c in allc:
    z_ = np.load(os.path.join(OUT, f"sky2_{c}.npz")); x_ = -z_["exc"][~z_["los"]]; neg.append(float((x_ < 0).mean()))
rng2("NegLoss", neg, lambda x: f(100 * x, 0))
def lp(tag, i_, nmax=None):
    z_ = np.load(os.path.join(OUT, f"sky2_etoile{tag}.npz")); T_ = z_["d_edge"][:, None, :] * np.tan(np.radians(z_["els"]))[None, :, None] >= float(z_["H95"]); n_ = nmax or z_["los"].shape[0]
    return float((z_["los"][:n_, i_] * T_[:n_, i_]).sum() / T_[:n_, i_].sum())
M5["InParFifteenBase"], M5["InParFifteenA"], M5["InParFifteenB"] = pts(lp("", 0, 60)), pts(lp("_in150", 0)), pts(lp("_in200", 0))
M5["InParThirtyMax"] = str(int(np.ceil(100 * max(abs(lp("", 3, 60) - lp(t_, 3)) for t_ in ("_in150", "_in200")))))
def lm(tag, i_):
    z_ = np.load(os.path.join(OUT, f"sky2_munich{tag}.npz")); T_ = z_["d_edge"][:, None, :] * np.tan(np.radians(z_["els"]))[None, :, None] >= float(z_["H95"])
    return float((z_["los"][:60, i_] * T_[:60, i_]).sum() / T_[:60, i_].sum())
M5["InMunMax"] = str(int(np.ceil(100 * max(abs(lm("_sbase", i_) - lm(t_, i_)) for t_ in ("_in150", "_in200") for i_ in (0, 3)))))
sd_los, sd_loss = [], []
for c in allc:
    a_ = np.load(os.path.join(OUT, f"sky2_{c}_c03.npz")); b_ = np.load(os.path.join(OUT, f"sky2_{c}_c03sd2.npz")); sd_los.append(abs(a_["los"][:, 3].mean() - b_["los"][:, 3].mean()))
    for i_ in range(3):
        sd_loss.append(abs(LS["concrete"][c]["bands"][i_]["cens"] - LS["concrete_seed2"][c]["bands"][i_]["cens"]))
rt_ = [float(m_.group(2)) / 300 for fn in ("main2.log", "round3.log") for m_ in re.finditer(r"^(\w+) +\S* ?done ([\d.]+)", open(os.path.join(OUT, fn)).read(), re.M) if float(m_.group(2)) > 500]
M5["RuntimeLo"], M5["RuntimeHi"] = f(min(rt_), 0), f(max(rt_), 0)
M5["TrSfLo"], M5["TrSfHi"] = "9", "15"
M5["SeedTwoLos"] = str(int(np.ceil(100 * max(sd_los)))); M5["SeedTwoLoss"] = f(max(sd_loss), 1)
M5["SeedTwoLossLow"] = f(max(abs(LS["concrete"][c]["bands"][i_]["cens"] - LS["concrete_seed2"][c]["bands"][i_]["cens"]) for c in allc for i_ in (0, 1)), 1)
M5["MunHighSeedTwo"] = f(LS["concrete_seed2"]["munich"]["bands"][2]["margin"], 1)
Ac = json.load(open(os.path.join(OUT, "analysis2_c03.json"))); M5["LossRsq"] = f(Ac["loss_fit_all"]["r2"], 2); M5["LossRsqMarble"] = f(A["loss_fit_all"]["r2"], 2)

TR_EL_ = np.arange(10, 91, 10.0); TR_CL_ = np.array([34.3, 30.9, 29.0, 27.7, 26.8, 26.2, 25.8, 25.5, 25.5]); TR_SF_DU = np.array([15.5, 13.9, 12.4, 11.7, 10.6, 10.5, 10.1, 9.2, 9.2])
# ---- round 5 (R2 report): road axis, Holm, realizations, position error, measured rule, half plane, polarization with diffraction ----
SN2 = json.load(open(os.path.join(OUT, "street_null2.json"))); BL = json.load(open(os.path.join(OUT, "brier_ladder.json")))
FAC1 = json.load(open(os.path.join(OUT, "sat_select_facade_first.json"))); FAC2 = json.load(open(os.path.join(OUT, "sat_select_facade_second.json"))); OSH = json.load(open(os.path.join(OUT, "osm_shift.json")))
S1_ = json.load(open(os.path.join(OUT, "sat_select2_c03.json"))); S2_ = json.load(open(os.path.join(OUT, "sat_select2_c03sd2.json"))); VS = json.load(open(os.path.join(OUT, "validate_screen3.json"))); OC = json.load(open(os.path.join(OUT, "osm_check.json")))
FS = FL["skill"]
SEEDS = ("seed1", "seed2"); allc = list(TAG); E30 = "30.0"
sw = lambda c, sd, pop, k, e=E30: SN2[c][sd][pop]["swing"][k][e]
ex = lambda c, sd, pop, ctl, k, e=E30: SN2[c][sd][pop]["excess"][f"{ctl}|{k}|{e}"]
def nz(t):
    """No negative zero in print: -0, -0.0, -0.000 read as 0."""
    return t.lstrip("-") if float(t) == 0 else t
def sgn(x, d=0):
    return nz(f"{100 * x:.{d}f}").replace("-", "$-$")
def pair(a, b, fn=pts):
    return f"{fn(a)}/{fn(b)}"
def R2(k, vals, fmt=pts):
    M5[f"{k}Lo"], M5[f"{k}Hi"] = fmt(min(vals)), fmt(max(vals))
sg = lambda x: sgn(x)
for k_, key in (("Road", "road"), ("Own", "own"), ("St", "st"), ("Cf", "cf"), ("Cftwo", "cf2")):
    R2(f"Sw{k_}One", [sw(c, "seed1", "all", key) for c in allc]); R2(f"Sw{k_}Two", [sw(c, "seed2", "all", key) for c in allc]); R2(f"Sw{k_}", [sw(c, sd, "all", key) for c in allc for sd in SEEDS], sg)
    R2(f"ExRot{k_}", [ex(c, sd, "all", "rotm", key)["mean"] for c in allc for sd in SEEDS], sg)
    R2(f"ExRot{k_}One", [ex(c, "seed1", "all", "rotm", key)["mean"] for c in allc], sg)
    if key != "road":
        R2(f"ExIso{k_}", [ex(c, sd, "all", "iso", key)["mean"] for c in allc for sd in SEEDS], sg)
    for ctl in ("rotm", "iso"):
        if key == "road" and ctl == "iso":
            continue
        for sd in SEEDS:
            M5[f"Sig{k_}{ctl.capitalize()}{'One' if sd == 'seed1' else 'Two'}"] = str(sum(ex(c, sd, "all", ctl, key)["p_holm"] < 0.05 for c in allc))
R2("SwingRotOwn", [ex(c, "seed1", "all", "rotm", "own")["null"] for c in allc]); R2("SwingIsoOwn", [ex(c, "seed1", "all", "iso", "own")["null"] for c in allc])
M5["ExMinOwnAll"] = sg(min(ex(c, sd, "all", q, "own")["mean"] for c in allc for sd in SEEDS for q in ("rotm", "iso")))
M5["ExMaxOwnAll"] = sg(max(ex(c, sd, "all", q, "own")["mean"] for c in allc for sd in SEEDS for q in ("rotm", "iso")))
R2("ExRotRoadCi", [v for c in allc for sd in SEEDS for v in ex(c, sd, "all", "rotm", "road")["ci"]], sg)
R2("NullRoad", [ex(c, sd, "all", "rotm", "road")["null"] for c in allc for sd in ("seed1",)], sg)
M5["RoadPHolmMax"] = f(max(ex(c, sd, "all", "rotm", "road")["p_holm"] for c in allc for sd in SEEDS), 4)
BB = SN2["munich"]["seed1"]["boot_B"]; M5["BootB"] = f"{BB:,}".replace(",", "{,}"); M5["BootPFloor"] = f(2 / (BB + 1), 4)
_nx = sorted(ex(c, "seed2", "all", "iso", "own")["p_holm"] for c in allc); M5["OwnIsoNextHolm"] = f(min(x for x in _nx if x >= 0.05), 3)
# m4: controls reweighted to the scene's shares of own-wall agreement and of road distance
def rw_excess(c, sd, nm):
    sc = SN2[c][sd]["strata"][nm]; tot = sw(c, sd, "all", "road"); vals = []
    for x in SN2[c]["ctl"]["rotm"]:
        cs = x["strata"][nm]; w = [q["share"] for q in sc]; sv = [q2["swing"] for q2 in cs]; okk = [a is not None for a in sv]; ws = sum(a for a, b_ in zip(w, okk) if b_)
        vals.append(sum(a * b_ for a, b_ in zip(w, sv) if b_ is not None) / ws)
    return tot - float(np.mean(vals))
R2("RwWall", [rw_excess(c, sd, "wall") for c in allc for sd in SEEDS], sg); R2("RwDist", [rw_excess(c, sd, "dist") for c in allc for sd in SEEDS], sg)
R2("WallShareScene", [SN2[c][sd]["strata"]["wall"][0]["share"] for c in allc for sd in SEEDS]); R2("WallShareRot", [np.mean([x["strata"]["wall"][0]["share"] for x in SN2[c]["ctl"]["rotm"]]) for c in allc])
np3 = [c for c in allc if c != "etoile"]; R2("NearSwing", [SN2[c][sd]["strata"]["dist"][0]["swing"] for c in np3 for sd in SEEDS], sg); R2("FarSwing", [SN2[c][sd]["strata"]["dist"][2]["swing"] for c in np3 for sd in SEEDS], sg); M5["FarSwingParis"] = sg(min(SN2["etoile"][sd]["strata"]["dist"][2]["swing"] for sd in SEEDS))
R2("StreetRoad", [sw(c, sd, "street", "road") for c in allc for sd in SEEDS], sg); R2("StreetOwnAll", [sw(c, sd, "street", "own") for c in allc for sd in SEEDS], sg)
R2("StreetExRot", [ex(c, sd, "street", "rotm", "own")["mean"] for c in allc for sd in SEEDS], sg); R2("StreetExIso", [ex(c, sd, "street", "iso", "own")["mean"] for c in allc for sd in SEEDS], sg)
R2("StreetExRoad", [ex(c, sd, "street", "rotm", "road")["mean"] for c in allc for sd in SEEDS], sg)
M5["SigStreetOwn"] = str(sum(ex(c, sd, "street", q, "own")["p_holm"] < 0.05 for c in allc for sd in SEEDS for q in ("rotm", "iso")))
ees = ("20.0", "45.0", "60.0")
for e, nm in zip(ees, ("Twenty", "FortyFive", "Sixty")):
    R2(f"ExRoadE{nm}", [ex(c, sd, "all", "rotm", "road", e)["mean"] for c in allc for sd in SEEDS], sg); R2(f"SwRoadE{nm}", [sw(c, sd, "all", "road", e) for c in allc for sd in SEEDS], sg)
    R2(f"ExOwnE{nm}", [ex(c, sd, "all", q, "own", e)["mean"] for c in allc for sd in SEEDS for q in ("rotm", "iso")], sg)
R2("RoadAgree", [SN2[c][sd]["agree"] for c in allc for sd in SEEDS]); R2("RotAgree", [np.mean([x["agree"] for x in SN2[c]["ctl"]["rotm"]]) for c in allc])
R2("RoadDist", [SN2[c][sd]["road_dist_med"] for c in allc for sd in SEEDS], lambda x: f(x, 0))
R2("OsmIou", [OC[c]["iou"] for c in allc], lambda x: f(x, 2)); R2("OsmIn", [OC[c]["drive_in_building"] for c in allc])
M5["FloorRmseModel"], M5["FloorRmseConst"], M5["FloorSpearman"], M5["FloorAbove"], M5["FloorN"] = f(100 * FS["rmse_model"], 1), f(100 * FS["rmse_const_mean"], 1), f(FS["spearman"], 2), str(FS["above"]), str(FS["n"])
SC = ("Munich", "Paris", "Florence", "S.F.")
def nrow(label, fn):
    return label + " & " + " & ".join(fn(c) for c in allc) + r"\\"
two = lambda a, b: f"{a}/{b}"
NR = [nrow("Road axis, scene", lambda c: two(pts(sw(c, "seed1", "all", "road")), pts(sw(c, "seed2", "all", "road")))),
      nrow(r"\quad minus rotated blocks", lambda c: two(sg(ex(c, "seed1", "all", "rotm", "road")["mean"]), sg(ex(c, "seed2", "all", "rotm", "road")["mean"]))),
      nrow(r"\quad 95\,\% interval, first set", lambda c: "%s to %s" % tuple(sg(v) for v in ex(c, "seed1", "all", "rotm", "road")["ci"])),
      nrow("Own axis, scene", lambda c: two(pts(sw(c, "seed1", "all", "own")), pts(sw(c, "seed2", "all", "own")))),
      nrow(r"\quad rotated blocks / rectangles", lambda c: two(pts(ex(c, "seed1", "all", "rotm", "own")["null"]), pts(ex(c, "seed1", "all", "iso", "own")["null"]))),
      nrow(r"\quad minus rotated blocks", lambda c: two(sg(ex(c, "seed1", "all", "rotm", "own")["mean"]), sg(ex(c, "seed2", "all", "rotm", "own")["mean"]))),
      nrow(r"\quad minus rectangles", lambda c: two(sg(ex(c, "seed1", "all", "iso", "own")["mean"]), sg(ex(c, "seed2", "all", "iso", "own")["mean"]))),
      nrow("Mask axis, minus rotated blocks", lambda c: two(sg(ex(c, "seed1", "all", "rotm", "st")["mean"]), sg(ex(c, "seed2", "all", "rotm", "st")["mean"]))),
      nrow("Cross-fitted, scene", lambda c: two(pts(sw(c, "seed1", "all", "cf")), pts(sw(c, "seed2", "all", "cf")))),
      nrow(r"\quad minus rotated blocks", lambda c: two(sg(ex(c, "seed1", "all", "rotm", "cf")["mean"]), sg(ex(c, "seed2", "all", "rotm", "cf")["mean"]))),
      nrow(r"\quad wider annulus, scene", lambda c: two(pts(sw(c, "seed1", "all", "cf2")), pts(sw(c, "seed2", "all", "cf2")))),
      nrow(r"\quad wider annulus, minus rotated", lambda c: two(sg(ex(c, "seed1", "all", "rotm", "cf2")["mean"]), sg(ex(c, "seed2", "all", "rotm", "cf2")["mean"]))),
      nrow(r"Street users, share scene (\%)", lambda c: two(pts(SN2[c]["seed1"]["agree"]), pts(SN2[c]["seed2"]["agree"]))),
      nrow(r"\quad share, rotated blocks (\%)", lambda c: pts(np.mean([x["agree"] for x in SN2[c]["ctl"]["rotm"]]))),
      nrow(r"\quad road axis, scene", lambda c: two(pts(sw(c, "seed1", "street", "road")), pts(sw(c, "seed2", "street", "road")))),
      nrow(r"\quad own axis, minus rotated", lambda c: two(sg(ex(c, "seed1", "street", "rotm", "own")["mean"]), sg(ex(c, "seed2", "street", "rotm", "own")["mean"]))),
      nrow("Second harmonic, scene", lambda c: f(SN[c]["real"]["harm"], 2)),
      nrow(r"\quad rotated / rectangles", lambda c: two(f(nullmean(c, "rotm", "harm"), 2), f(nullmean(c, "iso", "harm"), 2)))]
M5["NullRows"] = "\n".join(NR)
LR = [("TR 38.811, better table", "tr_best"), ("Schreiber et al.~\\cite{schreiber2026enhancing}", "schreiber"), ("Elevation, other cities", "elev"), ("Elevation, own axis", "axis_own"), ("Elevation, road axis", "axis_road"), ("Elevation, cross-fitted axis", "axis_cf"),
      ("Wall distance, no axis", "side_own"), ("Own axis, wall distance", "own_side"), ("Road axis, wall distance", "road_side"), ("Facade distances $t(a)$", "t"), ("$t(a)$, own axis", "t_own"), ("$t(a)$, road axis", "t_road"), ("Skyline, a check of Prop.~\\ref{prop:sky}", "skyline_check")]
M5["LadderRows"] = "\n".join(nrow(a_, lambda c, k_=k_: two(f(BL["seed1"][c][k_]["mean"], 3), f(BL["seed2"][c][k_]["mean"], 3))) for a_, k_ in LR)
blv = lambda k: [BL[sd][c][k]["mean"] for c in allc for sd in ("seed1", "seed2")]
for k_, nm in (("elev", "Elev"), ("axis_road", "AxRoad"), ("axis_own", "AxOwn"), ("road_side", "RoadSide"), ("t", "Tee"), ("skyline_check", "Sky")):
    R2(f"Bl{nm}", blv(k_), f3 if k_ == "skyline_check" else f2)
R2("BlGainRoad", blv("gain_elev_axis_road"), f3); R2("BlGainTroad", blv("gain_t_t_road"), lambda x: nz(f(x, 3)).replace("-", "$-$")); R2("BlGainTown", blv("gain_t_t_own"), lambda x: nz(f(x, 3)).replace("-", "$-$"))
R2("BlGainTfromRoadSide", blv("gain_road_side_t"), f3)
def srow(label, k, kind="los", D1=S1_, D2=S2_, sh="dense"):
    fm = (lambda v: pts(v)) if kind == "los" else (lambda v: f(v, 0)) if kind == "p90" else (lambda v: f(v, 1))
    key = {"los": "los", "p90": "p90", "ho": "ho"}[kind]
    return nrow(label, lambda c: two(fm(D1[c][sh][k][key]), fm(D2[c][sh][k][key])))
SR = [srow("Highest elevation", "HE"), srow("Road-axis table, exact position", "AZroad0"), srow(r"Road-axis table, $\sigma=10$~m", "AZroad10"), srow(r"Horizon, $\sigma=0$", "sig0"), srow(r"Horizon, $\sigma=10$~m", "sig10"), srow(r"Marginal map, $\sigma=3$~m", "marg3"), srow(r"Marginal map, $\sigma=5$~m", "marg5"), srow(r"Marginal map, $\sigma=10$~m", "marg10"), srow(r"Marginal map, $\sigma=20$~m", "marg20"), srow(r"Marginal, 0.1 margin, $\sigma=10$~m", "marg10h"),
      srow("Measured, 6~dB error, 6~dB margin", "RSRP6h6"), srow("Measured, 3~dB error, 3~dB margin", "RSRP3h3"), srow("Best ray-traced link (oracle)", "ORA"),
      srow("Path loss: highest elevation", "HE", "p90"), srow(r"Path loss: marginal, $\sigma=10$~m", "marg10", "p90"), srow("Path loss: measured, 6~dB", "RSRP6h6", "p90"), srow("Handovers: highest elevation", "HE", "ho"), srow(r"Handovers: marginal, $\sigma=10$~m", "marg10", "ho"), srow("Handovers: measured, 6~dB", "RSRP6h6", "ho")]
M5["SelRows"] = "\n".join(SR)
M5["SpRows"] = "\n".join([srow("Highest elevation", "HE", sh="sparse"), srow(r"Horizon, $\sigma=0$", "sig0", sh="sparse"), srow(r"Horizon, $\sigma=10$~m", "sig10", sh="sparse"), srow(r"Marginal map, $\sigma=10$~m", "marg10", sh="sparse"), srow("Measured, 6~dB error, 6~dB margin", "RSRP6h6", sh="sparse")] + [nrow(r"Any in LOS, sky maps, first set", lambda c: pts(SS["sparse"][c]["any_los"])), nrow("Any in LOS, independent TR draws", lambda c: pts(SS["sparse"][c]["tr_any"]["dense_urban"])), nrow("Any in LOS, independent own-curve draws", lambda c: pts(FAC1[c]["any"]["own_curve"]))])
dv = lambda k, sh="dense": [D[c][sh][k]["los"] for D in (S1_, S2_) for c in allc]
R2("NHe", dv("HE")); R2("NMapZero", dv("sig0")); R2("NMapTen", dv("sig10")); R2("NMeas", dv("RSRP6h6")); R2("NMeasThree", dv("RSRP3h3")); R2("NOra", dv("ORA")); R2("NKel", dv("K3elev"))
dif = lambda a, b="HE": [x - y for x, y in zip(dv(a), dv(b))]
R2("NGainMapZero", dif("sig0")); R2("NGainMapTen", dif("sig10")); R2("NGainMapTwenty", dif("sig20"), sg); R2("NGainMapThree", dif("sig3")); R2("NGainRoad", dif("AZroad0"), sg); R2("NGainRoadTen", dif("AZroad10"), sg); R2("NGainMeas", dif("RSRP6h6")); R2("NKmap", [x - y for x, y in zip(dv("K3map10"), dv("K3elev"))], lambda x: f"{100 * x:.1f}".replace("-", "$-$"))
M5["NMeasMinusMapThreeLo"], M5["NMeasMinusMapThreeHi"] = f"{100 * min(x - y for x, y in zip(dv('RSRP6h6'), dv('sig3'))):.1f}".replace("-", "$-$"), f"{100 * max(x - y for x, y in zip(dv('RSRP6h6'), dv('sig3'))):.1f}"
M5["NMapOverMeas"] = pts(max(x - y for x, y in zip(dv("sig0"), dv("RSRP6h6"))))
f1p = lambda x: nz(f"{100 * x:.1f}").replace("-", "$-$")
for s_, nm in ((3, "Three"), (5, "Five"), (10, "Ten"), (20, "Twenty")):
    R2(f"NGainMarg{nm}", dif(f"marg{s_}"), sg); R2(f"NMeasMinusMarg{nm}", [x - y for x, y in zip(dv("RSRP6h6"), dv(f"marg{s_}"))], f1p); M5[f"NMargWins{nm}"] = str(sum(y > x for x, y in zip(dv("RSRP6h6"), dv(f"marg{s_}")))); M5[f"NMargWinsThreeDb{nm}"] = str(sum(y > x for x, y in zip(dv("RSRP3h3"), dv(f"marg{s_}"))))
R2("NHxSeven", [x - y for x, y in zip(dv("sig0"), dv("sig0_hx07"))]); R2("NMarg", dv("marg10")); R2("NMargThree", dv("marg3")); R2("NGainMargHTen", dif("marg10h"), sg); R2("NGainMargAsFive", dif("marg10as5"), sg); R2("NGainMargAsTwenty", dif("marg10as20"), sg)
R2("NHybridMarg", [x - y for x, y in zip(dv("K3marg10"), dv("K3elev"))], f1p)
ho_ = lambda k: [D[c]["dense"][k]["ho"] for D in (S1_, S2_) for c in allc]; f1 = lambda x: f(x, 1)
R2("NHoHe", ho_("HE"), f1); R2("NHoMeas", ho_("RSRP6h6"), f1); R2("NHoMap", ho_("sig0"), f1); R2("NHoMarg", ho_("marg10"), f1); R2("NHoMargH", ho_("marg10h"), f1); R2("NHoPlugTen", ho_("sig10"), f1); R2("NHoMeasZero", ho_("RSRP6h0"), f1); R2("NMeasZero", dv("RSRP6h0"))
R2("NPnMarg", [D[c]["dense"]["marg10"]["p90"] for D in (S1_, S2_) for c in allc], lambda x: f(x, 0))
p9_ = lambda k: [D[c]["dense"][k]["p90"] for D in (S1_, S2_) for c in allc]; f0 = lambda x: f(x, 0)
R2("NPnHe", p9_("HE"), f0); R2("NPnMeas", p9_("RSRP6h6"), f0); R2("NPnMapZero", p9_("sig0"), f0); R2("NPnMapTen", p9_("sig10"), f0)
# half plane: true vertical sheet against the exact value; the box follows the horizontal plate
sh_ = lambda n, p, k_: np.array([x[k_] for x in VS[n][p]["path"]]); ref_ = dict(V="vhard", H="vsoft"); refh = dict(V="hhard", H="hsoft")
M5["ScreenTotMax"] = f(max(np.abs(sh_("sheet", p, "solver") - sh_("sheet", p, ref_[p])).max() for p in "VH"), 1); M5["ScreenTopMax"] = f(max(np.abs(sh_("sheet", p, "top") - sh_("sheet", p, ref_[p])).max() for p in "VH"), 2)
rm_ = np.concatenate([np.array(VS["sheet"][p]["rm_solver"]) - np.array(VS["sheet"][p]["rm_exact"][ref_[p]]) for p in "VH"]); M5["ScreenRmMed"] = nz(f(np.median(rm_), 1)).replace("-", "$-$"); M5["ScreenRmAbs"] = f(np.median(np.abs(rm_)), 1)
M5["ScreenCubeMax"] = f(max(np.abs(sh_("cube20cm", p, "solver") - sh_("cube20cm", p, refh[p])).max() for p in "VH"), 1); M5["ScreenCubeOff"] = f(max(np.abs(sh_("cube20cm", p, "solver") - sh_("cube20cm", p, ref_[p])).max() for p in "VH"), 0)
RM_ = VS["rim"]; rimd = np.concatenate([np.array(RM_["split_concrete"][p]["path"]) - np.array(RM_["welded_concrete"][p]["path"]) for p in "VH"]); R2("Rim", list(rimd), lambda x: f(x, 1))
# polarization and mixed paths with diffraction on
rows_ = {c: np.load(os.path.join(OUT, f"pol_{c}_c03.npz"))["rows"] for c in allc}; X = np.vstack(list(rows_.values())); P_ = X[np.isfinite(X[:, 4])]
M5["PolN"] = str(len(X)); M5["PolShare"] = f(100 * len(P_) / len(X), 1); M5["PathRmDiff"] = f(np.median(P_[:, 3][P_[:, 3] < 150] - db(P_[:, 4][P_[:, 3] < 150])), 1)
dshare = []
for c_ in allc:
    r_ = np.load(os.path.join(OUT, f"pol_{c_}_c03.npz"))["rows"]; r_ = r_[np.isfinite(r_[:, 4])]
    for lo_, hi_ in ((15, 30), (30, 60), (60, 90)):
        k_ = (r_[:, 1] >= lo_) & (r_[:, 1] < hi_); dshare.append(float((r_[k_, 10] > 0.5 * (r_[k_, 10] + r_[k_, 11] + r_[k_, 12])).mean()))
R2("DiffShare", dshare)
M5["PathRmAbs"] = f(np.median(np.abs(P_[:, 3][P_[:, 3] < 150] - db(P_[:, 4][P_[:, 3] < 150]))), 1)
M5["PolLinCirc"] = f(np.median(np.abs(db(P_[:, 6]) - db(P_[:, 4]))), 1); M5["PolLinLin"] = f(np.median(np.abs(db(P_[:, 5]) - db(P_[:, 4]))), 1); M5["PolCircCirc"] = f(np.median(db(np.maximum(P_[:, 7], 1e-30)) - db(P_[:, 4])), 1)
rises, mix, cmarg, pd_ = [], [], [], []
TRB = (LS["refs"]["tr"][0], LS["refs"]["tr"][1])
for c in allc:
    r_ = rows_[c]; ok = np.isfinite(r_[:, 4]); pid = np.where(ok, db(np.maximum(r_[:, 4], 1e-30)), np.inf); prr = np.where(ok, db(np.maximum(r_[:, 7], 1e-30)), np.inf); pnm = np.where(ok, db(np.maximum(r_[:, 10] + r_[:, 11], 1e-30)), np.inf); bm = np.where(r_[:, 3] < 150, r_[:, 3], np.inf)
    for i_, (lo_, hi_) in enumerate(((15, 30), (30, 60))):
        k = (r_[:, 1] >= lo_) & (r_[:, 1] < hi_); rises.append(float(np.median(prr[k]) - np.median(pid[k]))); mix.append(float(np.median(pnm[k]) - np.median(pid[k]))); cmarg.append(float(TRB[i_] - np.median(prr[k]))); pd_.append(float(np.median(pid[k]) - np.median(bm[k])))
R2("PolRise", rises, fd); R2("MixDrop", mix, fd); M5["PolMarginMin"] = f(min(cmarg), 0); R2("PathBand", pd_, lambda x: f(x, 0).replace("-", "$-$"))
# margins over both realizations
mc2 = [LS[st][c]["bands"][i_]["margin"] for st in ("concrete", "concrete_seed2") for c in allc for i_ in (0, 1)]; R2("MarginConc", mc2, fd)
R2("MarginConcOne", margins("concrete"), fd)
_mn = min(((LS[st][c]["bands"][i_]["margin"], LS[st][c]["bands"][i_]["margin_ci"]) for st in ("concrete", "concrete_seed2") for c in allc for i_ in (0, 1)), key=lambda q: q[0]); M5["MarginMinVal"], M5["MarginMinCiLo"], M5["MarginMinCiHi"] = f(_mn[0], 1), f(_mn[1][0], 1), f(_mn[1][1], 1)
tb_ = np.array([9.2, 15.5]); M5["TrSfLo"], M5["TrSfHi"] = f(np.interp(55, TR_EL_, TR_SF_DU), 1), f(np.interp(15, TR_EL_, TR_SF_DU), 1)
M5["TrClLo"], M5["TrClHi"] = f(np.interp(55, TR_EL_, TR_CL_), 1), f(np.interp(15, TR_EL_, TR_CL_), 1)
cell = []
for c in allc:
    z_ = np.load(os.path.join(OUT, f"sky2_{c}.npz")); cell += list(np.degrees(2.5 / (z_["plane_dist"] - 970)))
M5["CellAngLo"], M5["CellAngHi"] = f(min(cell), 2), f(max(cell), 2)
# one definition per name: later rounds win
defs = {}
with open(os.path.join(HERE, "..", "macros_p9.tex")) as fh:
    for ln in fh:
        mm = re.match(r"\\newcommand\{\\(\w+)\}", ln)
        if mm:
            defs[mm.group(1)] = ln.rstrip("\n")
# ---- round R4: facade scope of the selection comparison, registration, sparse-shell level, polarization and sweep scope ----
FACS = (("_c03", "Smooth concrete 0.3~m"), ("_c03s", "Rough concrete 0.3~m"), ("marble", "Marble 0.1~m"), ("_sc4", "Rough marble 0.1~m"))
for c in allc:      # the pipeline must reproduce the stored concrete runs, and map rules must not depend on the wall
    for k_ in S1_[c]["dense"]:
        assert FAC1[c]["_c03"]["dense"][k_] == S1_[c]["dense"][k_] and FAC2[c]["_c03sd2"]["dense"][k_] == S2_[c]["dense"][k_], (c, k_)
    for t_, _ in FACS:
        assert FAC1[c][t_]["dense"]["marg10"]["los"] == FAC1[c]["_c03"]["dense"]["marg10"]["los"] and FAC1[c][t_]["dense"]["sig10"]["los"] == FAC1[c]["_c03"]["dense"]["sig10"]["los"]
def _ci(v, ci, d=1):
    s_ = lambda x: nz(f"{x:.{d}f}").replace("-", "$-$")
    return f"{s_(v)} ({s_(ci[0])}, {s_(ci[1])})"
fp_ = lambda x: nz(f"{x:.1f}").replace("-", "$-$"); cp_ = lambda t_, c, s_=10: FAC1[c][t_]["paired"][f"marg{s_}"]
FT_ = (FACS[0], FACS[2])
rows_f = [r"\multicolumn{5}{@{}l}{LOS share, measured minus marginal map (points)}\\"] + [nrow(lb, lambda c, t_=t_: _ci(100 * cp_(t_, c)["los"], [100 * x for x in cp_(t_, c)["los_ci"]])) for t_, lb in FT_]
rows_f += [r"\midrule", r"\multicolumn{5}{@{}l}{90th percentile path loss, measured minus marginal map (dB)}\\"] + [nrow(lb, lambda c, t_=t_: _ci(cp_(t_, c)["p90"], cp_(t_, c)["p90_ci"])) for t_, lb in FT_]
M5["FacRows"] = "\n".join(rows_f)
ALLP = lambda s_: [FAC1[c]["_c03"]["paired"][f"marg{s_}"] for c in allc] + [FAC2[c]["_c03sd2"]["paired"][f"marg{s_}"] for c in allc]
for s_, nm in ((3, "Three"), (5, "Five"), (10, "Ten"), (20, "Twenty")):
    P_c = ALLP(s_); M5[f"PairMapClear{nm}"] = str(sum(p["los_ci"][1] < 0 for p in P_c)); M5[f"PairMeasClear{nm}"] = str(sum(p["los_ci"][0] > 0 for p in P_c)); M5[f"PairTie{nm}"] = str(sum(p["los_ci"][0] <= 0 <= p["los_ci"][1] for p in P_c))
    P_m = [cp_("marble", c, s_) for c in allc]; M5[f"MarMapClear{nm}"] = str(sum(p["los_ci"][1] < 0 for p in P_m)); M5[f"MarMeasClear{nm}"] = str(sum(p["los_ci"][0] > 0 for p in P_m)); M5[f"MarTie{nm}"] = str(sum(p["los_ci"][0] <= 0 <= p["los_ci"][1] for p in P_m)); R2(f"MarMinus{nm}", [100 * p["los"] for p in P_m], fp_)
    M5[f"MarWinsPt{nm}"] = str(sum(p["los"] < 0 for p in P_m))
for c, t in TAG.items():
    p_ = cp_("marble", c); M5[f"FacMarLos{t}"] = fp_(100 * p_["los"]); M5[f"FacMarPn{t}"] = fp_(p_["p90"]); M5[f"FacConLos{t}"] = fp_(100 * cp_("_c03", c)["los"]); M5[f"FacConPn{t}"] = fp_(cp_("_c03", c)["p90"]); M5[f"FacMarLosCi{t}"] = "%s to %s" % tuple(fp_(100 * x) for x in p_["los_ci"]); M5[f"FacMarPnCi{t}"] = "%s to %s" % tuple(fp_(x) for x in p_["p90_ci"])
    M5[f"NMeasMar{t}"] = pts(FAC1[c]["marble"]["dense"]["RSRP6h6"]["los"])
R2("FacMed", [cp_(t_, c)["p50"] for t_, _ in FACS for c in allc], fp_)
R2("NMeasMarble", [FAC1[c]["marble"]["dense"]["RSRP6h6"]["los"] for c in allc]); R2("NMeasRoughMarble", [FAC1[c]["_sc4"]["dense"]["RSRP6h6"]["los"] for c in allc]); R2("OraMarble", [FAC1[c]["marble"]["dense"]["ORA"]["los"] for c in allc]); R2("OraConc", [FAC1[c]["_c03"]["dense"]["ORA"]["los"] for c in allc])
R2("ServNlosConc", [FAC1[c]["_c03"]["served"]["RSRP6h6"]["nlos_med"] for c in allc], lambda x: f(x, 1)); R2("ServNlosMar", [FAC1[c]["marble"]["served"]["RSRP6h6"]["nlos_med"] for c in allc], lambda x: f(x, 1)); R2("ServShareConc", [FAC1[c]["_c03"]["served"]["RSRP6h6"]["under6"] for c in allc]); R2("ServShareMar", [FAC1[c]["marble"]["served"]["RSRP6h6"]["under6"] for c in allc])
_oo = [(D[c]["dense"][k_]["los"] - D[c]["dense"]["ORA"]["los"], c) for D in (S1_, S2_) for c in allc for k_ in D[c]["dense"] if k_ != "ORA"]; assert all(c == "etoile" for v, c in _oo if v >= 0); M5["NOraOver"] = f(100 * max(_oo)[0], 1)
M5["NMargThreeOverZero"] = f(100 * max(x - y for x, y in zip(dv("marg3"), dv("sig0"))), 1)
_T3 = ("munich", "florence", "san_francisco"); _tail = [cp_(t_, c)["p90"] for t_, _ in FACS for c in _T3] + [FAC2[c]["_c03sd2"]["paired"]["marg10"]["p90"] for c in _T3]
assert all(cp_(t_, c)["p90_ci"][1] < 0 for t_, _ in FACS for c in _T3) and all(FAC2[c]["_c03sd2"]["paired"]["marg10"]["p90_ci"][1] < 0 for c in _T3) and all(cp_(t_, "etoile")["p90_ci"][0] > 0 for t_, _ in FACS) and FAC2["etoile"]["_c03sd2"]["paired"]["marg10"]["p90_ci"][0] > 0     # "three of four cities on every wall"
assert all(cp_(t_, c)["p50_ci"][0] > 0 for t_, _ in FACS for c in allc); assert cp_("marble", "florence")["los_ci"][1] < 0 and cp_("marble", "munich")["los_ci"][0] <= 0 <= cp_("marble", "munich")["los_ci"][1] and cp_("marble", "etoile")["los_ci"][0] <= 0 <= cp_("marble", "etoile")["los_ci"][1] and cp_("marble", "san_francisco")["los_ci"][0] > 0
assert all(p["los_ci"][0] <= 0 <= p["los_ci"][1] for p in (cp_("_c03", "etoile"), FAC2["etoile"]["_c03sd2"]["paired"]["marg10"]))     # Paris ties on both realizations
assert all(cp_("marble", c, s_)["los_ci"][1] < 0 for c in allc for s_ in (3, 5)); assert all(cp_("marble", c, 20)["los_ci"][0] > 0 for c in ("munich", "etoile", "san_francisco")) and cp_("marble", "florence", 20)["los_ci"][0] <= 0 <= cp_("marble", "florence", 20)["los_ci"][1]
R2("TailGain", [abs(x) for x in _tail], lambda x: f(x, 0)); R2("TailPar", [cp_(t_, "etoile")["p90"] for t_, _ in FACS] + [FAC2["etoile"]["_c03sd2"]["paired"]["marg10"]["p90"]], lambda x: f(x, 1))
R2("RoughLosShift", [100 * abs(cp_(a_, c)["los"] - cp_(b_, c)["los"]) for a_, b_ in (("_c03", "_c03s"), ("marble", "_sc4")) for c in allc], lambda x: f(x, 1))
R2("MeasThreeMar", [100 * FAC1[c]["marble"]["paired3"]["marg10"]["los"] for c in allc], lambda x: f(x, 1)); M5["OraFloConc"], M5["OraFloMar"] = pts(FAC1["florence"]["_c03"]["dense"]["ORA"]["los"]), pts(FAC1["florence"]["marble"]["dense"]["ORA"]["los"])
# sparse shell: independence alone against the level of the table
own_gap = [FAC1[c]["any"]["own_curve"] - S_[c]["any_los"] for c in allc]; R2("SpOwnGap", own_gap); M5["SpTrGapParis"] = pts(S_["etoile"]["any_los"] - S_["etoile"]["tr_any"]["dense_urban"]); M5["TrDuThirty"] = "39.8"     # TR 38.811 dense urban, 30 deg
M5["FloOwnThirty"] = f(100 * np.load(os.path.join(OUT, "sky2_florence.npz"))["los"][:, 3, :].mean(), 0)
# registration
R2("OsmBest", [OSH[c]["best_dist"] for c in allc], lambda x: f(x, 0)); M5["OsmSwingShift"] = f(max(abs(OSH[c][sd]["swing_best"] - OSH[c][sd]["swing0"]) * 100 for c in allc for sd in SEEDS), 0)
R2("DriveSw", [sw(c, sd, "all", "drive") for c in allc for sd in SEEDS]); R2("DriveEx", [ex(c, sd, "all", "rotm", "drive")["mean"] for c in allc for sd in SEEDS], sg)
# polarization against the smallest margin, on stated bases; 60-user subset against all users; sweep extrapolation
assert _mn[0] == LS["concrete_seed2"]["florence"]["bands"][0]["margin"]
_b = LS["concrete_seed2"]["florence"]["bands"][0]; M5["MarginMinRef"], M5["MarginMinCens"] = f(_b["tr"], 1), f(_b["cens"], 1); _ri = rises[2 * allc.index("florence")]; M5["PolRiseFlo"], M5["PolWorstReach"] = f(_ri, 1), f(_b["cens"] + _ri, 1)
_ms, _ma, _cs, _ca = CENS("_sbase", 0), LS["marble"]["munich"]["bands"][0]["cens"], CENS("_wc03", 0), LS["concrete"]["munich"]["bands"][0]["cens"]
M5["SubMarSix"], M5["SubMarAll"], M5["SubConSix"], M5["SubConAll"] = f(_ms, 1), f(_ma, 1), f(_cs, 1), f(_ca, 1); R2("SubsetGap", [_ms - _ma, _cs - _ca], lambda x: f(x, 0))
_rise = CENS("_wm02", 0) - CENS("_sbase", 0); M5["SensMarRiseTwo"] = f(_rise, 1); M5["SensFloReach"] = f(LS["marble"]["florence"]["bands"][0]["cens"] + _rise, 1); M5["CloseMarginFreq"] = f(CB2["_sf35"]["bands"][0][3] - CENS("_sf35", 0), 1)
# Brier scale (relative to the elevation curve learned elsewhere) and the knife-edge baseline on both walls
_rel = [(BL[sd][c]["elev"]["mean"] - BL[sd][c]["axis_road"]["mean"]) / BL[sd][c]["elev"]["mean"] for sd in SEEDS for c in allc]
_fac = [(BL[sd][c]["elev"]["mean"] - BL[sd][c]["t"]["mean"]) / BL[sd][c]["elev"]["mean"] for sd in SEEDS for c in allc]
assert all(BL[sd][c]["gain_elev_axis_road"]["ci"][0] > 0 for sd in SEEDS for c in allc)
M5["BlRelRoadHi"] = f(100 * max(_rel), 0); R2("BlFacRed", [100 * x for x in _fac], lambda x: f(x, 0))
_AC = json.load(open(os.path.join(OUT, "analysis2_c03.json")))["loss_loo"]
R2("MaeKnifeConc", [_AC[c]["mae_knife"] for c in allc], lambda x: f(x, 0)); R2("MaeConstConc", [_AC[c]["mae_const"] for c in allc], lambda x: f(x, 0))
R2("MaeKnifeMar", [A["loss_loo"][c]["mae_knife"] for c in allc], lambda x: f(x, 0)); R2("MaeConstMar", [A["loss_loo"][c]["mae_const"] for c in allc], lambda x: f(x, 0))
defs_neg = {"LosNormMed", "LosNormPone"}
for k in defs_neg:
    defs[k] = defs[k].replace("{-", "{$-$")
for k, v in M5.items():
    defs[k] = "\\newcommand{\\%s}{%s}" % (k, v)
for k in ("ScreenDiffLo", "ScreenDiffHi", "HalfHardV", "HalfHardVthick", "HalfHardH", "HalfSoftV", "HalfSoftVthick", "HalfSoftH", "HalfHardLo", "HalfHardHi"):  # R3: withdrawn, the box was not the half plane
    defs.pop(k, None)
with open(os.path.join(HERE, "..", "macros_p9.tex"), "w") as fh:
    fh.write("% generated by sim/make_macros2.py; do not edit\n")
    for k in sorted(defs):
        fh.write(defs[k] + "\n")
print(len(defs), "macros in total")
