#!/usr/bin/env python3
"""
make_cluster_visualizer.py - cluster_visualizer.html: the paper's runs, in figure order.

    python make_cluster_visualizer.py              order json + html
    python make_cluster_visualizer.py --html-only  re-patch the html, keep the json
    python make_cluster_visualizer.py --json-only  recompute the order, keep the html

A COPY OF clustering_visualizer.html WITH THREE CHANGES AND NO OTHERS.

1. Only the runs the paper uses. The manifest lists 21 runs; the copy keeps the ones a
   paper figure is built from - convex NMF on the four concat feature sets (FIG 1, FIG 3,
   FIG 2's feature-set half) and k-means, Ward and archetypes on those sets (FIG 2's
   algorithm half) - in the paper's order, the standard (5 bands z-scored) first. The
   ungated cohort-2 runs and the per-condition cohort-3 runs are not offered.

2. No coverage panel. The two "Sampling" maps leave the dropdown and the report loses
   its "How well was this cohort sampled?" section and the two sampling turns. The
   coverage DATA is still loaded, marked hidden: the report measures its glass rotation
   box on the coverage map, and the patient gate on cluster maps never used it anyway.

3. Clusters in figure order. Every place the page iterates a cut's clusters - the map
   dropdown, the report's donut, table and per-cluster cards, the K-panel chips - goes
   through figureOrder(run, K, ids), which reads cluster_visualizer_order.json. That
   file is computed HERE with the figures' own code: for the four cNMF runs it is
   P2.block_order(), the call figure_1 makes, at every K of the sweep; for the algorithm
   runs it is the order of the reference cluster each one is Hungarian-matched to, which
   is how FIG 2 lays out its rows. Labels carry the position: "#2 · Cluster 3" is the
   second block from the left in FIG 1. The page also opens at K = 8, the paper's cut,
   where the sweep has it.

Every edit is an anchored substitution that must match exactly once in the source, so
a change to clustering_visualizer.html that moves an anchor fails here rather than
producing a copy that silently lost a patch. The copy is regenerated from the original
each time; nothing is hand-edited in it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SITE_DIR = Path.home() / "lorafanda.github.io"
SRC = SITE_DIR / "clustering_visualizer.html"
DST = SITE_DIR / "cluster_visualizer.html"
ORDER_JSON = SITE_DIR / "cluster_visualizer_order.json"
MANIFEST = (ROOT / "outputs" / "250_recon" / "fsaverage" / "coverage_viz" / "manifest.json")

PAPER_K = 8
FSETS = ["concat_bands5z", "concat_hg", "concat_rawds", "concat_bands5"]   # standard first
METHODS = ["cnmf", "kmeans", "hierarchical", "archetypes"]
COHORT = "cohort1_n27"


def load_p2_f2():
    sp = importlib.util.spec_from_file_location("p2fig1", ROOT / "00_Paper2_Figures.py")
    P2 = importlib.util.module_from_spec(sp); sp.loader.exec_module(P2)
    sf = importlib.util.spec_from_file_location("p2fig2", ROOT / "00_paper2_figures2_2.py")
    F2 = importlib.util.module_from_spec(sf); sf.loader.exec_module(F2)
    return P2, F2


def paper_runs(manifest):
    """The manifest's runs that a paper figure is built from, in the paper's order."""
    by = {}
    for r in manifest["runs"]:
        parts = r["id"].split("__")
        if len(parts) != 3 or r.get("cohort_id") != COHORT:
            continue
        method, fset, stamp = parts
        if method in METHODS and fset in FSETS:
            by[(method, fset)] = (r, stamp)
    out = []
    for method in METHODS:
        for fset in FSETS:
            if (method, fset) in by:
                out.append((method, fset) + by[(method, fset)])
    return out


def keys_of(meta, P2):
    return [f"{p}|{P2.norm(e)}" for p, e in zip(meta["patient_id"], meta["electrode"])]


def compute_order(manifest):
    P2, F2 = load_p2_f2()
    runs = paper_runs(manifest)
    ks = [int(k) for k in manifest.get("sweep_ks", list(range(5, 13)))]
    ref_fset = P2.MATCH_REF_FSET
    out, labels, skipped = {}, {}, []
    for method, fset, r, stamp in runs:
        out[r["id"]] = {}
        labels[r["id"]] = {"track_label": r.get("track_label"), "run": r.get("run"),
                           "method": method, "feature_set": fset}
    for k in ks:
        ref = P2.load_run(ref_fset, k)
        ref["fset"] = ref_fset
        C = P2.cube(ref["X"], ref)
        ref_means = np.stack([C[ref["lab"] == j].mean(0) for j in range(k)])
        ref_order = [int(j) for j in P2.block_order(ref_means, ref)[0]]
        ref_pos = {c: i for i, c in enumerate(ref_order)}
        ref_keys = keys_of(ref["meta"], P2)
        for method, fset, r, stamp in runs:
            rid = r["id"]
            try:
                if method == "cnmf":
                    d = P2.load_run(fset, k)
                    if d["run"].name != stamp:
                        raise RuntimeError(f"newest cnmf/{fset} run is {d['run'].name}, "
                                           f"manifest has {stamp}")
                    d["fset"] = fset
                    Cd = P2.cube(d["X"], d)
                    means = np.stack([Cd[d["lab"] == j].mean(0) for j in range(k)])
                    order = [int(j) for j in P2.block_order(means, d)[0]]
                else:
                    sol = F2.solution(method, fset, k)
                    if sol is None:
                        raise RuntimeError("no solution at this K")
                    if sol["run"].name != stamp:
                        raise RuntimeError(f"newest {method}/{fset} run is "
                                           f"{sol['run'].name}, manifest has {stamp}")
                    if sol["keys"] is not None and sol["keys"] != ref_keys:
                        raise RuntimeError("electrodes differ from the reference's")
                    m, _, _ = P2.match_clusters((sol["G"], sol["lab"]),
                                                (ref["Gn"], ref["lab"]))
                    n = int(sol["lab"].max()) + 1
                    order = sorted(range(n), key=lambda i: (ref_pos.get(int(m[i]), 10**6), i))
                out[rid][str(k)] = order
            except Exception as e:                       # one bad cut, not a dead page
                skipped.append(f"{rid} K={k}: {e}")
        print(f"  K={k:<3} reference order {ref_order}")
    return {"generated": dt.date.today().isoformat(),
            "reference": f"cnmf / {ref_fset} - the figures' matching reference "
                         f"(P2.MATCH_REF_FSET); the reference itself is ordered by "
                         f"cross-condition similarity, as in FIG 1",
            "paper_k": PAPER_K, "sweep_ks": ks, "runs": out, "labels": labels,
            "skipped": skipped}


# ---------------------------------------------------------------------------
# the html
# ---------------------------------------------------------------------------
def build_html(order):
    s = SRC.read_text(encoding="utf-8")
    n_patch = 0

    def sub(old, new, what):
        nonlocal s, n_patch
        c = s.count(old)
        if c != 1:
            raise SystemExit(f"anchor for {what!r} matches {c} times, not once:\n{old[:120]}")
        s = s.replace(old, new)
        n_patch += 1

    ids = list(order["runs"].keys())
    js_ids = ",\n  ".join(f'"{i}"' for i in ids)

    # ---- names -------------------------------------------------------------------
    sub("<title>Clustering Visualizer — sampling &amp; cluster probability</title>",
        "<title>Cluster Visualizer — the paper's runs, in figure order</title>", "title")
    sub('<h2 id="ctlHead">Clustering Visualizer<span',
        '<h2 id="ctlHead">Cluster Visualizer<span', "heading")
    sub('        <select id="runSel"></select>',
        '        <select id="runSel"></select>\n'
        '        <div style="font-size:11px;color:#9aa3ad;margin:5px 0 0;line-height:1.4">'
        'Paper runs only, standard first &middot; clusters in <b>figure order</b> '
        '(#1 = leftmost block in FIG&nbsp;1; FIG&nbsp;2 row order for the other '
        'algorithms) &middot; opens at K&nbsp;=&nbsp;8</div>', "run note")
    sub("Generated from <code>clustering_visualizer.html</code>",
        "Generated from <code>cluster_visualizer.html</code>", "footer")

    # ---- the run list and the figure order -----------------------------------------
    sub('const CLUST  = REPO + "02_FBM_Clustering/outputs/clustering/";',
        'const CLUST  = REPO + "02_FBM_Clustering/outputs/clustering/";\n'
        '\n'
        '// cluster_visualizer.html: the paper\'s runs only, standard first, and every\n'
        '// cluster list passed through figureOrder() - see make_cluster_visualizer.py.\n'
        'const PAPER_K = ' + str(PAPER_K) + ';\n'
        'const PAPER_RUNS = [\n  ' + js_ids + '\n];\n'
        'let ORDER = { runs: {} };\n'
        'function figureOrder(runId, k, ids) {\n'
        '  const o = ORDER.runs[runId] && ORDER.runs[runId][String(k)];\n'
        '  if (!o || !ids || !ids.length) return ids;\n'
        '  const asStr = typeof ids[0] === "string";\n'
        '  const have = new Set(ids.map(Number));\n'
        '  const out = o.map(Number).filter(c => have.has(c));\n'
        '  for (const c of ids.map(Number)) if (!out.includes(c)) out.push(c);\n'
        '  return asStr ? out.map(String) : out;\n'
        '}', "figureOrder")
    sub('  M = await (await fetch(BUNDLE + "manifest.json", fresh)).json();',
        '  M = await (await fetch(BUNDLE + "manifest.json", fresh)).json();\n'
        '  try { ORDER = await (await fetch("cluster_visualizer_order.json", fresh)).json(); }\n'
        '  catch (e) { console.warn("figure order not loaded; natural order", e); }\n'
        '  {\n'
        '    const keep = PAPER_RUNS.map(id => M.runs.find(r => r.id === id)).filter(Boolean);\n'
        '    if (keep.length) M.runs = keep;\n'
        '    else console.warn("none of the paper runs is in the manifest; showing all");\n'
        '  }', "run filter")

    # ---- open at the paper's K -------------------------------------------------------
    sub("  let wantK = Number(r.k);\n",
        "  let wantK = Number(r.k);\n"
        "  // the paper's cut, where the sweep has it\n"
        "  if (!keepK && r.sweep && r.sweep.ks.map(Number).includes(PAPER_K)) wantK = PAPER_K;\n",
        "default K")

    # ---- clusters in figure order, everywhere a cut is listed ------------------------
    sub("    CLU = { lh, rh };\n    RUN = RUN0;\n",
        "    CLU = { lh, rh };\n"
        "    RUN = Object.assign({}, RUN0, { clusters: figureOrder(RUN0.id, KSEL, RUN0.clusters) });\n",
        "applyK published cut")
    sub("      k: KSEL, clusters: st.clusters,\n",
        "      k: KSEL, clusters: figureOrder(RUN0.id, KSEL, st.clusters),\n",
        "applyK re-cut")
    sub("          const ids = (SWEEP.stats[String(kk)] || {}).clusters || [];",
        "          const ids = figureOrder(RUN0.id, kk, (SWEEP.stats[String(kk)] || {}).clusters || []);",
        "chips 1")
    sub("        const st = SWEEP.stats[String(k)] || {};\n        const ids = st.clusters || [];",
        "        const st = SWEEP.stats[String(k)] || {};\n"
        "        const ids = figureOrder(RUN0.id, k, st.clusters || []);", "chips 2")
    sub("    const B = KP.byK[k], st = B.stats || {}, ids = st.clusters || [];",
        "    const B = KP.byK[k], st = B.stats || {}, ids = figureOrder(RUN0.id, k, st.clusters || []);",
        "chips 3")
    sub("      label: `Cluster ${k} — P(cluster ${k} | electrode here)  ·  n=${size}`,",
        "      label: `#${i + 1} · Cluster ${k} — P(cluster ${k} | electrode here)  ·  n=${size}`,",
        "cluster label")

    # ---- no readout box ---------------------------------------------------------------
    # The bottom-left panel that explained the selected map. Hidden, not removed: the
    # render path writes its title, definition and note into it on every redraw, and a
    # missing element would throw there.
    sub("  #readout { bottom:16px; left:14px; width:352px; font-size:11.5px; line-height:1.45; }",
        "  #readout { bottom:16px; left:14px; width:352px; font-size:11.5px; line-height:1.45;\n"
        "             display:none !important; }   /* cluster_visualizer: no readout box */",
        "hide readout")

    # ---- the live centroid panel, where the readout was --------------------------------
    # Ingredients from make_centroid_bundle.py; the arithmetic happens here. Weighted
    # mean and weighted SD over every electrode whose loading on the selected cluster is
    # at or above the minimum-loading slider (graded runs), or the plain member mean
    # (hard runs, and the "paper definition" toggle, which is FIG 1's centroid).
    sub('  <div class="panel" id="readout">\n'
        '    <h3 id="roTitle">—</h3>\n'
        '    <div class="def" id="roDef">—</div>\n'
        '    <div class="note" id="roNote">—</div>\n'
        '  </div>',
        '  <div class="panel" id="readout">\n'
        '    <h3 id="roTitle">—</h3>\n'
        '    <div class="def" id="roDef">—</div>\n'
        '    <div class="note" id="roNote">—</div>\n'
        '  </div>\n'
        '  <div class="panel hidden" id="centroidPanel">\n'
        '    <div id="cpTitle">centroid</div>\n'
        '    <canvas id="cpCanvas" width="380" height="170"></canvas>\n'
        '    <div id="cpNote">—</div>\n'
        '    <label id="cpRepRow">shown on <select id="cpRep"></select> '
        '<span class="hint">— the same clusters, averaged over another representation '
        'of the same electrodes</span></label>\n'
        '    <label id="cpPaperRow"><input type="checkbox" id="cpPaper"> paper definition: '
        'argmax members, unweighted, minimum ignored (as FIG 1)</label>\n'
        '  </div>', "centroid panel markup")
    sub("  #readout b { color:#e8e8ec; }",
        "  #readout b { color:#e8e8ec; }\n"
        "  /* bottom-right, to the left of the colour bar (right:16px, 70px wide), above the hint */\n"
        "  #centroidPanel { bottom:28px; right:100px; left:auto; width:408px; font-size:11.5px; line-height:1.45; }\n"
        "  #centroidPanel #cpTitle { font-size:12px; font-weight:600; color:var(--fg); margin:0 0 6px; }\n"
        "  #centroidPanel canvas { display:block; width:380px; height:170px; border-radius:6px; background:#0f1014; }\n"
        "  #centroidPanel #cpNote { color:var(--muted); margin:6px 0 0; }\n"
        "  #centroidPanel label { display:block; color:var(--muted); margin:6px 0 0; cursor:pointer; }\n"
        "  #centroidPanel select { width:auto; display:inline-block; padding:2px 6px; font-size:11px; margin:0 4px; }\n"
        "  #centroidPanel .hint { color:var(--muted); opacity:.8; }",
        "centroid panel css")
    sub("function figureOrder(runId, k, ids) {", r"""// ---- the live centroid (cluster_visualizer) ---------------------------------------
// The selected cluster's centroid, recomputed here from what make_centroid_bundle.py
// writes: a loading-weighted mean over every electrode whose loading on the cluster is
// at or above the minimum-loading slider. The DATA it is averaged over can be any of
// the four feature sets - the bundle proves every run lists the same electrodes in the
// same order - so a clustering found on high gamma can be looked at on the 5-band
// representation of the same electrodes. Dispersion (+/-1 SD, weighted) is drawn for
// the single-band HFA line only; the band heatmaps show the mean. Every scale is GLOBAL
// - the same for every cluster of a run at a K - and labelled with its unit.
const CENTROIDS = REPO + "02_FBM_Clustering/outputs/clustering/paper_web/centroids/";
const REP_LABEL = { concat_bands5z: "5 bands, z-scored", concat_hg: "HFA (70-150 Hz)",
                    concat_rawds: "15 bands", concat_bands5: "5 bands" };
let CEN = null;                 // {meta, W: Uint16Array|null} for RUN0
let CEN_INDEX = null;           // index.json: geometry per feature set, the alignment proof
let CEN_REP = null;             // the feature set the centroid is drawn ON
const _cenX = new Map();        // feature set -> Int16Array, shared by every run
const _cenLim = new Map();      // FIG 1's colour limits per run, representation and K
async function cenIndex() {
  if (!CEN_INDEX) CEN_INDEX = await (await fetch(CENTROIDS + "index.json", { cache: "no-store" })).json();
  return CEN_INDEX;
}
async function cenX(fset) {
  let X = _cenX.get(fset);
  if (!X) { X = new Int16Array(await (await fetch(CENTROIDS + "x_" + fset + ".bin")).arrayBuffer()); _cenX.set(fset, X); }
  return X;
}
async function loadCentroids(runId) {
  CEN = null;
  try {
    const idx = await cenIndex();
    const meta = await (await fetch(CENTROIDS + runId + ".json", { cache: "no-store" })).json();
    const W = meta.w_file ? new Uint16Array(await (await fetch(CENTROIDS + meta.w_file)).arrayBuffer()) : null;
    CEN = { meta, W };
    // "shown on": the run's own representation, plus every other one the bundle proved
    // row-aligned with it. The choice is kept across runs.
    const own = meta.feature_set;
    const fsets = Object.keys(idx.feature_sets).filter(f => f === own || idx.rows_aligned);
    const sel = $("cpRep");
    sel.innerHTML = fsets.map(f => `<option value="${f}">${REP_LABEL[f] || f}</option>`).join("");
    if (!fsets.includes(CEN_REP)) CEN_REP = own;
    sel.value = CEN_REP;
    $("cpRepRow").style.display = fsets.length > 1 ? "" : "none";
    await cenX(CEN_REP);
  } catch (e) { console.warn("no live centroid for", runId, e); }
  drawCentroid();
}
function cenGeom() {
  const g = CEN_INDEX.feature_sets[CEN_REP];
  return { n: g.n, nc: g.conds.length, nb: g.bands.length, nt: g.nt, conds: g.conds,
           F: g.conds.length * g.bands.length * g.nt,
           unit: g.unit || "dB", unitLong: g.unit_long || "" };
}
// weights per electrode for cluster j at K: the loading (graded) or membership (hard)
function cenWeights(k, j, useMin, paper) {
  const m = CEN.meta, n = m.n, w = new Float32Array(n), lab = m.labels[String(k)];
  let nMembers = 0, nPass = 0;
  if (paper || !CEN.W || m.w_offsets[String(k)] === undefined) {
    if (!lab) return null;
    for (let i = 0; i < n; i++) if (lab[i] === j) { w[i] = 1; nMembers++; nPass++; }
    return { w, nMembers, nPass, weighted: false };
  }
  const off = m.w_offsets[String(k)], min = useMin ? state.minLoad : 0;
  for (let i = 0; i < n; i++) {
    const v = CEN.W[off + i * k + j] * m.w_scale;
    if (lab && lab[i] === j) nMembers++;
    if (v > 0 && v >= min) { w[i] = v; nPass++; }
  }
  return { w, nMembers, nPass, weighted: true };
}
// the weighted mean of the chosen representation over the electrodes cenWeights picks;
// the weighted SD too, but only for the single-band line, where it is drawn
function cenMean(k, j, useMin, paper) {
  const g = cenGeom(), X = _cenX.get(CEN_REP);
  if (!X || X.length !== g.n * g.F || g.n !== CEN.meta.n) return null;
  const r = cenWeights(k, j, useMin, paper); if (!r) return null;
  const mean = new Float32Array(g.F), sd = g.nb === 1 ? new Float32Array(g.F) : null;
  let sw = 0; for (let i = 0; i < g.n; i++) sw += r.w[i];
  if (sw <= 0) return Object.assign(r, { mean, sd, empty: true });
  const s = CEN_INDEX.x_scale;
  for (let i = 0; i < g.n; i++) { const wi = r.w[i]; if (!wi) continue; const b = i * g.F; for (let f = 0; f < g.F; f++) mean[f] += wi * X[b + f]; }
  for (let f = 0; f < g.F; f++) mean[f] = mean[f] * s / sw;
  if (sd) {
    for (let i = 0; i < g.n; i++) { const wi = r.w[i]; if (!wi) continue; const b = i * g.F; for (let f = 0; f < g.F; f++) { const d = X[b + f] * s - mean[f]; sd[f] += wi * d * d; } }
    for (let f = 0; f < g.F; f++) sd[f] = Math.sqrt(sd[f] / sw);
  }
  return Object.assign(r, { mean, sd, empty: false });
}
// GLOBAL scales, the same for every cluster of a run at this K on this representation,
// so clusters are comparable. Heatmap: symmetric colour limit at the 99th percentile
// of |argmax mean| over all clusters (FIG 1's rule). Line: the range of every cluster's
// mean +/- SD (FIG 1's rule), so the band never clips.
function cenLimits(k) {
  const key = CEN.meta.id + ":" + CEN_REP + ":" + k; if (_cenLim.has(key)) return _cenLim.get(key);
  const vals = []; let lo = 0, hi = 0;
  if (CEN.meta.labels[String(k)]) for (let j = 0; j < k; j++) {
    const st = cenMean(k, j, false, true); if (!st || st.empty) continue;
    for (let f = 0; f < st.mean.length; f++) {
      const v = st.mean[f], s = st.sd ? st.sd[f] : 0; vals.push(Math.abs(v));
      if (v - s < lo) lo = v - s; if (v + s > hi) hi = v + s;
    }
  }
  vals.sort((a, b) => a - b);
  const v = vals.length ? vals[Math.min(vals.length - 1, Math.floor(0.99 * vals.length))] : 1;
  const lim = { vlim: v || 1, ylo: lo * 1.06, yhi: hi * 1.06 }; _cenLim.set(key, lim); return lim;
}
// matplotlib RdBu, red to blue; the figures use RdBu_r, so t is flipped
const RDBU = [[103,0,31],[178,24,43],[214,96,77],[244,165,130],[253,219,199],[247,247,247],[209,229,240],[146,197,222],[67,147,195],[33,102,172],[5,48,97]];
function rdbu_r(t) {
  t = Math.max(0, Math.min(1, 1 - t));
  const x = t * (RDBU.length - 1), i = Math.min(RDBU.length - 2, Math.floor(x)), f = x - i, a = RDBU[i], b = RDBU[i + 1];
  return [a[0] + (b[0]-a[0])*f, a[1] + (b[1]-a[1])*f, a[2] + (b[2]-a[2])*f];
}
function drawCentroid() {
  const P = $("centroidPanel"); if (!P) return;
  const d = curDef();
  if (!CEN || !CEN_INDEX || !d || d.cluster === undefined) { P.classList.add("hidden"); return; }
  const m = CEN.meta, k = Number(KSEL), j = Number(d.cluster);
  if (!m.ks.includes(k)) { P.classList.add("hidden"); return; }
  const paper = !!$("cpPaper").checked;
  const st = cenMean(k, j, true, paper);
  if (!st) { P.classList.add("hidden"); return; }
  P.classList.remove("hidden");
  const g = cenGeom(), own = m.feature_set, cross = CEN_REP !== own;
  const pos = (RUN && RUN.clusters) ? RUN.clusters.map(Number).indexOf(j) + 1 : 0;
  $("cpTitle").textContent = `#${pos || "?"} · Cluster ${j} · ` + (st.weighted
    ? `n = ${st.nPass} of ${st.nMembers} members ≥ ${state.minLoad.toFixed(2)}`
    : `n = ${st.nMembers} members`) + (cross ? ` · on ${REP_LABEL[CEN_REP]}` : "");
  const sdTxt = g.nb === 1 ? " · band = ±1 SD" + (paper ? "" : " (weighted)") : "";
  $("cpNote").textContent = (paper
    ? "paper definition: mean over argmax members, unweighted — FIG 1's centroid"
    : CEN.W ? "loading-weighted mean over every electrode with loading ≥ the minimum"
            : "hard partition: mean over members; the minimum does not apply")
    + sdTxt + ` · scale global across clusters at K=${k}`
    + (cross ? ` · clusters from ${REP_LABEL[own]}, data from ${REP_LABEL[CEN_REP]}` : "");
  $("cpPaperRow").style.display = CEN.W ? "" : "none";
  const cv = $("cpCanvas"), ctx = cv.getContext("2d"), Wd = cv.width, Hd = cv.height;
  const lim = cenLimits(k), nc = g.nc, nb = g.nb, nt = g.nt, ncol = nc * nt, col = clusterCSS(j);
  // the single-band HFA line is drawn on white, as the figures are; the heatmaps stay
  // on the page's dark ground, where RdBu's white zero reads as "nothing"
  const line = nb === 1;
  const T = line ? { bg: "#ffffff", fg: "#1b232c", muted: "#68727d", grid: "#c9ced4", sep: "#1b232c", cue: "#68727d" }
                 : { bg: "#0f1014", fg: "#e8e8ec", muted: "#9aa3ad", grid: "#3a3f47", sep: "#e8e8ec", cue: "#c9ced4" };
  ctx.setLineDash([]); ctx.fillStyle = T.bg; ctx.fillRect(0, 0, Wd, Hd);
  // a right gutter carries the GLOBAL scale: a colour bar for the heatmap, a y axis for
  // the line, both labelled with the unit - the same for every cluster at this K
  const padL = line ? 40 : 6, padR = line ? 8 : 46, padT = 8, padB = 16, w = Wd - padL - padR, h = Hd - padT - padB;
  const unit = g.unit;
  ctx.font = "9px sans-serif"; ctx.fillStyle = T.muted;
  if (st.empty) { ctx.font = "11px sans-serif"; ctx.textAlign = "left"; ctx.fillText("no electrode passes the minimum", padL + 8, Hd / 2); return; }
  if (!line) {
    const img = ctx.createImageData(ncol, nb);
    for (let b = 0; b < nb; b++) for (let c = 0; c < nc; c++) for (let t = 0; t < nt; t++) {
      const f = (c * nb + b) * nt + t, rgb = rdbu_r((st.mean[f] + lim.vlim) / (2 * lim.vlim));
      const o = ((nb - 1 - b) * ncol + c * nt + t) * 4;           // origin lower, as imshow
      img.data[o] = rgb[0]; img.data[o+1] = rgb[1]; img.data[o+2] = rgb[2]; img.data[o+3] = 255;
    }
    const off = document.createElement("canvas"); off.width = ncol; off.height = nb;
    off.getContext("2d").putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false; ctx.drawImage(off, padL, padT, w, h);
    // colour bar: -vlim (blue) at the bottom to +vlim (red) at the top
    const bx = Wd - padR + 8, bw = 9;
    for (let y = 0; y < h; y++) { const rgb = rdbu_r(1 - y / (h - 1)); ctx.fillStyle = `rgb(${rgb[0]|0},${rgb[1]|0},${rgb[2]|0})`; ctx.fillRect(bx, padT + y, bw, 1); }
    ctx.strokeStyle = T.grid; ctx.lineWidth = 1; ctx.strokeRect(bx + 0.5, padT + 0.5, bw - 1, h - 1);
    ctx.fillStyle = T.muted; ctx.textAlign = "left";
    ctx.fillText(`+${lim.vlim.toFixed(1)}`, bx + bw + 3, padT + 8);
    ctx.fillText("0", bx + bw + 3, padT + h / 2 + 3);
    ctx.fillText(`−${lim.vlim.toFixed(1)}`, bx + bw + 3, padT + h - 1);
    ctx.save(); ctx.translate(Wd - 3, padT + h / 2); ctx.rotate(Math.PI / 2); ctx.textAlign = "center"; ctx.fillText(unit, 0, 0); ctx.restore();
  } else {
    const lo = lim.ylo, hi = lim.yhi, sy = v => padT + h * (1 - (v - lo) / (hi - lo));
    // y axis with the global range and the unit
    ctx.strokeStyle = T.grid; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(padL - 0.5, padT); ctx.lineTo(padL - 0.5, padT + h); ctx.stroke();
    ctx.fillStyle = T.muted; ctx.textAlign = "right";
    for (const v of [hi, 0, lo]) { const y = sy(v); ctx.beginPath(); ctx.moveTo(padL - 4, y + 0.5); ctx.lineTo(padL - 0.5, y + 0.5); ctx.stroke(); ctx.fillText(v.toFixed(1), padL - 6, y + 3); }
    ctx.save(); ctx.translate(9, padT + h / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.fillText(unit, 0, 0); ctx.restore();
    ctx.strokeStyle = T.grid; ctx.beginPath(); ctx.moveTo(padL, sy(0) + 0.5); ctx.lineTo(padL + w, sy(0) + 0.5); ctx.stroke();
    for (let c = 0; c < nc; c++) {
      const x0 = padL + w * c / nc, xw = w / nc, xt = t => x0 + xw * t / (nt - 1);
      if (st.sd) {                                                // +/-1 SD, weighted - the line only
        ctx.fillStyle = col; ctx.globalAlpha = 0.22; ctx.beginPath();
        for (let t = 0; t < nt; t++) { const f = c * nt + t; if (!t) ctx.moveTo(xt(t), sy(st.mean[f] + st.sd[f])); else ctx.lineTo(xt(t), sy(st.mean[f] + st.sd[f])); }
        for (let t = nt - 1; t >= 0; t--) { const f = c * nt + t; ctx.lineTo(xt(t), sy(st.mean[f] - st.sd[f])); }
        ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
      }
      ctx.strokeStyle = col; ctx.lineWidth = 1.4; ctx.beginPath();
      for (let t = 0; t < nt; t++) { const f = c * nt + t; if (!t) ctx.moveTo(xt(t), sy(st.mean[f])); else ctx.lineTo(xt(t), sy(st.mean[f])); }
      ctx.stroke();
    }
  }
  for (let c = 0; c < nc; c++) {                                  // blocks and the GO cue at 50%
    const x0 = padL + w * c / nc, xm = x0 + w / nc / 2;
    if (c) { ctx.strokeStyle = T.sep; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0 + 0.5, padT); ctx.lineTo(x0 + 0.5, padT + h); ctx.stroke(); }
    ctx.strokeStyle = T.cue; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(xm + 0.5, padT); ctx.lineTo(xm + 0.5, padT + h); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = T.muted; ctx.font = "10px sans-serif"; ctx.textAlign = "center"; ctx.fillText(g.conds[c], xm, Hd - 4);
  }
}
// NOTHING RUNS HERE AT LOAD: this block sits above the line that defines $, and a call
// at module evaluation threw before boot() could start. The controls are wired in wireUI.

function figureOrder(runId, k, ids) {""", "centroid panel js")
    sub('  autoScale(); render(); updateElectrodes();\n  $("status").style.display = "none";\n}',
        '  autoScale(); render(); updateElectrodes();\n  $("status").style.display = "none";\n'
        '  loadCentroids(r.id);                 // the panel follows; it draws when its data lands\n}',
        "centroid: on run select")
    sub("        refreshMapList();\n        autoScale(); render(); updateElectrodes();\n      } finally",
        "        refreshMapList();\n        autoScale(); render(); updateElectrodes(); drawCentroid();\n      } finally",
        "centroid: on K")
    sub('  $("mapSel").onchange = e => { state.map = e.target.value; autoScale(); render(); updateElectrodes(); };',
        '  $("mapSel").onchange = e => { state.map = e.target.value; autoScale(); render(); updateElectrodes(); drawCentroid(); };\n'
        '  $("cpPaper").onchange = drawCentroid;   // the centroid panel\'s "paper definition" toggle\n'
        '  $("cpRep").onchange = async e => { CEN_REP = e.target.value; await cenX(CEN_REP); drawCentroid(); };',
        "centroid: on map")
    sub('    $("minLoadVal").textContent = state.minLoad.toFixed(2);\n    updateElectrodes();\n  };',
        '    $("minLoadVal").textContent = state.minLoad.toFixed(2);\n    updateElectrodes(); drawCentroid();\n  };',
        "centroid: on minimum loading")

    # ---- the selected cluster follows its POSITION across runs ------------------------
    # Positions are matched to the paper's reference (convex NMF on bands5) for every
    # run, so #1 on the HFA run and #1 on the 5-band run are the Hungarian-matched pair.
    # Switching runs used to keep the cluster ID, which is arbitrary across runs; it now
    # keeps the position, so the matched cluster stays on screen.
    sub("  RUN0 = r; RUN = r; SWEEP = null; RUNCOV = null;\n",
        "  const keepPos = (RUN && RUN.clusters && /^c\\d+$/.test(state.map))\n"
        "    ? RUN.clusters.map(Number).indexOf(Number(state.map.slice(1))) : -1;\n"
        "  RUN0 = r; RUN = r; SWEEP = null; RUNCOV = null;\n", "position: remember")
    sub("  await applyK(wantK);\n  location.hash = r.id;\n",
        "  await applyK(wantK);\n"
        "  if (keepPos >= 0 && RUN.clusters && RUN.clusters[keepPos] !== undefined)\n"
        "    state.map = \"c\" + RUN.clusters[keepPos];       // same block position, new run\n"
        "  location.hash = r.id;\n", "position: restore")

    # ---- coverage: hidden in the dropdown, gone from the report ----------------------
    sub('    id: "coverage", group: "Sampling", label: "Coverage — P(patient sampled here)",',
        '    id: "coverage", hidden: true, group: "Sampling", label: "Coverage — P(patient sampled here)",',
        "hide coverage")
    sub('    id: "density", group: "Sampling", label: "Coverage — contact density", raw: true,',
        '    id: "density", hidden: true, group: "Sampling", label: "Coverage — contact density", raw: true,',
        "hide density")
    sub("  for (const d of DEFS) {\n    if (d.group !== g)",
        "  for (const d of DEFS) {\n    if (d.hidden) continue;\n    if (d.group !== g)",
        "dropdown skips hidden")
    sub('  if (!DEFS.some(d => d.id === keep)) state.map = "coverage";',
        '  const visible = DEFS.filter(d => !d.hidden);\n'
        '  if (!visible.some(d => d.id === keep)) state.map = visible.length ? visible[0].id : "coverage";',
        "dropdown fallback")

    # the report: no coverage or density captures ...
    i = s.index("    // ── coverage\n    btn.textContent = \"Report: coverage…\";")
    j = s.index("    const covMax = frozen.climMax;\n") + len("    const covMax = frozen.climMax;\n")
    s = s[:i] + ("    // coverage and contact density are not part of this report\n"
                 "    state.climMax = frozen.climMax;\n") + s[j:]
    n_patch += 1
    # ... and no sampling section; the glass-by-cluster turn stays
    i = s.index('<section class="card">\n  <h2 class="sec">How well was this cohort sampled?</h2>')
    j = s.index('    <div>\n      <h2 class="sec">The same cohort, glass</h2>')
    s = s[:i] + '<section class="card">\n  <div class="three">\n' + s[j:]
    n_patch += 1
    sub('        framed in the same box as every other turn in this report. The two panels to the\n'
        '        left answer "how much of cortex is estimable" and "where are the electrodes";\n'
        '        this one answers "which cluster is where", on one brain rather than ${K} apart.</p>',
        '        framed in the same box as every other turn in this report. It answers "which\n'
        '        cluster is where", on one brain rather than ${K} apart.</p>', "glass hint")

    # ---- report images: de-duplicate on the FINISHED document, not at build time ------
    # srcAttr() registered an image the first time any fragment was built and emitted a
    # data-mirror for every later identical one. The K panel builds the current cut's
    # cells early and that HTML is rebuilt later, so the first registration never
    # reached the file and 40 mirrors in a real report pointed at nothing - every
    # centroid, every contacts still, some dorsals. Emitting full srcs and de-duplicating
    # the assembled document in document order makes a dangling mirror impossible: the
    # source is, by construction, the first occurrence in the file.
    sub("function srcAttr(data) {\n"
        "  const prev = _srcSeen.get(data);\n"
        "  if (prev) return `data-mirror=\"${prev}\"`;\n"
        "  const id = \"i\" + (++_srcN);\n"
        "  _srcSeen.set(data, id);\n"
        "  return `src=\"${data}\" data-src-id=\"${id}\"`;\n"
        "}",
        "function srcAttr(data) {\n"
        "  // full src always; dedupeImages() turns later duplicates into mirrors once the\n"
        "  // whole document exists, so a mirror can only ever point at an earlier tag\n"
        "  return `src=\"${data}\"`;\n"
        "}\n"
        "function dedupeImages(doc) {\n"
        "  const seen = new Map(); let n = 0;\n"
        "  return doc.replace(/<img\\b([^>]*?)\\ssrc=\"(data:[^\"]+)\"([^>]*)>/g, (m, pre, data, post) => {\n"
        "    const id = seen.get(data);\n"
        "    if (id) return `<img${pre} data-mirror=\"${id}\"${post}>`;\n"
        "    const nid = \"i\" + (++n); seen.set(data, nid);\n"
        "    return `<img${pre} src=\"${data}\" data-src-id=\"${nid}\"${post}>`;\n"
        "  });\n"
        "}", "srcAttr -> post-pass dedupe")
    sub("    const blob = new Blob([html], { type: \"text/html;charset=utf-8\" });",
        "    const blob = new Blob([dedupeImages(html)], { type: \"text/html;charset=utf-8\" });",
        "dedupe before save")

    left = re.findall(r"\b(covShots|covSpin|densSpin|covStats|covMax)\b", s)
    if left:
        raise SystemExit(f"coverage report variables still referenced: {sorted(set(left))}")
    return s, n_patch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html-only", action="store_true")
    ap.add_argument("--json-only", action="store_true")
    a = ap.parse_args()
    if not SRC.exists():
        raise SystemExit(f"no {SRC}")
    if not MANIFEST.exists():
        raise SystemExit(f"no {MANIFEST}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    if not a.html_only:
        order = compute_order(manifest)
        ORDER_JSON.write_text(json.dumps(order, indent=1), encoding="utf-8")
        n_cuts = sum(len(v) for v in order["runs"].values())
        print(f"  {len(order['runs'])} runs, {n_cuts} cuts ordered -> {ORDER_JSON.name}")
        for x in order["skipped"]:
            print(f"  skipped  {x}")
    else:
        order = json.loads(ORDER_JSON.read_text(encoding="utf-8"))

    if not a.json_only:
        html, n = build_html(order)
        DST.write_text(html, encoding="utf-8")
        back = DST.read_text(encoding="utf-8")
        if back != html:
            raise SystemExit("read-back differs from what was written")
        print(f"  {n} patches -> {DST.name}  ({len(SRC.read_text(encoding='utf-8')):,} -> "
              f"{len(html):,} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
