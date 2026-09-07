#!/usr/bin/env python3
"""
make_cluster_visualizer.py - cluster_visualizer.html: the paper's runs, in figure order.

    python make_cluster_visualizer.py              order json + html
    python make_cluster_visualizer.py --html-only  re-patch the html, keep the json
    python make_cluster_visualizer.py --json-only  recompute the order, keep the html

A COPY OF clustering_visualizer.html WITH FIVE CHANGES AND NO OTHERS.

1. Only the runs the paper uses. The manifest lists 21 runs; the copy keeps the ones a
   paper figure is built from - convex NMF on the four concat feature sets (FIG 1, FIG 3,
   FIG 2's feature-set half) and k-means and Ward on those sets (FIG 2's algorithm
   half) - in the paper's order, the standard (5 bands z-scored) first. The
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

4. A COMPARE UI instead of the two dropdowns (2026-09-07). The runs become a permanent
   feature-set x algorithm MATRIX and the clusters a STRIP of chips in figure order, so
   what else there is to compare is on the panel rather than inside a menu. Two views do
   the comparing: OVERLAY draws several clusters on one brain, and GRID draws small
   multiples captured from the same canvas at one camera angle - every cluster of a run,
   or one position across the three algorithms or the four feature sets. Switching run
   keeps K and the POSITION, so you land on the matched cluster (FIG 2's correspondence),
   not on an unrelated cluster id. Glass is the only surface now: render() returns before
   painting a vertex in glass mode, so the surface buttons, the neighbourhood radius, the
   colour ceiling, the contrast and the min-n vertex gate were dead controls - they stay
   in the DOM, because render(), autoScale() and the report write into them, and are
   hidden. UI2_CSS and UI2_JS below hold the whole of it.

5. NO BRAIN CAPTURES IN THE REPORT (REPORT_3D = False). The three stills and the 360
   turn per cluster, the two whole-cohort turns and every cluster at every K are not
   captured and the sections built from them are not written. Everything else in the
   report is untouched - centroids, rasters, composition bars, K panel, published
   figures - so it is still one self-contained file per run, at about a tenth of the
   size and seconds rather than minutes. REPORT_3D = True restores all of it.

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
# The report's BRAIN CAPTURES. False (2026-09-07): the stills, the 360 turns and the two
# sections built from them leave the report - "remove all 3D renderings, it is not needed
# for the cluster card until I decide to put them back". Everything else in the report is
# untouched: the centroids, the rasters, the composition bars, the K panel, the published
# figures. True brings every render back and nothing else moves.
REPORT_3D = False
FSETS = ["concat_bands5z", "concat_hg", "concat_rawds", "concat_bands5"]   # standard first
METHODS = ["cnmf", "kmeans", "hierarchical"]          # archetypal analysis dropped 2026-09-06
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
# the compare UI: a run matrix, a cluster strip, an overlay view and a grid view
# ---------------------------------------------------------------------------
UI2_CSS = r"""
  /* ── compare UI — make_cluster_visualizer.py ──────────────────────────── */
  #controls { width:304px; }
  /* the panel is taller than the one it replaces; on a short window it scrolls rather
     than clipping the report button. Collapsed it keeps overflow:hidden, so the
     max-height animation still works. */
  #controls:not(.collapsed) #ctlBody { overflow-y:auto; }
  #cbar { display:none !important; }      /* glass paints no vertex map to put a scale on */
  #ui2 .u2g { margin:12px 0 0; }
  #ui2 .u2t { font-size:11px; text-transform:uppercase; letter-spacing:.6px;
              color:var(--muted); margin-bottom:5px; }
  #ui2 .u2t .u2sub { text-transform:none; letter-spacing:0; opacity:.75; }
  #ui2 .u2mx { display:grid; grid-template-columns:68px repeat(3, 1fr); gap:3px; }
  #ui2 .u2hd { font-size:10px; color:var(--muted); text-align:center; padding-bottom:2px; }
  #ui2 .u2rl { font-size:10.5px; color:var(--muted); display:flex; align-items:center;
               white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  #ui2 .u2rl.on { color:var(--fg); }
  #ui2 .u2c { height:22px; border-radius:5px; background:#15161b; cursor:pointer; padding:0;
              border:1px solid rgba(255,255,255,.14); color:var(--muted);
              font:inherit; font-size:11px; line-height:1; }
  #ui2 .u2c:hover { border-color:var(--accent); color:var(--fg); }
  #ui2 .u2c.on { background:var(--accent); border-color:var(--accent); color:#001; font-weight:700; }
  #ui2 .u2c.off { cursor:default; opacity:.28; border-style:dashed; }
  #ui2 .u2chips { display:grid; grid-template-columns:repeat(4, 1fr); gap:4px; }
  #ui2 .u2ch { display:flex; align-items:center; justify-content:center; gap:4px; height:24px;
               border-radius:6px; background:#15161b; cursor:pointer; padding:0 4px;
               border:1px solid rgba(255,255,255,.14); color:var(--fg); font:inherit; font-size:11px; }
  #ui2 .u2ch .sw { width:9px; height:9px; border-radius:2px; background:var(--c); flex:none; }
  #ui2 .u2ch i { font-style:normal; font-size:9.5px; color:var(--muted); }
  #ui2 .u2ch:hover { border-color:var(--accent); }
  #ui2 .u2ch.on { background:#1d2029; border-color:var(--c); box-shadow:inset 0 0 0 1px var(--c); }
  #ui2 .u2ch.on i { color:#cdd3dc; }
  #ui2 .segs button { padding:4px 0; font-size:11.5px; }
  #ui2 #u2gridby button { font-size:10.5px; }
  #ui2 .cohort { margin-top:5px; }
  #ui2 #kRow .ksub { display:none; }      /* the group already carries the heading */
  #ui2.u2busy { opacity:.55; pointer-events:none; }

  /* The grid of small multiples, over the canvas. The controls stay on top of it and
     usable - changing run, K or cluster re-captures it - so the panels start clear of
     them; the centroid and the provenance box are about the single selected cluster and
     step aside while the grid is up. */
  #u2grid { position:fixed; inset:0; z-index:4; background:rgba(0,0,0,.94);
            display:none; grid-template-rows:auto 1fr; padding:12px 14px 14px 332px; }
  #u2grid.on { display:grid; }
  body.u2gridon #centroidPanel, body.u2gridon #prov { display:none !important; }
  #u2gh { display:flex; align-items:center; gap:8px; flex-wrap:wrap; padding:0 0 9px;
          border-bottom:1px solid rgba(255,255,255,.08); }
  #u2gh .t { font-size:13px; font-weight:600; }
  #u2gh .s { font-size:11px; color:var(--muted); }
  #u2gh .sp { margin-left:auto; }
  #u2gh button { background:#15161b; color:var(--muted); font:inherit; font-size:11px;
                 border:1px solid rgba(255,255,255,.14); border-radius:6px;
                 padding:4px 9px; cursor:pointer; }
  #u2gh button:hover { color:var(--fg); border-color:var(--accent); }
  #u2gg { overflow:auto; display:grid; gap:10px; padding-top:10px; align-content:start;
          grid-template-columns:repeat(auto-fill, minmax(212px, 1fr)); }
  #u2gg figure { margin:0; cursor:pointer; border-radius:9px; overflow:hidden; background:#000;
                 border:1px solid rgba(255,255,255,.10); }
  #u2gg figure:hover { border-color:var(--accent); }
  #u2gg figure.on { border-color:var(--c); box-shadow:inset 0 0 0 1px var(--c); }
  #u2gg img { display:block; width:100%; height:auto; }
  #u2gg .cap { display:flex; align-items:center; gap:6px; padding:5px 7px 6px; font-size:11px;
               border-top:1px solid rgba(255,255,255,.07); }
  #u2gg .cap .sw { width:9px; height:9px; border-radius:2px; background:var(--c); flex:none; }
  #u2gg .cap .lab { color:var(--muted); font-size:10px; margin-left:auto;
                    font-variant-numeric:tabular-nums; }
  #u2gg .miss { display:grid; place-items:center; height:118px; color:var(--muted);
                font-size:11px; text-align:center; padding:8px; }
"""

# Appended to the page's own <script>, so it shares scope with everything above it:
# state, nv, CONTACTS, DEFS, RUN, KSEL, labelsForK, _OCT, captureView, figureOrder.
UI2_JS = r'''
// ─────────────────────────────────────────────────────────────────────────────
// COMPARE UI — cluster_visualizer.html only; built by make_cluster_visualizer.py.
//
// WHAT CHANGED AND WHY. The Run dropdown held twelve runs and the Visualization
// dropdown held K clusters, so "what do the other two algorithms do with this
// cluster" meant opening two menus and remembering what the last one looked like.
// Both are permanent now: the runs as a feature-set × algorithm MATRIX (along a row
// to change algorithm, down a column to change feature set) and the clusters as a
// STRIP of chips in figure order. Two views do the comparing — OVERLAY draws several
// clusters on one brain, GRID draws small multiples from this same canvas at one
// camera angle, across the clusters of a run or across the algorithms / feature sets
// at one position.
//
// POSITION, NOT CLUSTER ID, travels between runs. "#3" is the third block in FIG 1,
// and figureOrder() gives every run the order its clusters were matched into (FIG 2's
// row order). Switching algorithm keeps the position, so what appears is that
// cluster's counterpart rather than cluster id 3 of an unrelated numbering.
//
// GLASS ONLY. render() returns before painting a vertex when state.glass is set, so
// the surface buttons, the neighbourhood radius, the colour ceiling, the contrast and
// the min-n vertex gate changed nothing that was on screen. They stay in the DOM —
// render(), autoScale() and the report all write into them — and are hidden.
//
// Nothing here recomputes a clustering. Every dot is the page's own contact geometry
// and every panel is a capture of this canvas, so a grid cell is exactly what the big
// view shows at that angle.
// ─────────────────────────────────────────────────────────────────────────────
const UI2 = { on:false, mode:"single", sel:new Set(), gridBy:"clusters",
              busy:false, gridOpen:false, seq:0 };
const U2_METHODS = [["cnmf", "cNMF"], ["kmeans", "k-means"], ["hierarchical", "Ward"]];
const U2_MLABEL  = { cnmf:"cNMF", kmeans:"k-means", hierarchical:"Ward" };
const U2_FSETS   = [["concat_bands5z", "5 bands z"], ["concat_hg", "HFA"],
                    ["concat_rawds", "15 bands"], ["concat_bands5", "5 bands"]];
const u2esc = t => String(t == null ? "" : t)
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const u2Split = id => { const p = String(id).split("__"); return { method:p[0], fset:p[1] }; };
const u2Run   = (m, f) => ((M && M.runs) || []).find(r => r.id.indexOf(m + "__" + f + "__") === 0);
const u2Defs  = () => DEFS.filter(d => d.cluster !== undefined && !d.hidden);
const u2Pos   = () => { const i = u2Defs().findIndex(d => d.id === state.map); return i < 0 ? 0 : i; };
const u2Size  = c => (RUN && RUN.cluster_sizes && RUN.cluster_sizes[String(c)]) || 0;

// clusterRGB() indexed by POSITION rather than by cluster id, so #3 is the same colour
// in every run of the matrix — which is what makes a cross-run grid readable. For the
// current run the two agree exactly: DEFS iterates RUN.clusters, already in figure order.
function u2Hue(pos, K) {
  const h = ((360 * pos) / (K || 1)) / 360, s2 = 0.62, l = 0.52;
  const k = n => (n + h * 12) % 12, a = s2 * Math.min(l, 1 - l);
  const f = n => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  return [Math.round(255 * f(0)), Math.round(255 * f(8)), Math.round(255 * f(4))];
}
const u2CSS = c => `rgb(${c[0]},${c[1]},${c[2]})`;

// ── the contact geometry, with the cluster set as an argument ────────────────
// updateElectrodes() draws ONE cluster of the current run. The overlay and every grid
// cell need the same dots for an arbitrary (labels, loadings, set of clusters), so the
// loop lives here once: the same icosphere, the same loading gate, the same size and
// grey ramp. A contact in more than one SELECTED cluster is drawn white and larger —
// the overlap is what the overlay exists to show.
function u2Mesh(lab, load, want, colourOf, minLoad, cohortId) {
  if (!CONTACTS || !lab || !want || !want.size) return null;
  const pts = [], tris = [], cols = [];
  let vi = 0;
  for (const i of (CONTACTS.cohorts[cohortId] || [])) {
    const L = lab[i];
    if (!Array.isArray(L) || !L.length) continue;
    if (HEMI_ONLY && CONTACTS.hemi[i] !== HEMI_ONLY) continue;
    const hit = L.filter(c => want.has(c));
    if (!hit.length) continue;
    let colr = colourOf(hit) || [255, 232, 64];
    let w = 1, r = 1.73;
    if (load) {
      const lv = load[i];
      if (lv === null || lv === undefined) continue;
      if (lv < minLoad) continue;
      w = Math.max(0, Math.min(1, (lv - 0.14) / (0.85 - 0.14)));
      r = 1.05 + 2.18 * w;
      if (lv < 0.5) {
        const g = Math.max(0, Math.min(1, (lv - 0.14) / (0.5 - 0.14))), G = 148;
        colr = [Math.round(G + (colr[0] - G) * g), Math.round(G + (colr[1] - G) * g),
                Math.round(G + (colr[2] - G) * g)];
      }
    }
    if (hit.length > 1) { colr = [255, 255, 255]; r = Math.max(r, 2.5); }
    const [x, y, z] = CONTACTS.xyz[i];
    for (const o of _OCT) pts.push(x + o[0] * r, y + o[1] * r, z + o[2] * r);
    for (const f of _OCTF) tris.push(vi + f[0], vi + f[1], vi + f[2]);
    const al = load ? Math.round(90 + 165 * w) : 255;
    for (let j = 0; j < _OCT.length; j++) cols.push(colr[0], colr[1], colr[2], al);
    vi += _OCT.length;
  }
  if (!vi) return null;
  try {
    return new NVMesh(new Float32Array(pts), new Uint32Array(tris), "electrodes",
                      new Uint8Array(cols), 1.0, true, nv.gl);
  } catch (e) { console.error("contact overlay failed:", e); return null; }
}

// OVERLAY. updateElectrodes() is called from a dozen places, so the mode is read here
// rather than at every call site; single mode falls straight through to the page's own.
const u2SingleElectrodes = updateElectrodes;
updateElectrodes = function () {
  if (!UI2.on || UI2.mode !== "overlay" || !UI2.sel.size) return u2SingleElectrodes();
  if (elecMesh) { try { nv.removeMesh(elecMesh); } catch (e) {} elecMesh = null; }
  const defs = u2Defs(), K = defs.length;
  const pos = new Map(defs.map((d, i) => [d.cluster, i]));
  const m = u2Mesh(labelsForK(), loadingsForK(), UI2.sel,
                   hit => u2Hue(pos.has(hit[0]) ? pos.get(hit[0]) : 0, K),
                   state.minLoad, RUN.cohort_id);
  if (m) { elecMesh = m; nv.addMesh(m); }
  nv.drawScene();
};

// the strip and the matrix follow every re-cut and every run change, and this is the one
// call both paths already make
const u2RefreshMapList = refreshMapList;
refreshMapList = function () { u2RefreshMapList(); if (UI2.on) u2Sync(); };

// ── another run's labels at a K, for the cross-run grids ─────────────────────
// The published cut is already in memory (CONTACTS.runs / CONTACTS.loadings); any other
// K comes from that run's own sweep, fetched once and kept. No coverage, no neighbour
// index: a glass panel is contacts only.
const u2RD = new Map();
async function u2Data(runId, k) {
  if (RUN0 && runId === RUN0.id && Number(k) === Number(KSEL))
    return { lab: labelsForK(), load: loadingsForK(), clusters: (RUN.clusters || []).slice(),
             sizes: RUN.cluster_sizes || {}, cohort: RUN.cohort_id };
  const key = runId + "|" + k;
  if (u2RD.has(key)) return u2RD.get(key);
  const r = ((M && M.runs) || []).find(x => x.id === runId);
  let out = null;
  try {
    if (r && Number(r.k) === Number(k)) {
      out = { lab: CONTACTS.runs[runId] || null,
              load: (CONTACTS.loadings || {})[runId] || null,
              clusters: figureOrder(runId, k, (r.clusters || []).slice()),
              sizes: r.cluster_sizes || {}, cohort: r.cohort_id };
    } else if (r && r.sweep) {
      const base = `${BUNDLE}${r.dir}/`;
      const [labels, stats, loadings] = await Promise.all([
        fetch(base + r.sweep.labels).then(x => x.json()),
        fetch(base + r.sweep.stats).then(x => x.json()),
        r.sweep.loadings ? fetch(base + r.sweep.loadings).then(x => x.json()).catch(() => null)
                         : Promise.resolve(null),
      ]);
      const st = stats[String(k)] || {};
      out = { lab: labels[String(k)] || null,
              load: (loadings || {})[String(k)] || null,
              clusters: figureOrder(runId, k, (st.clusters || []).slice()),
              sizes: st.cluster_sizes || {}, cohort: r.cohort_id };
    }
  } catch (e) { console.warn("compare grid: cannot read", runId, "at K =", k, e); out = null; }
  u2RD.set(key, out);
  return out;
}

// ── the panel ────────────────────────────────────────────────────────────────
function u2Build() {
  $("ui2").innerHTML = `
    <div class="u2g">
      <div class="u2t">Run <span class="u2sub">feature set × algorithm</span></div>
      <div id="u2runs" class="u2mx"></div>
      <div class="cohort" id="u2runnote">—</div>
    </div>
    <div class="u2g" id="u2kwrap"><div class="u2t">Clusters (K)</div></div>
    <div class="u2g">
      <div class="u2t">Cluster <span class="u2sub" id="u2selnote"></span></div>
      <div id="u2chips" class="u2chips"></div>
    </div>
    <div class="u2g">
      <div class="u2t">View</div>
      <div class="segs" id="u2mode">
        <button type="button" data-m="single" class="on">one</button>
        <button type="button" data-m="overlay">overlay</button>
        <button type="button" data-m="grid">grid</button>
      </div>
      <div class="segs" id="u2gridby" style="display:none;margin-top:4px">
        <button type="button" data-g="clusters" class="on">clusters</button>
        <button type="button" data-g="algorithms">algorithms</button>
        <button type="button" data-g="fsets">feature sets</button>
      </div>
      <div class="cohort" id="u2modenote">—</div>
    </div>
    <div class="u2g" id="u2loadwrap"><div class="u2t">Minimum loading</div></div>
    <div class="u2g" id="u2btns"></div>`;
  // MOVED, not rebuilt: these carry the handlers wireUI() bound to them, and the K
  // buttons carry their own display toggling.
  $("u2kwrap").appendChild($("kRow"));
  $("u2loadwrap").appendChild($("loadRow"));
  $("u2loadwrap").appendChild($("loadNote"));
  $("u2btns").appendChild($("pdfBtn"));
  // the surface / radius / ceiling / contrast / vertex-gate controls: hidden, kept
  document.querySelectorAll("#ctlBody .grp").forEach(g => { g.style.display = "none"; });
  const g = document.createElement("div");
  g.id = "u2grid";
  g.innerHTML = `<div id="u2gh"></div><div id="u2gg"></div>`;
  document.body.appendChild(g);
}

function u2Runs() {
  const box = $("u2runs");
  if (!box || !RUN0) return;
  const cur = u2Split(RUN0.id);
  let h = `<div class="u2hd"></div>`
        + U2_METHODS.map(m => `<div class="u2hd">${u2esc(m[1])}</div>`).join("");
  for (const [f, short] of U2_FSETS) {
    h += `<div class="u2rl${f === cur.fset ? " on" : ""}" `
       + `title="${u2esc(REP_LABEL[f] || f)}">${u2esc(short)}</div>`;
    for (const [m, mlab] of U2_METHODS) {
      const r = u2Run(m, f), on = (m === cur.method && f === cur.fset);
      h += r
        ? `<button type="button" class="u2c${on ? " on" : ""}" data-run="${u2esc(r.id)}" `
          + `title="${u2esc(mlab + " · " + (REP_LABEL[f] || f) + " · " + r.run)}">`
          + `${on ? "✓" : ""}</button>`
        : `<div class="u2c off" title="not published"></div>`;
    }
  }
  box.innerHTML = h;
  box.querySelectorAll("button[data-run]").forEach(b => { b.onclick = () => u2GoRun(b.dataset.run); });
  $("u2runnote").innerHTML = `<b>${u2esc(U2_MLABEL[cur.method] || cur.method)}</b> · `
    + `${u2esc(REP_LABEL[cur.fset] || cur.fset)} — switching keeps K and the position, so `
    + `you land on the matched cluster`;
}

function u2Chips() {
  const box = $("u2chips");
  if (!box || !RUN) return;
  const defs = u2Defs(), K = defs.length;
  box.innerHTML = defs.map((d, i) => {
    const on = UI2.mode === "overlay" ? UI2.sel.has(d.cluster) : (state.map === d.id);
    return `<button type="button" class="u2ch${on ? " on" : ""}" data-map="${u2esc(d.id)}" `
         + `data-c="${d.cluster}" style="--c:${u2CSS(u2Hue(i, K))}" `
         + `title="position #${i + 1} in figure order · cluster ${d.cluster} · n=${u2Size(d.cluster)}">`
         + `<span class="sw"></span><b>#${i + 1}</b><i>${u2Size(d.cluster)}</i></button>`;
  }).join("");
  box.querySelectorAll("button[data-map]").forEach(b => {
    b.onclick = () => u2ClickChip(b.dataset.map, Number(b.dataset.c));
  });
  const pos = u2Pos(), d = defs[pos];
  $("u2selnote").textContent = UI2.mode === "overlay"
    ? `${UI2.sel.size} selected`
    : (d ? `#${pos + 1} · cluster ${d.cluster} · n=${u2Size(d.cluster)}` : "");
}

function u2Note() {
  const n = $("u2modenote");
  if (!n) return;
  n.innerHTML = UI2.mode === "single"
    ? "One cluster on the glass brain."
    : UI2.mode === "overlay"
      ? "Click chips to add or drop clusters. A contact in more than one selected cluster "
        + "is drawn <b>white and larger</b>."
      : (UI2.gridBy === "clusters"
          ? "Every cluster of this run at one camera angle. Click a panel to open it."
          : UI2.gridBy === "algorithms"
            ? "This position under each algorithm, same feature set and K."
            : "This position on each feature set, same algorithm and K.");
}

function u2Sync() {
  u2Runs(); u2Chips(); u2Note();
  $("u2gridby").style.display = UI2.mode === "grid" ? "" : "none";
  $("u2mode").querySelectorAll("button").forEach(b =>
    b.classList.toggle("on", b.dataset.m === UI2.mode));
  $("u2gridby").querySelectorAll("button").forEach(b =>
    b.classList.toggle("on", b.dataset.g === UI2.gridBy));
}

function u2Lock(on) { UI2.busy = on; $("ui2").classList.toggle("u2busy", on); }

// Selecting a cluster goes through the (hidden) map select, so everything wired to it —
// the render, the contacts, the centroid panel — happens exactly as it always did.
function u2Pick(mapId) {
  const sel = $("mapSel");
  if (!sel) return;
  state.map = mapId;
  sel.value = mapId;
  sel.dispatchEvent(new Event("change"));
  u2Sync();
}

function u2ClickChip(mapId, cid) {
  if (UI2.busy) return;
  if (UI2.mode === "overlay") {
    if (UI2.sel.has(cid)) { if (UI2.sel.size > 1) UI2.sel.delete(cid); }
    else UI2.sel.add(cid);
    state.map = mapId;
    const sel = $("mapSel"); if (sel) sel.value = mapId;
    updateElectrodes();
    if (typeof drawCentroid === "function") drawCentroid();
    u2Sync();
    return;
  }
  u2Pick(mapId);
  if (UI2.mode === "grid" && UI2.gridOpen) u2Grid();
}

async function u2GoRun(id) {
  if (!id || UI2.busy || (RUN0 && RUN0.id === id)) return;
  const pos = u2Pos();
  u2Lock(true);
  try {
    await selectRun(id, true);            // keepK: the same cut under the other algorithm
    const defs = u2Defs(), d = defs[Math.min(pos, defs.length - 1)];
    if (d) { UI2.sel = new Set([d.cluster]); u2Pick(d.id); }
  } catch (e) {
    console.error("run switch failed:", e);
  } finally {
    u2Lock(false);
    u2Sync();
    if (UI2.mode === "grid" && UI2.gridOpen) u2Grid();
  }
}

function u2SetMode(m) {
  UI2.mode = m;
  if (m === "overlay" && !UI2.sel.size) {
    const d = u2Defs()[u2Pos()];
    if (d) UI2.sel.add(d.cluster);
  }
  if (m === "grid") { u2Sync(); u2Grid(); }
  else { u2CloseGrid(); updateElectrodes(); u2Sync(); }
}

// ── the grid of small multiples ──────────────────────────────────────────────
function u2Panels() {
  const defs = u2Defs(), K = defs.length, pos = u2Pos(), cur = u2Split(RUN0.id);
  if (UI2.gridBy === "clusters")
    return defs.map((d, i) => ({ runId: RUN0.id, k: KSEL, cluster: d.cluster, pos: i, K,
                                 title: `#${i + 1}`, sub: `cluster ${d.cluster}` }));
  if (UI2.gridBy === "algorithms")
    return U2_METHODS.map(([m, lab]) => {
      const r = u2Run(m, cur.fset);
      return r ? { runId: r.id, k: KSEL, pos, K, title: lab,
                   sub: REP_LABEL[cur.fset] || cur.fset } : null;
    }).filter(Boolean);
  return U2_FSETS.map(([f]) => {
    const r = u2Run(cur.method, f);
    return r ? { runId: r.id, k: KSEL, pos, K, title: REP_LABEL[f] || f,
                 sub: U2_MLABEL[cur.method] || cur.method } : null;
  }).filter(Boolean);
}

// ONE crop box for the whole grid, measured with every contact of the run drawn, the way
// the report measures its own. Without it each panel is trimmed to its own dots, so a
// tight cluster fills its cell and a spread one shrinks — the panels would stop being
// comparable, which is the entire point of a grid.
async function u2Box() {
  if (elecMesh) { try { nv.removeMesh(elecMesh); } catch (e) {} elecMesh = null; }
  const m = u2Mesh(labelsForK(), loadingsForK(), new Set(RUN.clusters || []),
                   () => [255, 232, 64], 0, RUN.cohort_id);
  if (m) { elecMesh = m; nv.addMesh(m); }
  const b = await new Promise(res => {
    requestAnimationFrame(() => {
      try { nv.drawScene(); } catch (e) {}
      const box = inkBox(canvasEl, true), pad = 10;
      res(box ? { x: Math.max(0, box.x - pad), y: Math.max(0, box.y - pad),
                  w: Math.min(canvasEl.width, box.w + 2 * pad),
                  h: Math.min(canvasEl.height, box.h + 2 * pad) } : null);
    });
  });
  if (elecMesh) { try { nv.removeMesh(elecMesh); } catch (e) {} }
  elecMesh = null;
  return b;
}

async function u2Snap(p, box, w) {
  const D = await u2Data(p.runId, p.k);
  if (!D || !D.lab) return null;
  const cid = (p.cluster !== undefined) ? p.cluster : (D.clusters || [])[p.pos];
  if (cid === undefined) return null;
  const az = nv.scene.renderAzimuth, el = nv.scene.renderElevation;
  if (elecMesh) { try { nv.removeMesh(elecMesh); } catch (e) {} elecMesh = null; }
  const m = u2Mesh(D.lab, D.load, new Set([cid]), () => u2Hue(p.pos, p.K),
                   state.minLoad, D.cohort);
  if (m) { elecMesh = m; nv.addMesh(m); }
  const im = box ? await captureRaw(az, el, w, box, true)
                 : await captureView(az, el, w, true);
  if (elecMesh) { try { nv.removeMesh(elecMesh); } catch (e) {} }
  elecMesh = null;
  return { im, cid, n: (D.sizes || {})[String(cid)] };
}

function u2Head(txt, sub, prog) {
  $("u2gh").innerHTML = `<div class="t">${u2esc(txt)}</div><div class="s">${u2esc(sub)}</div>
    <div class="sp"></div>
    <button type="button" data-v="left">left</button>
    <button type="button" data-v="right">right</button>
    <button type="button" data-v="top">top</button>
    <button type="button" data-v="front">front</button>
    <button type="button" data-v="live">live angle</button>
    <button type="button" data-v="close">close ✕</button>
    <div class="s" id="u2prog">${u2esc(prog || "")}</div>`;
  $("u2gh").querySelectorAll("button[data-v]").forEach(b => {
    b.onclick = () => {
      const v = b.dataset.v;
      if (v === "close") return u2SetMode("single");
      const A = { left:[90, 0], right:[270, 0], top:[0, 90], front:[180, 0] }[v];
      if (A) { nv.scene.renderAzimuth = A[0]; nv.scene.renderElevation = A[1]; }
      u2Grid();
    };
  });
}

function u2CloseGrid() {
  UI2.gridOpen = false;
  const g = $("u2grid"); if (g) g.classList.remove("on");
  document.body.classList.remove("u2gridon");
}

async function u2Grid() {
  if (UI2.busy || !RUN) return;
  const seq = ++UI2.seq;
  UI2.gridOpen = true;
  $("u2grid").classList.add("on");
  document.body.classList.add("u2gridon");
  const panels = u2Panels(), cur = u2Split(RUN0.id);
  const what = UI2.gridBy === "clusters"
    ? `${U2_MLABEL[cur.method] || cur.method} · ${REP_LABEL[cur.fset] || cur.fset} · K=${KSEL}`
    : UI2.gridBy === "algorithms"
      ? `position #${u2Pos() + 1} · ${REP_LABEL[cur.fset] || cur.fset} · K=${KSEL}`
      : `position #${u2Pos() + 1} · ${U2_MLABEL[cur.method] || cur.method} · K=${KSEL}`;
  u2Head(UI2.gridBy === "clusters" ? "Every cluster, one angle"
       : UI2.gridBy === "algorithms" ? "The same position, three algorithms"
       : "The same position, four feature sets", what, "capturing…");
  const gg = $("u2gg");
  gg.innerHTML = panels.map((p, i) =>
    `<figure data-i="${i}" style="--c:${u2CSS(u2Hue(p.pos, p.K))}">
       <div class="miss">…</div>
       <div class="cap"><span class="sw"></span><b>${u2esc(p.title)}</b>
         <span class="lab">${u2esc(p.sub)}</span></div>
     </figure>`).join("");
  u2Lock(true);
  const W = Math.max(220, Math.min(520, Math.round(canvasEl.width / 3)));
  try {
    const box = await u2Box();
    for (let i = 0; i < panels.length; i++) {
      if (seq !== UI2.seq) return;                       // superseded by a newer grid
      const r = await u2Snap(panels[i], box, W);
      const fig = gg.querySelector(`figure[data-i="${i}"]`);
      if (!fig) continue;
      const slot = fig.querySelector(".miss, img");
      if (r && r.im && r.im.data) {
        const im = document.createElement("img");
        im.src = r.im.data;
        im.alt = `${panels[i].title} — ${panels[i].sub}`;
        slot.replaceWith(im);
        fig.dataset.cluster = r.cid;
        fig.dataset.run = panels[i].runId;
        if (r.n !== undefined)
          fig.querySelector(".cap .lab").textContent = `${panels[i].sub} · n=${r.n}`;
      } else {
        slot.textContent = "not published at this K";
      }
      const pr = $("u2prog"); if (pr) pr.textContent = `${i + 1} / ${panels.length}`;
    }
    const pr = $("u2prog");
    if (pr) pr.textContent = `${panels.length} panels · click one to open it`;
    const here = (u2Defs()[u2Pos()] || {}).cluster;
    gg.querySelectorAll("figure").forEach(f => {
      f.classList.toggle("on", f.dataset.run === RUN0.id
                               && Number(f.dataset.cluster) === Number(here));
      f.onclick = async () => {
        const runId = f.dataset.run, cid = Number(f.dataset.cluster);
        if (!runId) return;
        UI2.mode = "single"; u2CloseGrid();
        if (runId !== RUN0.id) await u2GoRun(runId);
        else {
          const d = u2Defs().find(x => Number(x.cluster) === cid);
          if (d) { UI2.sel = new Set([cid]); u2Pick(d.id); }
        }
        updateElectrodes(); u2Sync();
      };
    });
  } catch (e) {
    console.error("compare grid failed:", e);
    const pr = $("u2prog"); if (pr) pr.textContent = "failed — see the console";
  } finally {
    u2Lock(false);
    updateElectrodes();                 // the live view comes back exactly as it was
  }
}

// ── keys ─────────────────────────────────────────────────────────────────────
function u2Keys(e) {
  if (!UI2.on || UI2.busy) return;
  if (e.target && (e.target.tagName === "SELECT" || e.target.tagName === "INPUT")) return;
  const defs = u2Defs(), pos = u2Pos(), cur = u2Split(RUN0.id);
  const step = (arr, val, d) => {
    const i = arr.findIndex(x => x[0] === val);
    return arr[(Math.max(0, i) + d + arr.length) % arr.length][0];
  };
  if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
    const d = defs[(pos + (e.key === "ArrowRight" ? 1 : defs.length - 1)) % defs.length];
    if (d) {
      UI2.sel = new Set([d.cluster]);
      u2Pick(d.id);
      if (UI2.mode === "grid" && UI2.gridOpen) u2Grid();
    }
  } else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    const r = u2Run(cur.method, step(U2_FSETS, cur.fset, e.key === "ArrowDown" ? 1 : -1));
    if (r) u2GoRun(r.id);
  } else if (e.key === "1" || e.key === "2" || e.key === "3") {
    const m = U2_METHODS[Number(e.key) - 1], r = m && u2Run(m[0], cur.fset);
    if (r) u2GoRun(r.id);
  } else if (e.key === "g") {
    u2SetMode(UI2.mode === "grid" ? "single" : "grid");
  } else if (e.key === "o") {
    u2SetMode(UI2.mode === "overlay" ? "single" : "overlay");
  } else { return; }
  e.preventDefault();
}

// ── init, after boot has a run on screen ─────────────────────────────────────
async function ui2Init() {
  u2Build();
  // glass is the only surface here; entering it through the page's own handler keeps the
  // mesh reload, the white backdrop and the by-cluster contacts in one place
  try { await $("sGlass").onclick(); } catch (e) { console.warn("glass mode:", e); }
  UI2.on = true;
  const d0 = u2Defs()[u2Pos()];
  if (d0) UI2.sel = new Set([d0.cluster]);
  $("u2mode").querySelectorAll("button").forEach(b => {
    b.onclick = () => { if (!UI2.busy) u2SetMode(b.dataset.m); };
  });
  $("u2gridby").querySelectorAll("button").forEach(b => {
    b.onclick = () => { if (UI2.busy) return; UI2.gridBy = b.dataset.g; u2Sync(); u2Grid(); };
  });
  // the report is built one cluster at a time, never from a grid or an overlay
  $("pdfBtn").onclick = async () => { u2SetMode("single"); await exportReport(); u2Sync(); };
  window.addEventListener("keydown", u2Keys);
  u2Sync();
}
'''


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
        "  #centroidPanel canvas { display:block; width:380px; height:170px; border-radius:6px; background:#ffffff; }\n"
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
    // reachable from the console, so the report's image path can be checked without
    // running a whole export: cvDebug.centroidImage(8, 3, "concat_hg", 0.4, 640, 300)
    window.cvDebug = { centroidImage: (...a) => centroidImage(...a), cenMean: (...a) => cenMean(...a),
                       cenX: (...a) => cenX(...a), reps: KROW_REPS };
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
// the representations a report's K rows show, in this order
const KROW_REPS = ["concat_hg", "concat_bands5z", "concat_rawds"];
function cenGeom(rep) {
  const r = rep || CEN_REP, g = CEN_INDEX.feature_sets[r];
  if (!g) return null;
  return { rep: r, n: g.n, nc: g.conds.length, nb: g.bands.length, nt: g.nt, conds: g.conds,
           F: g.conds.length * g.bands.length * g.nt,
           unit: g.unit || "dB", unitLong: g.unit_long || "" };
}
// weights per electrode for cluster j at K: the loading (graded) or membership (hard).
// `min` is the minimum loading applied - the slider's, or the copy a report froze.
function cenWeights(k, j, min, paper) {
  const m = CEN.meta, n = m.n, w = new Float32Array(n), lab = m.labels[String(k)];
  let nMembers = 0, nPass = 0;
  if (paper || !CEN.W || m.w_offsets[String(k)] === undefined) {
    if (!lab) return null;
    for (let i = 0; i < n; i++) if (lab[i] === j) { w[i] = 1; nMembers++; nPass++; }
    return { w, nMembers, nPass, weighted: false };
  }
  const off = m.w_offsets[String(k)];
  for (let i = 0; i < n; i++) {
    const v = CEN.W[off + i * k + j] * m.w_scale;
    if (lab && lab[i] === j) nMembers++;
    if (v > 0 && v >= (min || 0)) { w[i] = v; nPass++; }
  }
  return { w, nMembers, nPass, weighted: true };
}
// the weighted mean of representation `rep` over the electrodes cenWeights picks; the
// weighted SD too, but only for the single-band line, where it is drawn
function cenMean(k, j, min, paper, rep) {
  const g = cenGeom(rep); if (!g) return null;
  const X = _cenX.get(g.rep);
  if (!X || X.length !== g.n * g.F || g.n !== CEN.meta.n) return null;
  const r = cenWeights(k, j, min, paper); if (!r) return null;
  const mean = new Float32Array(g.F), sd = g.nb === 1 ? new Float32Array(g.F) : null;
  let sw = 0; for (let i = 0; i < g.n; i++) sw += r.w[i];
  if (sw <= 0) return Object.assign(r, { mean, sd, empty: true, g });
  const s = CEN_INDEX.x_scale;
  for (let i = 0; i < g.n; i++) { const wi = r.w[i]; if (!wi) continue; const b = i * g.F; for (let f = 0; f < g.F; f++) mean[f] += wi * X[b + f]; }
  for (let f = 0; f < g.F; f++) mean[f] = mean[f] * s / sw;
  if (sd) {
    for (let i = 0; i < g.n; i++) { const wi = r.w[i]; if (!wi) continue; const b = i * g.F; for (let f = 0; f < g.F; f++) { const d = X[b + f] * s - mean[f]; sd[f] += wi * d * d; } }
    for (let f = 0; f < g.F; f++) sd[f] = Math.sqrt(sd[f] / sw);
  }
  return Object.assign(r, { mean, sd, empty: false, g });
}
// GLOBAL scales, the same for every cluster of a run at this K on this representation,
// so clusters are comparable. Heatmap: symmetric colour limit at the 99th percentile
// of |argmax mean| over all clusters (FIG 1's rule). Line: the range of every cluster's
// mean +/- SD (FIG 1's rule), so the band never clips.
function cenLimits(k, rep) {
  const g = cenGeom(rep); if (!g) return { vlim: 1, ylo: -1, yhi: 1 };
  const key = CEN.meta.id + ":" + g.rep + ":" + k; if (_cenLim.has(key)) return _cenLim.get(key);
  const vals = []; let lo = 0, hi = 0;
  if (CEN.meta.labels[String(k)]) for (let j = 0; j < k; j++) {
    const st = cenMean(k, j, 0, true, g.rep); if (!st || st.empty) continue;
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
// DRAWS cluster j at K on representation `rep` into any 2-D context, at any size:
// the panel calls it on its canvas, the report on offscreen canvases. `min` is the
// minimum loading applied, `paper` selects FIG 1's argmax-member mean, `scale` sizes
// text and line widths (1 at the panel's 380 px). Returns the stats drawn, or null.
function renderCentroid(ctx, Wd, Hd, k, j, rep, min, paper, scale) {
  const st = cenMean(k, j, min, paper, rep); if (!st) return null;
  const g = st.g, lim = cenLimits(k, g.rep), nc = g.nc, nb = g.nb, nt = g.nt, ncol = nc * nt, col = clusterCSS(j);
  const S = scale || 1, font = px => `${Math.round(px * S)}px sans-serif`;
  // everything on white, as the figures are: a dark ground around a heatmap read as a
  // black frame on the report's white page. The heatmap gets a thin grey frame instead,
  // so its edge still shows where RdBu's white zero meets the page.
  const line = nb === 1;
  const T = { bg: "#ffffff", fg: "#1b232c", muted: "#68727d", grid: "#c9ced4", sep: "#1b232c", cue: "#68727d" };
  ctx.setLineDash([]); ctx.fillStyle = T.bg; ctx.fillRect(0, 0, Wd, Hd);
  // a right gutter carries the GLOBAL scale: a colour bar for the heatmap, a y axis for
  // the line, both labelled with the unit - the same for every cluster at this K
  const padL = (line ? 40 : 6) * S, padR = (line ? 8 : 46) * S, padT = 8 * S, padB = 16 * S, w = Wd - padL - padR, h = Hd - padT - padB;
  const unit = g.unit;
  ctx.font = font(9); ctx.fillStyle = T.muted;
  if (st.empty) { ctx.font = font(11); ctx.textAlign = "left"; ctx.fillText("no electrode passes the minimum", padL + 8 * S, Hd / 2); return st; }
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
    ctx.strokeStyle = T.grid; ctx.lineWidth = 1; ctx.strokeRect(padL + 0.5, padT + 0.5, w - 1, h - 1);   // the plot's edge, on white
    // colour bar: -vlim (blue) at the bottom to +vlim (red) at the top
    const bx = Wd - padR + 8 * S, bw = 9 * S;
    for (let y = 0; y < h; y++) { const rgb = rdbu_r(1 - y / (h - 1)); ctx.fillStyle = `rgb(${rgb[0]|0},${rgb[1]|0},${rgb[2]|0})`; ctx.fillRect(bx, padT + y, bw, 1); }
    ctx.strokeStyle = T.grid; ctx.lineWidth = 1; ctx.strokeRect(bx + 0.5, padT + 0.5, bw - 1, h - 1);
    ctx.fillStyle = T.muted; ctx.textAlign = "left";
    ctx.fillText(`+${lim.vlim.toFixed(1)}`, bx + bw + 3 * S, padT + 8 * S);
    ctx.fillText("0", bx + bw + 3 * S, padT + h / 2 + 3 * S);
    ctx.fillText(`−${lim.vlim.toFixed(1)}`, bx + bw + 3 * S, padT + h - 1);
    ctx.save(); ctx.translate(Wd - 3 * S, padT + h / 2); ctx.rotate(Math.PI / 2); ctx.textAlign = "center"; ctx.fillText(unit, 0, 0); ctx.restore();
  } else {
    const lo = lim.ylo, hi = lim.yhi, sy = v => padT + h * (1 - (v - lo) / (hi - lo));
    // y axis with the global range and the unit
    ctx.strokeStyle = T.grid; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(padL - 0.5, padT); ctx.lineTo(padL - 0.5, padT + h); ctx.stroke();
    ctx.fillStyle = T.muted; ctx.textAlign = "right";
    for (const v of [hi, 0, lo]) { const y = sy(v); ctx.beginPath(); ctx.moveTo(padL - 4 * S, y + 0.5); ctx.lineTo(padL - 0.5, y + 0.5); ctx.stroke(); ctx.fillText(v.toFixed(1), padL - 6 * S, y + 3 * S); }
    ctx.save(); ctx.translate(9 * S, padT + h / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.fillText(unit, 0, 0); ctx.restore();
    ctx.strokeStyle = T.grid; ctx.beginPath(); ctx.moveTo(padL, sy(0) + 0.5); ctx.lineTo(padL + w, sy(0) + 0.5); ctx.stroke();
    for (let c = 0; c < nc; c++) {
      const x0 = padL + w * c / nc, xw = w / nc, xt = t => x0 + xw * t / (nt - 1);
      if (st.sd) {                                                // +/-1 SD, weighted - the line only
        ctx.fillStyle = col; ctx.globalAlpha = 0.22; ctx.beginPath();
        for (let t = 0; t < nt; t++) { const f = c * nt + t; if (!t) ctx.moveTo(xt(t), sy(st.mean[f] + st.sd[f])); else ctx.lineTo(xt(t), sy(st.mean[f] + st.sd[f])); }
        for (let t = nt - 1; t >= 0; t--) { const f = c * nt + t; ctx.lineTo(xt(t), sy(st.mean[f] - st.sd[f])); }
        ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
      }
      ctx.strokeStyle = col; ctx.lineWidth = 1.4 * S; ctx.beginPath();
      for (let t = 0; t < nt; t++) { const f = c * nt + t; if (!t) ctx.moveTo(xt(t), sy(st.mean[f])); else ctx.lineTo(xt(t), sy(st.mean[f])); }
      ctx.stroke();
    }
  }
  for (let c = 0; c < nc; c++) {                                  // blocks and the GO cue at 50%
    const x0 = padL + w * c / nc, xm = x0 + w / nc / 2;
    if (c) { ctx.strokeStyle = T.sep; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0 + 0.5, padT); ctx.lineTo(x0 + 0.5, padT + h); ctx.stroke(); }
    ctx.strokeStyle = T.cue; ctx.setLineDash([4 * S, 3 * S]); ctx.beginPath(); ctx.moveTo(xm + 0.5, padT); ctx.lineTo(xm + 0.5, padT + h); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = T.muted; ctx.font = font(10); ctx.textAlign = "center"; ctx.fillText(g.conds[c], xm, Hd - 4 * S);
  }
  return st;
}
// a centroid as an image the report can embed - {data, w, h}, the shape tryImage
// returns, plus the counts the caption needs. Always the weighted definition.
function centroidImage(k, j, rep, min, w, h) {
  if (!CEN || !CEN_INDEX) return { data: "", w: 0, h: 0 };
  const c = document.createElement("canvas"); c.width = w || 760; c.height = h || 340;
  const st = renderCentroid(c.getContext("2d"), c.width, c.height, k, j, rep, min, false, c.width / 380);
  if (!st) return { data: "", w: 0, h: 0 };
  return { data: c.toDataURL("image/png"), w: c.width, h: c.height,
           n: st.nPass, members: st.nMembers, weighted: st.weighted, rep };
}
// the K rows: one cell per representation in KROW_REPS, or the single published chip
// when the live centroids were not available
function centroidCells(three, fallback) {
  if (three && three.some(x => x && x.data))
    return three.map((im, q) => cellImg(im, REP_LABEL[KROW_REPS[q]] || KROW_REPS[q])).join("");
  return cellImg(fallback, "centroid") + "<div></div><div></div>";
}
function drawCentroid() {
  const P = $("centroidPanel"); if (!P) return;
  const d = curDef();
  if (!CEN || !CEN_INDEX || !d || d.cluster === undefined) { P.classList.add("hidden"); return; }
  const m = CEN.meta, k = Number(KSEL), j = Number(d.cluster);
  if (!m.ks.includes(k)) { P.classList.add("hidden"); return; }
  const paper = !!$("cpPaper").checked, cv = $("cpCanvas");
  const st = renderCentroid(cv.getContext("2d"), cv.width, cv.height, k, j, CEN_REP, state.minLoad, paper, 1);
  if (!st) { P.classList.add("hidden"); return; }
  P.classList.remove("hidden");
  const g = st.g, own = m.feature_set, cross = CEN_REP !== own;
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

    # ---- the report uses the LIVE centroids, at the minimum loading frozen at save ----
    # Every centroid in the saved report is the loading-weighted mean over electrodes at
    # or above the min-loading slider as it stood when Save was pressed (HFA with its
    # +/-1 SD band), drawn by the same renderer as the panel. The per-cluster card keeps
    # ONE centroid - the run's own representation. The K rows carry three: HFA, 5 bands
    # z-scored, 15 bands (KROW_REPS). The published PNGs are used only if the bundle is
    # not loaded, so a report can never come out empty.
    sub("    surf: state.surf, map: state.map, glass: state.glass,\n  };",
        "    surf: state.surf, map: state.map, glass: state.glass,\n"
        "    minLoad: state.minLoad,             // every centroid below is computed at this\n"
        "  };", "frozen minLoad")
    sub('    <span class="chip">gamma <b>${frozen.gamma.toFixed(2)}</b></span>',
        '    <span class="chip">gamma <b>${frozen.gamma.toFixed(2)}</b></span>\n'
        '    <span class="chip">min loading <b>${frozen.minLoad.toFixed(2)}</b></span>',
        "header chip: min loading")
    sub("  radius ${frozen.radius} mm · gate n ≥ ${frozen.minN} · ${esc(frozen.surf)} surface ·",
        "  radius ${frozen.radius} mm · gate n ≥ ${frozen.minN} · min loading "
        "${frozen.minLoad.toFixed(2)} (centroids: loading-weighted mean over electrodes at or "
        "above it; HFA with ±1 SD) · ${esc(frozen.surf)} surface ·", "footer: min loading")
    sub(r"""      const centroid = kDir
        ? await tryImage(`${runBase}cluster_centroids/${kDir}cluster_${cc}.png`, FIGW, false)
        : (fig.centroid ? await tryImage(runBase + fig.centroid, FIGW, false)
                        : { data: "", w: 0, h: 0 });""",
        r"""      // THE LIVE CENTROID at the minimum loading frozen when Save was pressed, on the
      // run's own representation; the published PNG only if the bundle is not loaded
      const live = (CEN && CEN_INDEX)
        ? centroidImage(KSEL, Number(d.cluster), CEN.meta.feature_set, frozen.minLoad, FIGW, Math.round(FIGW * 0.45))
        : null;
      const centroid = (live && live.data) ? live : (kDir
        ? await tryImage(`${runBase}cluster_centroids/${kDir}cluster_${cc}.png`, FIGW, false)
        : (fig.centroid ? await tryImage(runBase + fig.centroid, FIGW, false)
                        : { data: "", w: 0, h: 0 }));
      // the same cluster on the three representations the K rows show
      const centroids3 = (CEN && CEN_INDEX)
        ? KROW_REPS.map(rep => centroidImage(KSEL, Number(d.cluster), rep, frozen.minLoad, 640, 300))
        : null;""", "card centroid: live")
    sub("      blocks.push({ d, st, shots, centroid, raster, spinFrames,",
        "      blocks.push({ d, st, shots, centroid, centroids3, raster, spinFrames,",
        "card block carries centroids3")
    sub(r"""      <div>${img(centroid, "cluster centroid — the mean response this cluster is defined by", "wide lite")}</div>""",
        r"""      <div>${img(centroid, centroid.members !== undefined
        ? `cluster centroid — loading-weighted mean of ${centroid.n} electrodes with loading ≥ ${frozen.minLoad.toFixed(2)} (of ${centroid.members} members), on ${REP_LABEL[centroid.rep] || centroid.rep}${centroid.rep === "concat_hg" ? ", band = ±1 SD" : ""}`
        : "cluster centroid — the mean response this cluster is defined by", "wide lite")}${
        (centroids3 || []).filter(im => im && im.data && im.rep !== centroid.rep).length
          ? `<div class="cluesub">${(centroids3 || []).filter(im => im && im.data && im.rep !== centroid.rep).slice(0, 2)
              .map(im => img(im, `the same cluster on ${REP_LABEL[im.rep] || im.rep}`, "lite")).join("")}</div>`
          : ""}</div>""",
        "card caption")
    sub("    const clusterHTML = blocks.map(({ d, st, shots, glassShots, centroid,\n"
        "                                 raster, spinFrames, glassSpin, climMax }, i) => {",
        "    const clusterHTML = blocks.map(({ d, st, shots, glassShots, centroid, centroids3,\n"
        "                                 raster, spinFrames, glassSpin, climMax }, i) => {",
        "card: destructure centroids3")
    # the two other representations, small, right under the main centroid
    sub(".cluepair{display:grid;grid-template-columns:1fr 1fr;gap:14px;align-items:start;margin-top:16px}",
        ".cluepair{display:grid;grid-template-columns:1fr 1fr;gap:14px;align-items:start;margin-top:16px}\n"
        ".cluesub{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:8px}\n"
        ".cluesub figure{margin:0} .cluesub figcaption{font-size:10.5px}",
        "card: sub-centroid css")
    sub(r"""      for (const k of KP.ks) {
        btn.textContent = `Report: K panel ${k}…`;""",
        r"""      if (CEN && CEN_INDEX) {                     // the three matrices the K rows draw on
        btn.textContent = "Report: centroid data…";
        try { await Promise.all(KROW_REPS.map(cenX)); } catch (e) { console.warn("centroid matrices", e); }
      }
      for (const k of KP.ks) {
        btn.textContent = `Report: K panel ${k}…`;""", "K rows: preload the matrices")
    sub(r"""        const chips = await Promise.all(ids.map(c =>
          tryImage(`${runBase}cluster_centroids/k_${k}/`
                   + `cluster_${String(c).padStart(2, "0")}.png`, 320, false)));""",
        r"""        // LIVE centroids on the three representations, at the frozen minimum loading;
        // the published chip only if the bundle is not loaded
        const chips = (CEN && CEN_INDEX)
          ? ids.map(c => KROW_REPS.map(rep => centroidImage(k, Number(c), rep, frozen.minLoad, 640, 300)))
          : await Promise.all(ids.map(c =>
              tryImage(`${runBase}cluster_centroids/k_${k}/`
                       + `cluster_${String(c).padStart(2, "0")}.png`, 320, false)));""",
        "K rows: chips")
    sub("    const cardsRows = blocks.map(({ d, shots, glassShots, centroid }, i) => {",
        "    const cardsRows = blocks.map(({ d, shots, glassShots, centroid, centroids3 }, i) => {",
        "current-K row: destructure")
    sub(r"""        ${cellImg(centroid, "centroid")}
        ${cells}""",
        r"""        ${centroidCells(centroids3, centroid)}
        ${cells}""", "current-K row: cells")
    sub(r"""            ${cellImg(B.chips[i], "centroid")}""",
        r"""            ${centroidCells(Array.isArray(B.chips[i]) ? B.chips[i] : null, B.chips[i])}""",
        "other-K row: cells")
    sub(r"""        <div class="cardhead"><div></div><div>centroid</div>""",
        r"""        <div class="cardhead"><div></div>${KROW_REPS.map(r => `<div>${REP_LABEL[r] || r}</div>`).join("")}""",
        "K rows: header")
    sub(".cardhead,.cardrow,.cardaxis{display:grid;grid-template-columns:30px .95fr 1.6fr 1.6fr 1.05fr 1.15fr;gap:9px;align-items:center}",
        ".cardhead,.cardrow,.cardaxis{display:grid;grid-template-columns:30px 1fr 1fr 1fr 1.25fr 1.25fr .85fr 1.05fr;gap:8px;align-items:center}",
        "K rows: eight columns")
    sub(r"""        <div class="cardaxis"><div></div><div></div><div></div><div></div><div></div>""",
        r"""        <div class="cardaxis"><div></div><div></div><div></div><div></div><div></div><div></div><div></div>""",
        "K rows: axis row")

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

    # ---- the compare UI ---------------------------------------------------------------
    # The two dropdowns hid what there was to compare. The runs become a feature-set x
    # algorithm matrix and the clusters a strip, both permanent; overlay and grid views
    # do the comparing. Glass is the only surface, so the controls that only ever drove
    # the vertex map are hidden - see UI2_JS for why they were dead.
    sub("  option { color:var(--fg); }", "  option { color:var(--fg); }\n" + UI2_CSS.rstrip("\n"),
        "compare UI: css")
    sub('    <div id="ctlBody">', '    <div id="ctlBody">\n      <div id="ui2"></div>',
        "compare UI: container")
    sub("  <div id=\"hint\">drag rotate · scroll zoom · <b>h</b> panels · <b>c</b> controls</div>",
        "  <div id=\"hint\">drag rotate · scroll zoom · <b>←&nbsp;→</b> cluster · "
        "<b>↑&nbsp;↓</b> feature set · <b>1 2 3</b> algorithm · <b>g</b> grid · "
        "<b>o</b> overlay · <b>h</b> panels · <b>c</b> controls</div>", "compare UI: hint")
    sub("  wireUI();\n  await selectRun(pick.id);\n",
        "  wireUI();\n  await selectRun(pick.id);\n  await ui2Init();\n", "compare UI: boot")
    sub("boot().catch(e => {", UI2_JS + "\nboot().catch(e => {", "compare UI: module")

    # ---- the report's brain captures ---------------------------------------------------
    # REPORT_3D=False removes every render the viewer captures - the three stills and the
    # 360 turn per cluster, the whole-cohort turns - and the sections built from them.
    # The centroids, rasters, composition bars, K panel and published figures are
    # untouched, so a report is still one file per run and takes seconds rather than
    # minutes. True restores all of it; nothing else in this file changes with the flag.
    sub("const PAPER_K = " + str(PAPER_K) + ";\n",
        "const PAPER_K = " + str(PAPER_K) + ";\n"
        "// the viewer's brain captures in the saved report; REPORT_3D in "
        "make_cluster_visualizer.py\n"
        "const REPORT_3D = " + ("true" if REPORT_3D else "false") + ";\n", "report: the flag")
    sub("      const st = render(); await sleep(80);\n"
        "      const shots = [];\n"
        "      for (const [az, el] of VIEWS) shots.push(await captureView(az, el, CAPW, true));\n"
        "      const spinFrames = await captureSpin(0, SPIN_N, SPIN_W, true);\n",
        "      const st = render();          // the stats are read either way\n"
        "      const shots = [];\n"
        "      let spinFrames = [];\n"
        "      if (REPORT_3D) {\n"
        "        await sleep(80);\n"
        "        for (const [az, el] of VIEWS) shots.push(await captureView(az, el, CAPW, true));\n"
        "        spinFrames = await captureSpin(0, SPIN_N, SPIN_W, true);\n"
        "      }\n", "report: per-cluster captures")

    # the glass pass - stills, turns and every cluster at every K - skipped whole, so the
    # surface is never swapped and nothing needs restoring
    i = s.index("    const glassWas = { glass: state.glass, byCluster: state.elecByCluster,")
    j = s.index("    // ── whole-run figures", i)
    s = s[:i] + "    if (REPORT_3D) {\n" + s[i:j].rstrip() + "\n    }\n\n" + s[j:]
    n_patch += 1

    # the per-cluster probability stills, the glass stills and the two captions
    i = s.index('  <p class="sechint">P(cluster ${k} | labelled electrode within')
    j = s.index('  <div style="margin-top:16px">\n    <div class="${raster', i)
    s = s[:i] + '  ${!REPORT_3D ? "" : `' + s[i:j].rstrip() + '`}\n' + s[j:]
    n_patch += 1

    # ... the per-cluster pair of turns ...
    a = '    <div style="margin-top:14px">\n      <h2 class="sec">Seen from every angle</h2>'
    b = '${spin(glassSpin, `Cluster ${k} contacts`)}\n      </div>\n    </div>'
    i = s.index(a)
    j = s.index(b, i) + len(b)
    s = s[:i] + '    ${!REPORT_3D ? "" : `' + s[i:j].lstrip() + '`}\n' + s[j:]
    n_patch += 1

    # ... and the two whole-cohort turn sections
    a = '<section class="card">\n  <div class="three">\n'
    if s.count(a) != 1:
        raise SystemExit(f"anchor for the glass turn sections matches {s.count(a)} times")
    i = s.index(a)
    j = s.index("${resultsHTML}", i)
    s = s[:i] + '${!REPORT_3D ? "" : `' + s[i:j].rstrip() + '`}\n\n' + s[j:]
    n_patch += 1

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
