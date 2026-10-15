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

5. BRAIN CAPTURES IN THE REPORT (REPORT_3D: False / "stills" / True; "stills" since 2026-10-13 -
   the three probability stills and the three glass stills of every cluster at the K on
   screen, no turns, no every-K pass). With False: the three stills and the 360
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
# "stills" (2026-10-13, Lora: "include the brain screenshots the way it used to"): the three
# probability stills and the three glass stills of every cluster AT THE K ON SCREEN come back;
# the 360 turns, the two whole-cohort turns and the every-cluster-at-every-K pass stay out,
# so a report still takes seconds, not minutes, and stays small.
REPORT_3D = "stills"
# The four feature sets the page shows, in band order. A set missing from this list is
# invisible to the page however complete its runs are - the same trap as TRACKS in
# make_coverage_bundle.py, and it caught concat_bands15 / bands2 / hfa for three days.
# The pre-10-13 sets (concat_bands5z, concat_hg, concat_rawds) are deliberately out:
# they duplicated these four under near-identical names and only confused the matrix.
FSETS = ["concat_bands2", "concat_bands5", "concat_bands15", "concat_hfa"]
METHODS = ["cnmf", "kmeans", "hierarchical"]          # archetypal analysis dropped 2026-09-06
COHORT_FALLBACK = "cohort1_n29"   # only used if a manifest carries several cohorts


def cohort_id(manifest) -> str:
    """Which cohort the manifest is for, read from the manifest and not pinned here.

    make_coverage_bundle names a cohort after its patient count, so the id moves with
    every rebuild (v8 cohort1_n27 -> v10 cohort1_n29). A pinned id silently matches no
    run at all, which is worse than failing: the page builds, with nothing in it.
    """
    ids = list(manifest.get("cohorts", {}))
    if len(ids) == 1:
        return ids[0]
    if COHORT_FALLBACK in ids:
        return COHORT_FALLBACK
    raise SystemExit(f"manifest carries {len(ids)} cohorts {ids} and none is "
                     f"{COHORT_FALLBACK}; re-run make_coverage_bundle.py")


def load_p2_f2():
    sp = importlib.util.spec_from_file_location("p2fig1", ROOT / "00_Paper2_Figures.py")
    P2 = importlib.util.module_from_spec(sp); sp.loader.exec_module(P2)
    sf = importlib.util.spec_from_file_location("p2fig2", ROOT / "00_paper2_figures2_2.py")
    F2 = importlib.util.module_from_spec(sf); sf.loader.exec_module(F2)
    return P2, F2


def paper_runs(manifest):
    """The manifest's runs that a paper figure is built from, in the paper's order."""
    by = {}
    cohort = cohort_id(manifest)
    for r in manifest["runs"]:
        parts = r["id"].split("__")
        if len(parts) != 3 or r.get("cohort_id") != cohort:
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
  /* the template's panel (its Run dropdown, pial / inflated / glass, radius ...) is
     what boot() would show until ui2Init() replaces it; nothing of it is on screen
     before the compare UI is built (ui2Init adds body.u2ready at its end) */
  body:not(.u2ready) #controls, body:not(.u2ready) #prov, body:not(.u2ready) #hint { visibility:hidden; }
  body:not(.u2ready) #ctlBody { transition:none; }   /* a phone folds it before the reveal */
  /* the splash is opaque (the template panel is not seen through it); the staged line
     sits on the white glass page, so it is dark and has no spinner */
  #status { background:#000; }
  #status.staged .spin { display:none; }
  #status.staged .statusText { color:#3a3f47; opacity:1; font-weight:500; }
  #status.failed .spin { animation:none; }
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
  #ui2 .u2ch em.rl { font-style:normal; font-size:9px; font-weight:700; margin-left:1px; }
  #ui2 .u2ch:hover { border-color:var(--accent); }
  #ui2 .u2ch.on { background:#1d2029; border-color:var(--c); box-shadow:inset 0 0 0 1px var(--c); }
  #ui2 .u2ch.on i { color:#cdd3dc; }
  /* one patient holds more than half the cluster */
  .dom { color:#ff5a5f; font-weight:700; margin-left:1px; }
  /* the feature set the centroid is drawn on; green = the run's own features */
  #ui2 .u2rp { height:24px; border-radius:6px; background:#15161b; cursor:pointer; padding:0 4px;
               border:1px solid rgba(255,255,255,.14); color:var(--fg); font:inherit;
               font-size:10.5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  #ui2 .u2rp:hover { border-color:var(--accent); }
  #ui2 .u2rp.own { border-color:#3fb950; }
  #ui2 .u2rp.on { background:var(--accent); color:#001; font-weight:700; }
  #ui2 .u2rp.on.own { box-shadow:inset 0 0 0 2px #3fb950; }
  #ui2 #cpPaperRow { display:block; color:var(--muted); font-size:10.5px; margin-top:6px;
                     cursor:pointer; line-height:1.35; }
  #ui2 #cpPaperRow input { accent-color:var(--accent); margin-right:4px; }
  #ui2 .u2t .ownkey { color:#3fb950; }
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
  body.u2gridon #prov { display:none !important; }

  /* THE CENTROID, PER VIEW. One cluster: bottom right, as before. Grid: bottom left,
     the width of the control panel, under it, and hovering a panel previews that
     cluster. Overlay: the single panel gives way to one small centroid per selected
     cluster, laid out around the edges so the brain stays visible. The writing that
     used to sit under it is gone - the feature set is chosen on the left panel. */
  #centroidPanel { z-index:6; }
  #centroidPanel #cpNote, #centroidPanel #cpRepRow { display:none !important; }
  body[data-u2mode="grid"] #centroidPanel { left:14px; right:auto; bottom:14px; width:304px; }
  body[data-u2mode="grid"] #centroidPanel canvas { width:276px; height:124px; }
  body[data-u2mode="grid"] #ctlBody { max-height:calc(100vh - 320px); }
  body[data-u2mode="overlay"] #centroidPanel { display:none !important; }
  body[data-u2mode="overlay"] #prov { display:none !important; }
  #u2cens { position:fixed; inset:0; z-index:6; display:none; pointer-events:none; }
  #u2cens.on { display:block; }
  /* the width and the canvas size must match U2_CW / U2_CANH in UI2_JS */
  #u2cens .u2cen { position:absolute; width:276px; padding:5px 6px 6px; cursor:grab;
                   background:var(--panel-bg); backdrop-filter:blur(8px); border-radius:9px;
                   border:1px solid rgba(255,255,255,.10); border-left:3px solid var(--c);
                   box-shadow:0 6px 24px rgba(0,0,0,.45); pointer-events:auto;
                   touch-action:none; user-select:none; }
  #u2cens .u2cen:hover { border-color:var(--accent); border-left-color:var(--c); }
  #u2cens .u2cen.drag { cursor:grabbing; border-color:var(--accent); }
  #u2cens .u2cen .h { display:flex; align-items:center; gap:5px; font-size:10.5px; margin:0 0 4px; }
  #u2cens .u2cen .h .sw { width:9px; height:9px; border-radius:2px; background:var(--c); flex:none; }
  #u2cens .u2cen .h i { font-style:normal; font-size:9.5px; color:var(--muted); margin-left:auto; }
  #u2cens .u2cen canvas { display:block; width:276px; height:122px; border-radius:5px;
                          background:#ffffff; }
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

  /* THE SAMPLES BEHIND A CLUSTER (2026-10-15). Double-click a grid panel, or press v,
     and the overlay splits in two: the panels compress into the top half and every
     electrode that cluster contains is drawn underneath as its own concatenated ERSP,
     on the same global colour scale the centroid above it uses. The question it answers
     is whether a centroid is a response TYPE or the average of unlike things - which no
     centroid, however well drawn, can show. The grid stays on screen on purpose: the
     samples are only readable against the mean they came from - and because that is
     the reason, v opens the grid when it is closed (2026-10-16) instead of returning
     early, which was indistinguishable from a broken key. */
  #u2grid.sampon { grid-template-rows:auto minmax(0, 1fr) minmax(0, 1fr); }
  #u2samp { display:none; overflow:auto; margin-top:9px; padding-top:1px;
            border-top:1px solid rgba(255,255,255,.16); }
  #u2grid.sampon #u2samp { display:block; }
  /* the header scrolls with nothing - it carries the counts and the two definitions */
  #u2samp .sh { position:sticky; top:0; z-index:2; display:flex; align-items:baseline;
                gap:8px; flex-wrap:wrap; padding:8px 0 8px; background:rgba(0,0,0,.94); }
  #u2samp .sh .t { font-size:12.5px; font-weight:600; }
  #u2samp .sh .s { font-size:10.5px; color:var(--muted); }
  #u2samp .sh .sp { margin-left:auto; }
  #u2samp .sh button { background:#15161b; color:var(--muted); font:inherit; font-size:10.5px;
                       border:1px solid rgba(255,255,255,.14); border-radius:6px;
                       padding:3px 8px; cursor:pointer; }
  #u2samp .sh button:hover { color:var(--fg); border-color:var(--accent); }
  #u2samp .sh button.on { color:var(--fg); border-color:var(--accent); background:#1d2230; }
  #u2samp .rows { display:grid; gap:7px; align-content:start; padding-bottom:10px;
                  grid-template-columns:repeat(auto-fill, minmax(188px, 1fr)); }
  #u2samp .r { border-radius:7px; overflow:hidden; background:#000;
               border:1px solid rgba(255,255,255,.10); border-left:3px solid var(--c); }
  /* a SOLID left edge is an argmax member, a DASHED one an electrode that only clears
     the loading minimum - the difference the two definitions turn on, per row */
  #u2samp .r.weak { border-left-style:dashed; }
  #u2samp .r canvas { display:block; width:100%; height:auto; background:#ffffff; }
  #u2samp .r .c { display:flex; align-items:center; gap:5px; padding:3px 6px 4px;
                  font-size:10px; border-top:1px solid rgba(255,255,255,.07); }
  #u2samp .r .c b { font-weight:600; overflow:hidden; text-overflow:ellipsis;
                    white-space:nowrap; }
  #u2samp .r .c .lab { color:var(--muted); margin-left:auto; flex:none;
                       font-variant-numeric:tabular-nums; }
  #u2samp .note { color:var(--muted); font-size:11px; padding:10px 2px 14px; }

  /* ON A PHONE the panel is a sheet that opens upward from a pill at the bottom left,
     closed until tapped, so the brain has the screen and the pill is never over it. The
     centroid sits above the pill at the full width; in the grid it steps aside (the
     grid is the clusters). The run box shrinks into the top-right corner, the keyboard
     hint has nothing to say to a thumb. Desktop: none of this applies. */
  /* (this block sits above the page's own panel rules, so anything that competes with
     one of them is written `body #x` to outrank it) */
  @media (max-width: 640px) {
    body #controls { top:auto; bottom:10px; left:10px; right:10px; width:auto; z-index:7;
                     padding:8px 12px 10px; }
    body #controls.collapsed { right:auto; padding:7px 13px; border-radius:999px; }
    body #controls.collapsed #ctlBody { width:0; }   /* else the folded body still sets the pill's width */
    body #controls h2 { font-size:12.5px; }
    body #ctlBody, body[data-u2mode="grid"] #ctlBody { max-height:calc(100vh - 150px); }
    body #centroidPanel { left:10px; right:10px; bottom:52px; width:auto; padding:8px 10px 9px; }
    body #centroidPanel canvas { width:100%; height:auto; }
    #controls:not(.collapsed) ~ #centroidPanel { display:none; }
    body[data-u2mode="grid"] #centroidPanel { display:none !important; }
    body #prov { top:8px; right:8px; max-width:64vw; font-size:9.5px; line-height:1.4; }
    body #hint { display:none; }
    #u2grid { padding:10px 8px 54px; }
    #u2gg { grid-template-columns:repeat(auto-fill, minmax(160px, 1fr)); gap:8px; }
    #u2samp .rows { grid-template-columns:repeat(auto-fill, minmax(148px, 1fr)); gap:6px; }
    #u2cens .u2cen { width:min(276px, calc(100vw - 20px)); }
    #u2cens .u2cen canvas { width:100%; height:auto; }
  }
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
// the loading screen (the turning brain) stays up at least this long, so a cached
// load does not flash past it
const U2_MIN_LOAD_MS = 2000;
let U2_HOLD = false, U2_HIDE_PENDING = false;   // boot(): the splash is up; a hide was asked for meanwhile
const UI2 = { on:false, mode:"single", sel:new Set(), gridBy:"clusters",
              busy:false, gridOpen:false, seq:0,
              samp:null,            // {cid, runId, by, scale}: the electrodes under the grid
              cenPos:new Map() };   // cluster -> where its overlay card was dragged
const U2_METHODS = [["cnmf", "cNMF"], ["kmeans", "k-means"], ["hierarchical", "Ward"]];
const U2_MLABEL  = { cnmf:"cNMF", kmeans:"k-means", hierarchical:"Ward" };
const U2_FSETS   = [["concat_bands2", "2 bands"], ["concat_bands5", "5 bands"],
                    ["concat_bands15", "15 bands"], ["concat_hfa", "HFA"]];
const u2esc = t => String(t == null ? "" : t)
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const u2Split = id => { const p = String(id).split("__"); return { method:p[0], fset:p[1] }; };
const u2Run   = (m, f) => ((M && M.runs) || []).find(r => r.id.indexOf(m + "__" + f + "__") === 0);
const u2Defs  = () => DEFS.filter(d => d.cluster !== undefined && !d.hidden);
const u2Pos   = () => { const i = u2Defs().findIndex(d => d.id === state.map); return i < 0 ? 0 : i; };
// the run's own row count for a cluster, at no gate: the number the manifest publishes
// and the report prints, kept for the tooltips
const u2Rows  = c => (RUN && RUN.cluster_sizes && RUN.cluster_sizes[String(c)]) || 0;

// ── what is actually on the brain, and whose it is ───────────────────────────
// cluster_sizes and cluster_patient_counts in the bundle are computed over the run's
// ROWS with no loading gate, so they cannot follow the minimum-loading slider. These
// count the CONTACTS the slider leaves drawn - the same test updateElectrodes applies -
// so the number on a chip is the number of dots on the brain and the composition is the
// composition of those dots. Counted over localised contacts, which is fewer than the
// run's rows: the tooltips carry both.
const u2TallyCache = new Map();
function u2TallyOf(key, lab, load, cohortId, rel) {
  const ck = `${key}|${state.minLoad}|${rel ? "r" : "-"}`;
  if (u2TallyCache.has(ck)) return u2TallyCache.get(ck);
  const n = {}, pat = {}, rl = {};
  if (lab && CONTACTS) {
    for (const i of (CONTACTS.cohorts[cohortId] || [])) {
      const L = lab[i];
      if (!Array.isArray(L) || !L.length) continue;
      if (load) { const lv = load[i]; if (lv == null || lv < state.minLoad) continue; }
      const p = CONTACTS.patient[i];
      for (const c of L) {
        n[c] = (n[c] || 0) + 1;
        const m = pat[c] || (pat[c] = {});
        m[p] = (m[p] || 0) + 1;
        if (rel && rel[i] === 1) rl[c] = (rl[c] || 0) + 1;
      }
    }
  }
  const out = { n, pat, rel: rel ? rl : null };
  if (u2TallyCache.size > 80) u2TallyCache.clear();
  u2TallyCache.set(ck, out);
  return out;
}
// RELIABILITY (2026-10-13): contacts.json["reliability"][run] marks, per contact, whether it
// gets the same component from the full fit and from each trial half (plot_hg_timebins_test).
// Measured at the run's published K only, so the chips show it at that K and not at others.
const relForK = () => (RUN0 && CONTACTS && CONTACTS.reliability && Number(KSEL) === Number(RUN0.k))
  ? (CONTACTS.reliability[RUN0.id] || null) : null;
const u2Now   = () => u2TallyOf(`${RUN0 ? RUN0.id : "?"}|${KSEL}`, labelsForK(), loadingsForK(),
                                RUN ? RUN.cohort_id : "", relForK());
const u2RelPct = c => { const T = u2Now(); if (!T.rel || !T.n[c]) return null;
  return Math.round(100 * (T.rel[c] || 0) / T.n[c]); };
const u2RelCol = p => p >= 70 ? "#7fd38a" : p >= 50 ? "#f0c060" : "#f08080";
const u2RelText = c => { const p = u2RelPct(c); return p == null ? "" :
  ` · ${p}% reliable (${u2Now().rel[c] || 0} of ${u2Size(c)} drawn: same component from both trial halves)`; };
const u2Size  = c => u2Now().n[c] || 0;          // contacts drawn at the current minimum

// ── one patient, more than half the cluster ──────────────────────────────────
// A cluster that is mostly one patient is that patient's cortex, not a response type
// shared across the cohort, and it is the first thing to know before reading anything
// into it. Recomputed with the counts above, so raising the minimum can make the mark
// appear or go away - which is itself worth seeing.
function u2Dom(counts) {
  if (!counts) return null;
  let tot = 0, pid = null, top = 0;
  for (const p in counts) { tot += counts[p]; if (counts[p] > top) { top = counts[p]; pid = p; } }
  if (!tot || !pid) return null;
  return { pid, n: top, tot, frac: top / tot };
}
const u2DomOf = cid => u2Dom(u2Now().pat[cid]);
const u2DomPct = d => d ? Math.round(100 * d.frac) : 0;
const u2DomMark = d => (d && d.frac > 0.5) ? '<b class="dom" title="one patient holds more '
  + `than half of this cluster">*</b>` : "";
const u2DomText = d => d
  ? ` · biggest patient ${d.pid}, ${u2DomPct(d)}%${d.frac > 0.5 ? " — over half" : ""}` : "";

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
refreshMapList = function () {
  u2RefreshMapList();
  // a new run or a new cut is a new set of clusters, so where their cards were dragged
  // means nothing any more; the ring lays itself out again
  UI2.cenPos.clear();
  if (UI2.on) u2Sync();
};

// ── another run's labels at a K, for the cross-run grids ─────────────────────
// The published cut is already in memory (CONTACTS.runs / CONTACTS.loadings); any other
// K comes from that run's own sweep, fetched once and kept. No coverage, no neighbour
// index: a glass panel is contacts only.
const u2RD = new Map();
async function u2Data(runId, k) {
  if (RUN0 && runId === RUN0.id && Number(k) === Number(KSEL))
    return { lab: labelsForK(), load: loadingsForK(), clusters: (RUN.clusters || []).slice(),
             sizes: RUN.cluster_sizes || {}, pat: RUN.cluster_patient_counts || {},
             cohort: RUN.cohort_id };
  const key = runId + "|" + k;
  if (u2RD.has(key)) return u2RD.get(key);
  const r = ((M && M.runs) || []).find(x => x.id === runId);
  let out = null;
  try {
    if (r && Number(r.k) === Number(k)) {
      out = { lab: CONTACTS.runs[runId] || null,
              load: (CONTACTS.loadings || {})[runId] || null,
              clusters: figureOrder(runId, k, (r.clusters || []).slice()),
              sizes: r.cluster_sizes || {}, pat: r.cluster_patient_counts || {},
              cohort: r.cohort_id };
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
              sizes: st.cluster_sizes || {}, pat: st.cluster_patient_counts || {},
              cohort: r.cohort_id };
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
    <div class="u2g">
      <div class="u2t">Contact colour</div>
      <div class="segs" id="u2colour">
        <button type="button" data-c="cluster" class="on">by cluster</button>
        <button type="button" data-c="patient">by patient</button>
      </div>
      <div class="cohort" id="u2colnote">—</div>
    </div>
    <div class="u2g" id="u2cenwrap">
      <div class="u2t">Centroid on <span class="u2sub" id="u2cennote"></span></div>
      <div id="u2reps" class="u2chips"></div>
    </div>
    <div class="u2g" id="u2loadwrap"><div class="u2t">Minimum loading</div></div>
    <div class="u2g" id="u2btns"></div>`;
  // MOVED, not rebuilt: these carry the handlers wireUI() bound to them, and the K
  // buttons carry their own display toggling.
  $("u2kwrap").appendChild($("kRow"));
  $("u2loadwrap").appendChild($("loadRow"));
  $("u2loadwrap").appendChild($("loadNote"));
  $("u2btns").appendChild($("pdfBtn"));
  // the centroid's "paper definition" toggle belongs with the other centroid controls
  // now that nothing is written under the centroid itself
  $("u2cenwrap").appendChild($("cpPaperRow"));
  // the surface / radius / ceiling / contrast / vertex-gate controls: hidden, kept
  document.querySelectorAll("#ctlBody .grp").forEach(g => { g.style.display = "none"; });
  const g = document.createElement("div");
  g.id = "u2grid";
  g.innerHTML = `<div id="u2gh"></div><div id="u2gg"></div><div id="u2samp"></div>`;
  document.body.appendChild(g);
  const c = document.createElement("div");
  c.id = "u2cens";
  document.body.appendChild(c);
  window.addEventListener("resize", () => { if (UI2.mode === "overlay") u2Cens(); });
}

// ── which feature set the centroid is drawn on ───────────────────────────────
// The same clusters, averaged over another representation of the same electrodes -
// the bundle ships the proof that the rows line up. The run's OWN features carry a
// green outline, so a centroid drawn on anything else is visibly a cross-representation
// view rather than the thing the clustering was fitted to.
function u2Reps() {
  const box = $("u2reps");
  if (!box) return;
  if (!CEN || !CEN_INDEX) { box.innerHTML = ""; $("u2cennote").textContent = ""; return; }
  const own = CEN.meta.feature_set;
  const have = Object.keys(CEN_INDEX.feature_sets).filter(f => f === own || CEN_INDEX.rows_aligned);
  // the listed sets and nothing else. This used to append every set the centroid
  // bundle happened to carry but U2_FSETS did not list, unlabelled, which is how
  // raw "concat_b..." keys ended up as chips next to the named ones.
  const order = U2_FSETS.map(x => x[0]).filter(f => have.includes(f));
  box.innerHTML = order.map(f => {
    const short = (U2_FSETS.find(x => x[0] === f) || [f, f])[1];
    return `<button type="button" class="u2rp${f === CEN_REP ? " on" : ""}`
         + `${f === own ? " own" : ""}" data-rep="${u2esc(f)}" `
         + `title="${u2esc(REP_LABEL[f] || f)}${f === own ? " — this run's own features" : ""}">`
         + `${u2esc(short)}</button>`;
  }).join("");
  box.querySelectorAll("button[data-rep]").forEach(b => { b.onclick = () => u2SetRep(b.dataset.rep); });
  $("u2cennote").innerHTML = order.length > 1
    ? `<span class="ownkey">green</span> = this run's own`
    : "";
}

async function u2SetRep(rep) {
  if (UI2.busy || rep === CEN_REP) return;
  u2Lock(true);
  try {
    CEN_REP = rep;
    const sel = $("cpRep"); if (sel) sel.value = rep;    // the hidden select stays in step
    await cenX(rep);
  } catch (e) {
    console.warn("centroid representation:", e);
  } finally {
    // the open samples strip is drawn ON this representation too, so it has to follow.
    // Without this its canvases keep the PREVIOUS representation while the centroid
    // above them switches - two different feature sets on screen, both captioned as one,
    // which is the exact comparison the strip exists to make.
    u2Lock(false); u2Reps(); drawCentroid(); if (UI2.samp) u2Samples();
  }
}

// ── the centroid, per view ───────────────────────────────────────────────────
function u2SizeCP() {
  const cv = $("cpCanvas");
  if (!cv) return;
  const w = UI2.mode === "grid" ? 276 : 380;
  if (cv.width !== w) { cv.width = w; cv.height = Math.round(w * 170 / 380); }
}

// hovering a panel in the by-cluster grid previews that cluster without selecting it
function u2Preview(cid) {
  const cv = $("cpCanvas");
  if (!cv || !CEN || !CEN_INDEX || cid === undefined || Number.isNaN(cid)) return;
  try {
    const st = renderCentroid(cv.getContext("2d"), cv.width, cv.height, Number(KSEL),
                              Number(cid), CEN_REP, state.minLoad, false, cv.width / 380);
    if (!st) return;
  } catch (e) { return; }
  const pos = u2Defs().findIndex(d => Number(d.cluster) === Number(cid));
  const dm = u2DomOf(Number(cid));
  $("cpTitle").textContent = `#${pos + 1} · Cluster ${cid} · n = ${u2Size(cid)}`
    + (dm ? ` (${dm.pid} ${u2DomPct(dm)}%)` : "");
  $("centroidPanel").classList.remove("hidden");
}

// OVERLAY: one small centroid per selected cluster, laid out around the edges of the
// window so the brain in the middle stays visible. Slots are filled edge by edge -
// right, bottom, top, then the left below the control panel - taking roughly an equal
// share of each, so three cards sit on three sides rather than stacking in a column.
// card width and canvas height; the CSS above repeats them and must agree. Wide enough
// that the centroid's right-hand gutter - the colour bar or the y axis, the scale every
// cluster of the run shares - is drawn at a readable size next to the heatmap.
const U2_CW = 276, U2_CANH = 122, U2_CH = U2_CANH + 21;
const u2Clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
function u2Slots(n) {
  const W = window.innerWidth, H = window.innerHeight, m = 14, gap = 8;
  const ctl = $("controls").getBoundingClientRect();
  const x0 = Math.max(m, ctl.right + gap);                 // rows must clear the panel
  const rightX = W - m - U2_CW, bottomY = H - m - 18 - U2_CH, topY = m;
  // a phone: the controls are a pill at the bottom (44 px with its margin, whatever the
  // sheet is doing mid-fold), so one column down the right side from the top to the
  // pill; anything past it takes the fallback corner and is dragged
  if (u2Phone()) {
    const cap = Math.max(1, Math.floor((H - 44 - m) / (U2_CH + gap)));
    return Array.from({ length: Math.min(n, cap) }, (_, i) => ({ x: rightX, y: m + i * (U2_CH + gap) }));
  }
  const colCap = Math.max(1, Math.floor((H - 2 * m + gap) / (U2_CH + gap)));
  const rowCap = Math.max(0, Math.floor((rightX - gap - x0 + gap) / (U2_CW + gap)));
  const leftCap = Math.max(0, Math.floor((H - m - (ctl.bottom + gap) + gap) / (U2_CH + gap)));
  const edges = [
    { cap: colCap,  at: i => ({ x: rightX, y: m + i * (U2_CH + gap) }) },
    { cap: rowCap,  at: i => ({ x: rightX - (i + 1) * (U2_CW + gap), y: bottomY }) },
    { cap: rowCap,  at: i => ({ x: rightX - (i + 1) * (U2_CW + gap), y: topY }) },
    { cap: leftCap, at: i => ({ x: m, y: ctl.bottom + gap + i * (U2_CH + gap) }) },
  ];
  const out = [];
  let rem = n;
  for (let e = 0; e < edges.length && rem > 0; e++) {
    const take = Math.min(edges[e].cap, Math.ceil(rem / (edges.length - e)));
    for (let i = 0; i < take; i++) out.push(edges[e].at(i));
    rem -= take;
  }
  return out;
}

function u2Cens() {
  const box = $("u2cens");
  if (!box) return;
  const show = UI2.on && UI2.mode === "overlay" && CEN && CEN_INDEX && UI2.sel.size > 0
               && CEN.meta.ks.includes(Number(KSEL));
  box.classList.toggle("on", !!show);
  if (!show) { box.innerHTML = ""; return; }
  const defs = u2Defs(), K = defs.length;
  const items = defs.map((d, i) => ({ d, i })).filter(x => UI2.sel.has(x.d.cluster));
  // a card the user has dragged keeps where it was put; the ring is laid out around the
  // ones that have not been moved, so dragging one does not shuffle the rest
  const slots = u2Slots(items.filter(x => !UI2.cenPos.has(x.d.cluster)).length);
  let q = 0;
  box.innerHTML = items.map(x => {
    const put = UI2.cenPos.get(x.d.cluster);
    const s = put ? { x: u2Clamp(put.x, 0, window.innerWidth - U2_CW - 4),
                      y: u2Clamp(put.y, 0, window.innerHeight - U2_CH - 4) }
                  : (slots[q++] || { x: 14, y: 14 });
    const dom = u2DomOf(x.d.cluster);
    return `<div class="u2cen" style="left:${s.x}px;top:${s.y}px;--c:${u2CSS(u2Hue(x.i, K))}" `
         + `title="cluster ${x.d.cluster} · ${u2Size(x.d.cluster)} contacts drawn of `
         + `${u2Rows(x.d.cluster)} in the run${u2esc(u2DomText(dom))}`
         + ` — drag to move, click to drop it from the overlay">`
         + `<div class="h"><span class="sw"></span><b>#${x.i + 1}</b>${u2DomMark(dom)}`
         + `<i>${u2Size(x.d.cluster)}`
         + `${dom ? ` (${u2esc(dom.pid)} ${u2DomPct(dom)}%)` : ""}</i></div>`
         + `<canvas width="${U2_CW}" height="${U2_CANH}"></canvas></div>`;
  }).join("");
  box.querySelectorAll(".u2cen").forEach((el, i) => {
    const cv = el.querySelector("canvas");
    try {
      renderCentroid(cv.getContext("2d"), cv.width, cv.height, Number(KSEL),
                     Number(items[i].d.cluster), CEN_REP, state.minLoad, false, cv.width / 380);
    } catch (e) { console.warn("overlay centroid:", e); }
    u2Drag(el, items[i].d);
  });
}

// Drag a card anywhere, click it to drop the cluster from the overlay: one pointer
// gesture, told apart by whether it moved. The pointer is captured, so a fast drag does
// not escape the card, and the brain underneath never sees the gesture.
function u2Drag(el, d) {
  const cid = Number(d.cluster);
  let sx = 0, sy = 0, ox = 0, oy = 0, moved = false, live = false;
  el.addEventListener("pointerdown", e => {
    if (e.button !== 0) return;
    live = true; moved = false;
    sx = e.clientX; sy = e.clientY;
    ox = el.offsetLeft; oy = el.offsetTop;
    try { el.setPointerCapture(e.pointerId); } catch (err) {}
    e.preventDefault();
  });
  el.addEventListener("pointermove", e => {
    if (!live) return;
    const dx = e.clientX - sx, dy = e.clientY - sy;
    if (!moved && Math.abs(dx) + Math.abs(dy) < 4) return;
    if (!moved) { moved = true; el.classList.add("drag"); }
    const x = u2Clamp(ox + dx, 0, window.innerWidth - el.offsetWidth);
    const y = u2Clamp(oy + dy, 0, window.innerHeight - el.offsetHeight);
    el.style.left = x + "px"; el.style.top = y + "px";
    UI2.cenPos.set(cid, { x, y });
  });
  const end = e => {
    if (!live) return;
    live = false;
    el.classList.remove("drag");
    try { el.releasePointerCapture(e.pointerId); } catch (err) {}
    if (!moved) u2ClickChip(d.id, cid);
  };
  el.addEventListener("pointerup", end);
  el.addEventListener("pointercancel", end);
}

// every path that redraws the single centroid - the map select, the minimum-loading
// slider, a new run's bundle - also sizes it for the current view and refreshes the
// overlay ring, so there is one place this can go out of step and it is here
const u2DrawCentroid1 = drawCentroid;
drawCentroid = function () {
  u2SizeCP();
  u2DrawCentroid1();
  if (UI2.on) u2Reps();     // loadCentroids ends here, and only then is CEN_INDEX known
  u2Cens();
};

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
    const dom = u2DomOf(d.cluster);
    return `<button type="button" class="u2ch${on ? " on" : ""}" data-map="${u2esc(d.id)}" `
         + `data-c="${d.cluster}" style="--c:${u2CSS(u2Hue(i, K))}" `
         + `title="position #${i + 1} in figure order · cluster ${d.cluster} · `
         + `${u2Size(d.cluster)} contacts drawn of ${u2Rows(d.cluster)} in the run`
         + `${u2esc(u2DomText(dom))}${u2esc(u2RelText(d.cluster))}">`
         + `<span class="sw"></span><b>#${i + 1}</b>${u2DomMark(dom)}`
         + `<i>${u2Size(d.cluster)}</i>`
         + (u2RelPct(d.cluster) == null ? "" : `<em class="rl" style="color:${u2RelCol(u2RelPct(d.cluster))}">${u2RelPct(d.cluster)}%</em>`)
         + `</button>`;
  }).join("");
  box.querySelectorAll("button[data-map]").forEach(b => {
    b.onclick = () => u2ClickChip(b.dataset.map, Number(b.dataset.c));
  });
  const pos = u2Pos(), d = defs[pos];
  const dm = d ? u2DomOf(d.cluster) : null;
  $("u2selnote").textContent = UI2.mode === "overlay"
    ? `${UI2.sel.size} selected`
    : (d ? `#${pos + 1} · cluster ${d.cluster} · n=${u2Size(d.cluster)}`
         + (dm ? ` (${dm.pid} ${u2DomPct(dm)}%)` : "")
         + (u2RelPct(d.cluster) == null ? "" : ` · ${u2RelPct(d.cluster)}% reliable`) : "");
}

function u2Note() {
  const n = $("u2modenote");
  if (!n) return;
  n.innerHTML = UI2.mode === "single"
    ? "One cluster on the glass brain." + (relForK() ? " The coloured % on a chip: its drawn contacts "
        + "that get the same component from both trial halves (green ≥ 70, amber ≥ 50)." : "")
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
  document.body.dataset.u2mode = UI2.mode;      // the centroid's place is a CSS matter
  u2Runs(); u2Chips(); u2Note(); u2Reps();
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
  if (m === "grid") { u2Sync(); drawCentroid(); u2Grid(); }
  else { u2CloseGrid(); updateElectrodes(); u2Sync(); drawCentroid(); }
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
  // counted the same way as the chips: the contacts this panel actually drew
  const T = u2TallyOf(`${p.runId}|${p.k}`, D.lab, D.load, D.cohort);
  return { im, cid, n: T.n[cid] || 0, rows: (D.sizes || {})[String(cid)],
           dom: u2Dom(T.pat[cid]) };
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

// The minimum-loading slider moved. It changes which contacts are drawn, so the counts
// and the composition on every chip and card change with it - and an open grid is now
// showing the wrong dots, so it re-captures once the slider settles rather than on every
// pixel of the drag.
let u2GridTimer = null;
function u2GridSoon() {
  if (!UI2.gridOpen) return;
  clearTimeout(u2GridTimer);
  u2GridTimer = setTimeout(() => { if (UI2.busy) return u2GridSoon(); u2Grid(); }, 450);
}
function u2MinLoad() {
  if (!UI2.on) return;
  u2Chips(); u2Cens(); u2GridSoon();
}

function u2CloseGrid() {
  u2SampClose();                     // the strip is part of the grid, not of the run
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
        fig.querySelector(".cap").innerHTML =
          `<span class="sw"></span><b>${u2esc(panels[i].title)}</b>${u2DomMark(r.dom)}`
          + `<span class="lab">${u2esc(panels[i].sub)} · n=${r.n}`
          + `${r.dom ? ` (${u2esc(r.dom.pid)} ${u2DomPct(r.dom)}%)` : ""}</span>`;
        fig.title = `cluster ${r.cid} · ${r.n} contacts drawn`
                  + `${r.rows !== undefined ? " of " + r.rows + " in the run" : ""}`
                  + u2DomText(r.dom);
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
      // hovering previews that cluster's centroid in the panel at the bottom left;
      // only within one run, since the centroid bundle is loaded per run
      if (UI2.gridBy === "clusters") {
        f.onmouseenter = () => u2Preview(Number(f.dataset.cluster));
        f.onmouseleave = () => drawCentroid();
      }
      // ONE click opens the cluster, TWO show the electrodes it contains. The open
      // waits out the double-click window: fired at once it closes the grid before
      // the second click can land, and the strip belongs to the grid.
      let hold = null;
      f.onclick = () => {
        if (hold) return;
        hold = setTimeout(async () => {
          hold = null;
          const runId = f.dataset.run, cid = Number(f.dataset.cluster);
          if (!runId) return;
          UI2.mode = "single"; u2CloseGrid();
          if (runId !== RUN0.id) await u2GoRun(runId);
          else {
            const d = u2Defs().find(x => Number(x.cluster) === cid);
            if (d) { UI2.sel = new Set([cid]); u2Pick(d.id); }
          }
          updateElectrodes(); u2Sync();
        }, 240);
      };
      f.ondblclick = () => {
        if (hold) { clearTimeout(hold); hold = null; }
        if (!f.dataset.run || f.dataset.cluster === undefined) return;
        u2SampToggle(Number(f.dataset.cluster), f.dataset.run);
      };
    });
  } catch (e) {
    console.error("compare grid failed:", e);
    const pr = $("u2prog"); if (pr) pr.textContent = "failed — see the console";
  } finally {
    u2Lock(false);
    updateElectrodes();                 // the live view comes back exactly as it was
    // K moving under the strip is fine - the same cluster at another cut. The RUN
    // moving is not: cluster 3 of k-means is not cluster 3 of cNMF, and re-aiming
    // the id would draw different electrodes under an unchanged heading.
    if (UI2.samp) { if (UI2.samp.runId !== RUN0.id) u2SampClose(); else u2Samples(); }
  }
}

// ── the electrodes behind a cluster ──────────────────────────────────────────
// A centroid is a mean, and the mean of unlike things looks exactly like a type. These
// panels draw the electrodes THEMSELVES - one small concatenated ERSP per electrode -
// underneath the grid, so the mean and the things it is a mean of are on screen at the
// same time. Double-click a panel, or press v. Nothing is fetched: x_<fset>.bin IS the
// matrix the centroid is averaged from, and cenWeights() already says which rows belong.
const U2_SW = 300, U2_SH = 132;        // the backing store of one sample canvas
let _u2SampObs = null;                 // panels draw as they scroll in, not all at once

// Row i of the feature matrix, as the stats object renderCentroid draws. No SD: one
// electrode has no spread.
function cenRow(i, rep) {
  const g = cenGeom(rep); if (!g) return null;
  const X = _cenX.get(g.rep);
  if (!X || X.length !== g.n * g.F || i < 0 || i >= g.n) return null;
  const s = CEN_INDEX.x_scale, mean = new Float32Array(g.F), b = i * g.F;
  for (let f = 0; f < g.F; f++) mean[f] = X[b + f] * s;
  return { mean, sd:null, empty:false, g, w:null, nMembers:1, nPass:1, weighted:false };
}
// "PAT_3066|HAD1" per row, if the bundle published the electrode order it proves
// identical across runs. A bundle from before row_keys was added still works - it just
// cannot put a name on a panel, and says "row 142" instead.
function cenRowName(i) {
  const keys = CEN_INDEX && CEN_INDEX.row_keys;
  if (!keys || !keys[i]) return { pid:"", el:"row " + i };
  const p = String(keys[i]).split("|");
  return { pid:p[0] || "", el:p[1] || ("row " + i) };
}
// THE SAMPLES' OWN SCALE, for the "own scale" button. The centroid's limits are the 99th
// percentile of |cluster MEAN|, and a single electrode is several times noisier than any
// mean of them, so on the shared scale most panels saturate. Same rule, measured on the
// rows actually shown: p99 of |value| for the heatmap, p0.5/p99.5 for the line. Every
// third feature is enough to put a percentile on 400 x 1350 numbers.
function u2SampLimits(rows, g) {
  const X = _cenX.get(g.rep), s = CEN_INDEX.x_scale, v = [];
  for (const r of rows) { const b = r.i * g.F; for (let f = 0; f < g.F; f += 3) v.push(X[b + f] * s); }
  if (!v.length) return null;
  const abs = v.map(Math.abs).sort((a, b) => a - b), srt = v.slice().sort((a, b) => a - b);
  const q = (arr, p) => arr[Math.min(arr.length - 1, Math.max(0, Math.floor(p * arr.length)))];
  const vlim = q(abs, 0.99) || 1, lo = q(srt, 0.005), hi = q(srt, 0.995);
  return { vlim, ylo:Math.min(lo, 0) * 1.06, yhi:Math.max(hi, 0) * 1.06 };
}

function u2SampClose() {
  UI2.samp = null;
  if (_u2SampObs) { _u2SampObs.disconnect(); _u2SampObs = null; }
  const g = $("u2grid"); if (g) g.classList.remove("sampon");
  const s = $("u2samp"); if (s) s.innerHTML = "";
}
// v and double-click are the same toggle; asking again for the cluster already open
// closes it. The membership definition starts as the one the centroid panel is showing,
// so the strip underneath is the mean above broken into its parts and not another cut.
function u2SampToggle(cid, runId) {
  const id = runId || (RUN0 && RUN0.id);
  if (cid === undefined || Number.isNaN(Number(cid))) return;
  if (UI2.samp && UI2.samp.cid === Number(cid) && UI2.samp.runId === id) return u2SampClose();
  const paper = !!(($("cpPaper") || {}).checked);
  UI2.samp = { cid:Number(cid), runId:id, scale:"centroid",
               by:(paper || !(CEN && CEN.W)) ? "argmax" : "load" };
  u2Samples();
}

function u2Samples() {
  const S = $("u2samp"), G = $("u2grid");
  if (!S || !G || !UI2.samp) return;
  G.classList.add("sampon");
  const note = t => { S.innerHTML = `<div class="note">${u2esc(t)}</div>`; };
  const cid = UI2.samp.cid, k = Number(KSEL);
  // the centroid bundle is loaded for ONE run at a time, so these can only ever be the
  // open run's electrodes. The algorithm and feature-set grids show other runs' panels.
  if (UI2.samp.runId !== (RUN0 && RUN0.id))
    return note("that panel belongs to another run \u2014 open it (\u2193 \u2191 or 1 2 3), then press v");
  if (!CEN || !CEN_INDEX) return note("no centroid bundle for this run");
  if (!CEN.meta.ks.includes(k)) return note(`this run has no cut at K=${k}`);
  const g = cenGeom(CEN_REP); if (!g) return note("no geometry for " + CEN_REP);
  if (!_cenX.get(g.rep)) {                       // the matrix for a cross-run "shown on"
    cenX(g.rep).then(() => { if (UI2.samp) u2Samples(); });
    return note("loading the electrode matrix\u2026");
  }
  const lab = CEN.meta.labels[String(k)];
  const r = cenWeights(k, cid, state.minLoad, false);
  if (!r || !lab) return note("no membership at this K");
  const weighted = r.weighted, byLoad = UI2.samp.by === "load" && weighted;
  // WHICH ELECTRODES. "argmax" is the paper's definition - the ones whose largest
  // loading is this cluster. "loading" is the panel's - every electrode at or over the
  // minimum, which takes in electrodes that belong more to another cluster, and those
  // are the ones worth looking at. Strongest loading first, so the top of the strip is
  // the cluster at its purest and the bottom is where it frays.
  // the RAW loading, not cenWeights's: that one is zeroed under the minimum, which is
  // right for a weighted mean and wrong for a label - an argmax member just under the
  // slider would read 0.00 and sort to the bottom of its own cluster. The minimum still
  // decides which electrodes the loading view shows; it does not decide the number.
  const woff = weighted ? CEN.meta.w_offsets[String(k)] : undefined;
  const rawW = i => woff === undefined ? 0 : CEN.W[woff + i * k + cid] * CEN.meta.w_scale;
  const rows = [];
  for (let i = 0; i < g.n; i++) {
    const isMem = lab[i] === cid;
    if (byLoad ? !(r.w[i] > 0) : !isMem) continue;
    rows.push({ i, w: weighted ? rawW(i) : 1, isMem });
  }
  rows.sort((a, b) => (b.w - a.w) || (a.i - b.i));
  if (!rows.length) return note("no electrode in this cluster at that definition");
  const own = UI2.samp.scale === "own" ? u2SampLimits(rows, g) : null;
  const nm = (RUN && RUN.clusters) ? RUN.clusters.map(Number).indexOf(cid) + 1 : 0;
  const defTxt = byLoad
    ? `loading \u2265 ${state.minLoad.toFixed(2)} \u00b7 ${r.nMembers} of them are argmax members`
    : weighted ? `argmax members \u00b7 ${r.nPass} clear loading ${state.minLoad.toFixed(2)}`
               : "members of a hard partition \u00b7 no loadings to order by";
  const scTxt = own
    ? `own scale, \u00b1${own.vlim.toFixed(1)} ${g.unit} \u2014 readable per electrode, NOT comparable with the centroid`
    // a line panel is bounded by CEN_YLIM, a heatmap by CEN_VLIM - say the one the
    // reader is actually looking at
    : `the shared \u00b1${g.nb <= 2 ? CEN_YLIM : CEN_VLIM} ${g.unit} scale \u2014 the same on every feature set, `
      + `so this panel, the centroid above it and any other run are comparable by eye`;
  S.innerHTML =
    `<div class="sh">
       <div class="t">Cluster ${cid}${nm ? ` (#${nm})` : ""} \u00b7 ${rows.length} electrodes</div>
       <div class="s">${u2esc(defTxt)} \u00b7 on ${u2esc(REP_LABEL[g.rep] || g.rep)} \u00b7 ${u2esc(scTxt)}</div>
       <div class="sp"></div>
       <button type="button" data-a="load"${weighted ? "" : " disabled"}>loading</button>
       <button type="button" data-a="argmax">argmax</button>
       <button type="button" data-a="scale">${own ? "centroid scale" : "own scale"}</button>
       <button type="button" data-a="close">close \u2715</button>
     </div><div class="rows">`
    + rows.map(x => {
        const n = cenRowName(x.i);
        return `<div class="r${x.isMem ? "" : " weak"}" data-i="${x.i}" style="--c:${clusterCSS(cid)}"`
             + ` title="${u2esc(n.pid + " " + n.el)}${weighted ? ` \u00b7 loading ${x.w.toFixed(3)}` : ""}`
             + `${x.isMem ? " \u00b7 argmax member" : " \u00b7 argmax goes to another cluster"}">`
             + `<canvas width="${U2_SW}" height="${U2_SH}"></canvas>`
             + `<div class="c"><b>${u2esc(n.pid)}</b><span>${u2esc(n.el)}</span>`
             + `<span class="lab">${weighted ? x.w.toFixed(2) : ""}</span></div></div>`;
      }).join("")
    + `</div>`;
  S.querySelectorAll(".sh button[data-a]").forEach(b => {
    b.classList.toggle("on", b.dataset.a === UI2.samp.by);
    b.onclick = () => {
      const a = b.dataset.a;
      if (a === "close") return u2SampClose();
      if (a === "scale") UI2.samp.scale = own ? "centroid" : "own";
      else UI2.samp.by = a;
      u2Samples();
    };
  });
  S.scrollTop = 0;
  // DRAWN AS THEY SCROLL IN. A cluster at K=6 can hold 400 electrodes, and 400 canvases
  // drawn up front lock the tab for seconds and hold tens of MB of backing store. Each
  // is drawn once, the first time it comes near the viewport, and then left alone.
  if (_u2SampObs) _u2SampObs.disconnect();
  _u2SampObs = new IntersectionObserver(es => {
    for (const e of es) {
      if (!e.isIntersecting) continue;
      const el = e.target;
      if (_u2SampObs) _u2SampObs.unobserve(el);
      if (el.dataset.drawn) continue;
      el.dataset.drawn = "1";
      const cv = el.querySelector("canvas"), st = cenRow(Number(el.dataset.i), g.rep);
      if (!cv || !st) continue;
      if (own) st.lim = own;
      renderCentroid(cv.getContext("2d"), cv.width, cv.height, k, cid, g.rep, 0, false,
                     cv.width / 380, st);
    }
  }, { root:S, rootMargin:"320px 0px" });
  S.querySelectorAll(".r").forEach(el => _u2SampObs.observe(el));
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
  } else if (e.key === "v") {
    // the electrodes behind the SELECTED panel. The strip sits under the grid and
    // is only readable against the means it came from - so rather than do nothing
    // with the grid shut (which read as a dead key), v opens the grid itself.
    const d = u2Defs()[u2Pos()];
    if (!d || d.cluster === undefined) return;
    if (!UI2.gridOpen) {
      u2SetMode("grid");
      if (!UI2.gridOpen) return;   // u2Grid() bails while a capture is in flight;
    }                              // never leave the strip under no grid
    u2SampToggle(d.cluster, RUN0.id);
  } else { return; }
  e.preventDefault();
}

// ── init, after boot has a run on screen ─────────────────────────────────────
function u2SyncColour() {
  const byPat = state.elecColour === "patient";
  $("u2colour").querySelectorAll("button").forEach(b =>
    b.classList.toggle("on", (b.dataset.c === "patient") === byPat));
  // THE COHORT'S PATIENTS, NOT THE TABLE'S. patientIndex() spans every contact the
  // coordinate file holds - 68 of them - so quoting its size here claimed a cohort more
  // than twice the real one. The index stays global on purpose, because that is what
  // keeps a patient's hue the same across runs; only the COUNT has to be the drawn one.
  const ids = (CONTACTS && RUN && CONTACTS.cohorts[RUN.cohort_id]) || [];
  const n = new Set(ids.map(i => CONTACTS.patient[i])).size;
  $("u2colnote").textContent = byPat
    ? `${n} patients \u00b7 hue is stable across runs and K, and arbitrary`
    : "the cluster palette, matching the figures";
}

// ON A PHONE (the CSS's 640-px sheet) the controls fold to a pill: at the start, so
// the brain comes first, and when a view is chosen, since the sheet would cover the
// grid or the overlay just asked for. Chips, K and run buttons leave it open.
const U2_NARROW = window.matchMedia("(max-width: 640px)");
const u2Phone = () => U2_NARROW.matches;
function u2Fold(open) {
  const c = $("controls");
  if (c.classList.contains("collapsed") === !open) return;
  c.classList.toggle("collapsed", !open); $("ctlChevron").textContent = open ? "\u25be" : "\u25b8";
}
// THE LOADING ROW, before render()'s glass early return. render() sets the caption and
// the slider's enabled state after painting the vertices, and in glass mode it returns
// before painting - so with glass from the first frame the row was never set, and on a
// run switch in glass mode it had always been stale.
function syncLoadRow() {
  const hasLoad = !!loadingsForK();
  $("loadRow").style.opacity = hasLoad ? "1" : ".35";
  $("minLoad").disabled = !hasLoad;
  $("loadNote").textContent = hasLoad
    ? `graded: ${RUN.k} components, contacts sized and faded by their top weight`
    : "graded loading — convex-NMF runs only; this run is a hard partition";
}

// crossing the line either way (a rotated phone, a resized window) takes the fold along;
// watched from both events, since an emulated viewport fires only the resize
let u2WasNarrow = U2_NARROW.matches;
const u2Cross = () => { if (U2_NARROW.matches !== u2WasNarrow) { u2WasNarrow = U2_NARROW.matches; u2Fold(!u2WasNarrow); } };
U2_NARROW.addEventListener("change", u2Cross);
window.addEventListener("resize", u2Cross);

async function ui2Init() {
  u2Build();
  // glass is the only surface here, and boot() loaded it as such (see the boot patch in
  // make_cluster_visualizer.py): no pial-then-glass reload, nothing to click
  UI2.on = true;
  const d0 = u2Defs()[u2Pos()];
  if (d0) UI2.sel = new Set([d0.cluster]);
  $("u2mode").querySelectorAll("button").forEach(b => {
    b.onclick = () => { if (UI2.busy) return; if (u2Phone()) u2Fold(); u2SetMode(b.dataset.m); };
  });
  $("u2gridby").querySelectorAll("button").forEach(b => {
    b.onclick = () => { if (UI2.busy) return; UI2.gridBy = b.dataset.g; u2Sync(); u2Grid(); };
  });
  // Colour only: every filter that decides WHICH contacts are drawn is left alone, so
  // flipping this re-encodes the same picture instead of changing it. A cluster that is
  // one patient's electrode strip is a solid block of one hue under "by patient" and
  // invisible under "by cluster", which is the question this cohort keeps raising.
  $("u2colour").querySelectorAll("button").forEach(b => {
    b.onclick = () => {
      if (UI2.busy || state.elecColour === b.dataset.c) return;
      state.elecColour = b.dataset.c;
      // colouring contacts you cannot see is not a setting anyone means to choose
      if (!state.showElec) { state.showElec = true; const c = $("showElec"); if (c) c.checked = true; }
      u2SyncColour();
      updateElectrodes();
    };
  });
  u2SyncColour();
  // the report is built one cluster at a time, never from a grid or an overlay
  $("pdfBtn").onclick = async () => { u2SetMode("single"); await exportReport(); u2Sync(); };
  window.addEventListener("keydown", u2Keys);
  if (u2Phone()) { u2Fold(); void $("controls").offsetHeight; }   // committed before the reveal
  u2Sync();
  drawCentroid();          // sizes the canvas for the view and draws the ring if needed
  document.body.classList.add("u2ready");   // the panel, the run box and the hint appear
}
'''


# ---------------------------------------------------------------------------
# the html
# ---------------------------------------------------------------------------
_REPORT_EXTRAS_JS = r'''
// ── report extras: cohort make-up and native anatomy ─────────────────────────
// Added 2026-09-28. contacts.json carries, from make_coverage_bundle.add_report_fields:
//   regions       a vocabulary of native anatomy labels
//   region[i]     index into it per contact, -1 where the BIDS tissueLabel gave none
//   patient_meta  {patient: {centre, lang}} from patient_meta.csv
// Every one is optional: a bundle built before this simply draws no extra bar.

// A stable colour per label, so a region keeps its colour between clusters and between
// reports. Hue from a string hash; fixed s/l so nothing comes out muddy or invisible.
function hueOf(str) {
  let h = 0;
  for (let i = 0; i < str.length; i++) h = (h * 31 + str.charCodeAt(i)) % 360;
  return h;
}
function labelColour(s, i, n) {
  return `hsl(${hueOf(s)} 52% ${n && i === n - 1 ? 78 : 56}%)`;
}

// Fixed, so a centre or a language keeps its colour between the cohort bars at the top
// and every cluster's own bars below - the hash colour would have given the same label
// one colour in one place and another elsewhere, which is exactly what stops the two
// being read against each other. Matches FIG C.0 on the status page.
const CEN_PAL_JS = { Bern: "#5b6bbf", Geneva: "#c77f2a" };
const LANG_PAL_JS = { GER: "#7d5ba6", FRE: "#3f9e8c", ENG: "#b5651d", ITA: "#8aa62f" };

// The patients of the run on screen, once each.
function runPatients() {
  const lab = labelsForK(), out = new Set();
  for (const i of (CONTACTS.cohorts[RUN.cohort_id] || [])) {
    const L = lab && lab[i];
    if (Array.isArray(L) && L.length) out.add(CONTACTS.patient[i]);
  }
  return out;
}

// Patients grouped by one field of patient_meta, as stackbar items + a caption.
function patientsBy(field) {
  const meta = CONTACTS.patient_meta;
  if (!meta) return null;
  const cnt = new Map();
  for (const p of runPatients()) {
    const v = (meta[p] || {})[field] || "unknown";
    cnt.set(v, (cnt.get(v) || 0) + 1);
  }
  if (!cnt.size) return null;
  // biggest first, but "unknown" always last so it never leads the bar
  const keys = [...cnt.keys()].sort((a, b) =>
    (a === "unknown") - (b === "unknown") || cnt.get(b) - cnt.get(a) || a.localeCompare(b));
  const tot = [...cnt.values()].reduce((a, b) => a + b, 0);
  const pal = field === "centre" ? CEN_PAL_JS : (field === "lang" ? LANG_PAL_JS : null);
  const nm = field === "centre" ? CENTRE_NAME : null;
  return {
    tot,
    items: keys.map(k => ({
      label: (nm && nm[k]) || k, value: cnt.get(k),
      color: k === "unknown" ? "#c8ccd2" : ((pal && pal[k]) || labelColour(k))
    })),
    cap: keys.map(k => `${esc((nm && nm[k]) || k)} ${cnt.get(k)}`).join(" · ")
  };
}

// One labelled bar with its caption.
function metaBar(title, hint, d) {
  if (!d) return "";
  return `<div><div class="sbcap" style="margin:0 0 3px"><b>${esc(title)}</b> — ${esc(hint)}</div>
    <div class="sbtrack">${stackbar(d.items, d.tot, 15)}</div>
    <div class="sbcap">${d.cap}</div></div>`;
}

function cohortBarsHTML() {
  const c = patientsBy("centre"), l = patientsBy("lang");
  if (!c && !l) return "";
  return `<div class="two" style="margin-top:10px">
    ${metaBar("Recording centre", `${c ? c.tot : 0} patients`, c)}
    ${metaBar("Task language", "from the behavioural filename, or read off the stimuli", l)}
  </div>`;
}

// Native anatomy of one cluster: the top regions, the rest folded into "other".
// Counted over the contacts the run gave that cluster, which is fewer than the
// cluster's samples wherever a contact has no native label.
function regionItemsFor(cid, topN) {
  if (!CONTACTS.regions || !CONTACTS.region) return null;
  const lab = labelsForK(), cnt = new Map();
  let n = 0, unl = 0;
  for (const i of (CONTACTS.cohorts[RUN.cohort_id] || [])) {
    const L = lab && lab[i];
    if (!Array.isArray(L) || L.indexOf(cid) < 0) continue;
    n++;
    const ri = CONTACTS.region[i];
    if (ri == null || ri < 0) { unl++; continue; }
    const r = CONTACTS.regions[ri];
    cnt.set(r, (cnt.get(r) || 0) + 1);
  }
  if (!cnt.size) return null;
  const sorted = [...cnt.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  const keep = sorted.slice(0, topN || 6);
  const rest = sorted.slice(topN || 6).reduce((a, e) => a + e[1], 0);
  const items = keep.map(([r, v]) => ({ label: r, value: v, color: labelColour(r) }));
  if (rest) items.push({ label: `other (${sorted.length - keep.length} regions)`, value: rest, color: "#aeb4bc" });
  const tot = n - unl;
  return {
    items, tot, unl,
    cap: keep.slice(0, 3).map(([r, v]) => `${esc(r)} ${(100 * v / tot).toFixed(0)}%`).join(" · ")
         + (unl ? ` · ${unl} unlabelled` : "")
  };
}

// A cluster's electrodes split by one patient_meta field. Counted over ELECTRODES, not
// patients, so it reads against the patient composition bar directly above it: the
// question is how much of THIS cluster came from one centre or one language, and a
// cluster carried by a single centre is the confound worth seeing.
function clusterSplitBy(cid, field) {
  const meta = CONTACTS.patient_meta;
  if (!meta) return null;
  const lab = labelsForK(), cnt = new Map();
  let n = 0;
  for (const i of (CONTACTS.cohorts[RUN.cohort_id] || [])) {
    const L = lab && lab[i];
    if (!Array.isArray(L) || L.indexOf(cid) < 0) continue;
    const v = (meta[CONTACTS.patient[i]] || {})[field] || "unknown";
    cnt.set(v, (cnt.get(v) || 0) + 1);
    n++;
  }
  if (!n) return null;
  const keys = [...cnt.keys()].sort((a, b) =>
    (a === "unknown") - (b === "unknown") || cnt.get(b) - cnt.get(a) || a.localeCompare(b));
  return { tot: n, keys, cnt };
}

// The recording centres by the names Lora uses for them.
const CENTRE_NAME = { Bern: "InselSpital", Geneva: "HUG" };

function miniSplit(cid, field, pal, rename) {
  const d = clusterSplitBy(cid, field);
  if (!d) return "";
  const items = d.keys.map(k => ({
    label: (rename && rename[k]) || k, value: d.cnt.get(k),
    color: k === "unknown" ? "#c8ccd2" : pal(k)
  }));
  const cap = items.map(it => `${esc(it.label)} ${(100 * it.value / d.tot).toFixed(0)}%`).join(" · ");
  return `<div style="flex:1 1 0;min-width:0">
      <div class="sbtrack">${stackbar(items, d.tot, 7)}</div>
      <div class="sbcap" style="opacity:.8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${cap}</div>
    </div>`;
}

// The native anatomy (top 3 plus the rest), under the cluster's patient composition and
// in the same cell. Centre and language used to sit here too, squeezed in beside it;
// they are a column of their own now (centreLangCell), because who recorded a cluster
// and in what language is read across clusters, and a reader cannot compare values that
// are buried at three different widths inside another column.
function regionTrack(cid) {
  const d = regionItemsFor(cid, 3);
  if (!d) return "";
  return `<div class="sbtrack" style="margin-top:3px">${stackbar(d.items, d.tot, 9)}</div>
     <div class="sbcap" style="opacity:.85">${d.cap}</div>`;
}

// The condensed centre / language column of a cluster card. Stacked, not side by side:
// the column is narrow, and two full-width bars read better than two half-width ones.
function centreLangCell(cid) {
  const cen = miniSplit(cid, "centre", k => CEN_PAL_JS[k] || "#9aa3ad", CENTRE_NAME);
  const lan = miniSplit(cid, "lang", k => LANG_PAL_JS[k] || "#9aa3ad", null);
  if (!cen && !lan) return "<div></div>";
  return `<div style="display:flex;flex-direction:column;gap:4px;min-width:0">${cen}${lan}</div>`;
}
'''


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
        '    <label id="cpPaperRow"><input type="checkbox" id="cpPaper"> paper definition '
        '<span class="hint">(FIG 1: argmax members, unweighted)</span></label>\n'
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
// The 2026-10-13 family and the sets that came before it. A name here is what the
// "shown on" selector and every grid caption say, so the two 15-band sets and the two
// HFA sets have to be distinguishable on sight.
// ONE colour limit for every feature set, run and K, so any two heatmaps on the page
// are comparable by eye. It used to be the 99th percentile of |cluster mean|, which
// drifted per run and per set (+/-1.4, +/-1.8 ...) and, being a limit of MEANS, was far
// too tight for single electrodes - the samples strip saturated almost everywhere.
// +/-3 dB is the chosen limit; p99 of |single electrode| is 2.52-3.54 on the four sets
// (p99.9 = 4.03 / 4.54 / 5.52 / 4.09 for bands2 / bands5 / bands15 / hfa), so the top
// per cent or so of an electrode's range clips while a cluster MEAN, which peaks near
// 1.8, still keeps real contrast. All four sets are in dB, which is what makes one
// shared scale meaningful at all.
const CEN_VLIM = 3;
// and the LINE axis, which cannot share that number: it draws mean +/- SD, and measured
// over every run drawn as lines (hfa and bands2, three algorithms, K 5-12) the extreme of
// mean +/- SD is +3.70 dB (Ward on 2 bands; the rest sit between -2.46 and +3.18). A fixed
// +/-3 would cut the top off that band, and a flat-topped band reads as data rather than as
// a clipped axis. +/-4 clears it and is still the same on every panel - two line plots with
// different y axes cannot be compared, which is the whole reason 2 bands is drawn as lines.
const CEN_YLIM = 4;
const REP_LABEL = { concat_bands2: "2 bands (30-50, 70-150 Hz)",
                    concat_bands5: "5 bands (1-170 Hz in 5 bands)",
                    concat_bands15: "15 bands (1-170 Hz in 15 bands)",
                    concat_hfa: "HFA only (70-150 Hz)" };
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
const KROW_REPS = ["concat_bands15", "concat_bands5", "concat_hfa"];
function cenGeom(rep) {
  const r = rep || CEN_REP, g = CEN_INDEX.feature_sets[r];
  if (!g) return null;
  return { rep: r, n: g.n, nc: g.conds.length, nb: g.bands.length, nt: g.nt, conds: g.conds,
           bands: g.bands,
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
// weighted SD too, for the sets drawn as lines (<= 2 bands), where it is shaded
function cenMean(k, j, min, paper, rep) {
  const g = cenGeom(rep); if (!g) return null;
  const X = _cenX.get(g.rep);
  if (!X || X.length !== g.n * g.F || g.n !== CEN.meta.n) return null;
  const r = cenWeights(k, j, min, paper); if (!r) return null;
  const mean = new Float32Array(g.F), sd = g.nb <= 2 ? new Float32Array(g.F) : null;
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
// Both numbers are now fixed for every run, feature set and K - see CEN_VLIM / CEN_YLIM,
// which were measured over the cohort rather than guessed. k and rep are kept so callers
// do not change, and so a future per-set exception has somewhere to go. This used to call
// cenMean() once per cluster on every lookup, purely to derive limits that are constants.
function cenLimits(k, rep) {
  return { vlim: CEN_VLIM, ylo: -CEN_YLIM, yhi: CEN_YLIM };
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
// `stIn` draws a stats object the caller already built - cenRow()'s single electrode
// for the samples strip - instead of the weighted mean, and its `lim` overrides the
// global colour limits, which are the limits of MEANS and saturate on one electrode.
function renderCentroid(ctx, Wd, Hd, k, j, rep, min, paper, scale, stIn) {
  const st = stIn || cenMean(k, j, min, paper, rep); if (!st) return null;
  const g = st.g, lim = st.lim || cenLimits(k, g.rep), nc = g.nc, nb = g.nb, nt = g.nt, ncol = nc * nt, col = clusterCSS(j);
  const S = scale || 1, font = px => `${Math.round(px * S)}px sans-serif`;
  // everything on white, as the figures are: a dark ground around a heatmap read as a
  // black frame on the report's white page. The heatmap gets a thin grey frame instead,
  // so its edge still shows where RdBu's white zero meets the page.
  const line = nb <= 2;        // HFA, and 2 bands as two traces rather than two rows
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
    // ONE TRACE PER BAND. The highest band carries the cluster colour - it is the one
    // that matches the dots on the brain - and any band below it is grey, so it reads as
    // the second trace rather than as another cluster. With nb === 1 (HFA) this is
    // exactly what it always was: one cluster-coloured line and its band.
    const LOW = "#8d969e", LOW_FILL = "#b9c0c6";
    const bandCol = b => (b === nb - 1 ? col : LOW);
    for (let c = 0; c < nc; c++) {
      const x0 = padL + w * c / nc, xw = w / nc, xt = t => x0 + xw * t / (nt - 1);
      // every SD band first, then every line, so no fill is drawn over a line
      if (st.sd) for (let b = 0; b < nb; b++) {
        const f0 = (c * nb + b) * nt;
        ctx.fillStyle = b === nb - 1 ? col : LOW_FILL;
        ctx.globalAlpha = b === nb - 1 ? 0.22 : 0.3; ctx.beginPath();
        for (let t = 0; t < nt; t++) { const f = f0 + t; if (!t) ctx.moveTo(xt(t), sy(st.mean[f] + st.sd[f])); else ctx.lineTo(xt(t), sy(st.mean[f] + st.sd[f])); }
        for (let t = nt - 1; t >= 0; t--) { const f = f0 + t; ctx.lineTo(xt(t), sy(st.mean[f] - st.sd[f])); }
        ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
      }
      for (let b = 0; b < nb; b++) {
        const f0 = (c * nb + b) * nt;
        ctx.strokeStyle = bandCol(b); ctx.lineWidth = (b === nb - 1 ? 1.4 : 1.1) * S;
        ctx.beginPath();
        for (let t = 0; t < nt; t++) { const f = f0 + t; if (!t) ctx.moveTo(xt(t), sy(st.mean[f])); else ctx.lineTo(xt(t), sy(st.mean[f])); }
        ctx.stroke();
      }
    }
    // which line is which. Only with more than one band - the single HFA line needs no
    // legend, and the panel is small enough that an unnecessary one costs real space.
    if (nb > 1) {
      const labs = g.bands || [], lh = (nb * 10 + 5) * S;
      const lx = padL + 5 * S, ly = padT + 4 * S, lw = 52 * S;
      ctx.fillStyle = "rgba(255,255,255,.8)"; ctx.fillRect(lx, ly, lw, lh);
      ctx.textAlign = "left"; ctx.font = font(8);
      for (let b = nb - 1, q = 0; b >= 0; b--, q++) {
        const yy = ly + 4 * S + q * 10 * S;
        ctx.fillStyle = bandCol(b); ctx.fillRect(lx + 4 * S, yy + 2 * S, 8 * S, 3 * S);
        ctx.fillStyle = T.muted;
        ctx.fillText(labs[b] || `band ${b + 1}`, lx + 15 * S, yy + 6 * S);
      }
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
  // scale from the canvas, not 1: the compare UI resizes this canvas per view
  const st = renderCentroid(cv.getContext("2d"), cv.width, cv.height, k, j, CEN_REP,
                            state.minLoad, paper, cv.width / 380);
  if (!st) { P.classList.add("hidden"); return; }
  P.classList.remove("hidden");
  const g = st.g, own = m.feature_set, cross = CEN_REP !== own;
  const pos = (RUN && RUN.clusters) ? RUN.clusters.map(Number).indexOf(j) + 1 : 0;
  $("cpTitle").textContent = `#${pos || "?"} · Cluster ${j} · ` + (st.weighted
    ? `${st.nPass} contacts at loading ≥ ${state.minLoad.toFixed(2)} · ${st.nMembers} argmax members`
    : `n = ${st.nMembers} members`) + (cross ? ` · on ${REP_LABEL[CEN_REP]}` : "");
  const sdTxt = g.nb <= 2 ? " · band = ±1 SD" + (paper ? "" : " (weighted)") : "";
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
    # The overlay is hidden through hideStatus() now, so it fades instead of snapping;
    # this anchor moved with it. It failing loudly when the source changed is the point
    # of anchoring, and is how the change was caught.
    sub('  autoScale(); render(); updateElectrodes();\n  hideStatus();\n}',
        '  autoScale(); render(); updateElectrodes();\n  hideStatus();\n'
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
        '    $("minLoadVal").textContent = state.minLoad.toFixed(2);\n'
        '    updateElectrodes(); drawCentroid();\n'
        '    if (typeof u2MinLoad === "function") u2MinLoad();   // counts follow the gate\n  };',
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
        "<b>o</b> overlay · <b>v</b> electrodes · <b>h</b> panels · <b>c</b> controls</div>", "compare UI: hint")
    # THE FIRST SECONDS, REWRITTEN AS ONE BLOCK (the template's tail from `await meshReady`
    # to the closing brace). Glass from the first frame: the template boots on the opaque
    # pial mesh and ui2Init used to click the glass button afterwards - a grey brain under
    # the template's own panel, then a second mesh load; the state is set before the mesh
    # is asked for, so selectRun's render / updateElectrodes already draw glass and
    # by-cluster contacts. The run load starts as soon as the mesh is in and runs UNDER
    # the splash, which holds U2_MIN_LOAD_MS from the start of boot (U2_HOLD keeps
    # showStatus from dropping to the staged line and defers hideStatus); then the brain
    # comes up, whatever is left of the run load shows as the staged line, and the
    # compare UI is built.
    sub("  const meshReady = loadSurface(state.surf);\n",
        "  U2_HOLD = true;\n"
        "  const held = new Promise(res => setTimeout(res, U2_MIN_LOAD_MS));\n"
        "  state.glass = true; state.surf = \"pial\"; state.showElec = true; state.elecByCluster = true;\n"
        "  const meshReady = loadSurface(\"pial\", GLASS_RGBA);   // glass: uniform rgba, or the alpha is ignored\n",
        "compare UI: boot glass")
    sub("  await meshReady;\n"
        "  $(\"brain\").classList.add(\"up\");\n"
        "  stageStatus(\"Loading coverage\u2026\");\n"
        "\n"
        "  wireUI();\n"
        "  PROGRESS = (got, total) =>\n"
        "    setStatus(`Loading coverage\u2026 ${(got / 1048576).toFixed(1)} / ${(total / 1048576).toFixed(1)} MB`);\n"
        "  try { await selectRun(pick.id); } finally { PROGRESS = null; }\n"
        "  hideStatus();\n"
        "}\n",
        "  await meshReady;\n"
        "  applyOpacity();                    // white backdrop, 0.18 alpha\n"
        "  wireUI();\n"
        "  $(\"showElec\").checked = true;      // glass forces the contacts on; the (hidden) box says so\n"
        "  PROGRESS = (got, total) =>\n"
        "    setStatus(`Loading coverage\u2026 ${(got / 1048576).toFixed(1)} / ${(total / 1048576).toFixed(1)} MB`);\n"
        "  const runLoad = selectRun(pick.id).finally(() => { PROGRESS = null; });\n"
        "  runLoad.catch(() => {});           // awaited below; this only keeps the rejection handled meanwhile\n"
        "  await held;\n"
        "  U2_HOLD = false;\n"
        "  $(\"brain\").classList.add(\"up\");\n"
        "  if (U2_HIDE_PENDING) hideStatus(); else $(\"status\").classList.add(\"staged\");\n"
        "  await runLoad;\n"
        "  await ui2Init();\n"
        "  hideStatus();\n"
        "}\n",
        "compare UI: boot tail")
    # the loading row (caption, slider enabled) is set before render()'s glass early
    # return: with glass from the first frame it was never set, and on a run switch in
    # glass mode it had always been stale
    sub("  if (state.glass) { nv.drawScene(); return; }\n  const d = curDef(), hi = d.raw ? MAXC * state.climMax : state.climMax;",
        "  syncLoadRow();                       // before the glass return, or never in glass mode\n"
        "  if (state.glass) { nv.drawScene(); return; }\n  const d = curDef(), hi = d.raw ? MAXC * state.climMax : state.climMax;",
        "compare UI: loading row before glass")
    sub("  const hasLoad = !!loadingsForK();\n  const lr = $(\"loadRow\");\n"
        "  lr.style.opacity = hasLoad ? \"1\" : \".35\";\n  $(\"minLoad\").disabled = !hasLoad;\n"
        "  $(\"loadNote\").textContent = hasLoad\n"
        "    ? `graded: ${RUN.k} components, contacts sized and faded by their top weight`\n"
        "    : \"graded loading \u2014 convex-NMF runs only; this run is a hard partition\";\n\n",
        "", "compare UI: loading row once")
    # a failure keeps its message and shows the wired template panel; a module that
    # never runs says so
    sub("function showStatus(t, staged) {\n  const e = $(\"status\"); if (!e) return;\n  e.style.display = \"grid\";",
        "let HIDE_T = 0;   // hideStatus's pending display:none, cancelled by the next showStatus\n"
        "function showStatus(t, staged) {\n  const e = $(\"status\"); if (!e) return;\n"
        "  clearTimeout(HIDE_T);\n  e.style.display = \"grid\";",
        "compare UI: showStatus cancels the hide")
    sub("  setTimeout(() => { e.style.display = \"none\"; }, 500);   // after the fade, not instead of it",
        "  HIDE_T = setTimeout(() => { e.style.display = \"none\"; }, 500);   // after the fade, not instead of it",
        "compare UI: hideStatus timer")
    sub("boot().catch(e => { showStatus(\"Failed to load: \" + e.message, false); console.error(e); });",
        "boot().catch(e => {\n"
        "  U2_HOLD = false;\n"
        "  showStatus(\"Failed to load: \" + e.message + \" \u2014 reload the page\", false);\n"
        "  $(\"status\").classList.add(\"failed\");\n"
        "  document.body.classList.add(\"u2ready\");   // whatever chrome is wired is at least visible\n"
        "  console.error(e);\n});",
        "compare UI: boot failure")
    sub("<script type=\"module\">",
        "<script type=\"module\" onerror=\"document.getElementById('statusText').textContent="
        "'The viewer library did not load (unpkg.com unreachable?) \u2014 reload, or check the network'\">",
        "compare UI: module load failure")
    sub("    if (e.key === \"h\") document.querySelectorAll(\".panel\").forEach(p => p.classList.toggle(\"hidden\"));",
        "    if (!document.body.classList.contains(\"u2ready\")) return;   // nothing to hide or fold yet\n"
        "    if (e.key === \"h\") document.querySelectorAll(\".panel\").forEach(p => p.classList.toggle(\"hidden\"));",
        "compare UI: h/c after the reveal")
    sub("function showStatus(t, staged) {\n  const e = $(\"status\"); if (!e) return;\n"
        "  clearTimeout(HIDE_T);\n  e.style.display = \"grid\";\n"
        "  e.classList.remove(\"gone\");\n"
        "  e.classList.toggle(\"staged\", staged !== false);",
        "function showStatus(t, staged) {\n  const e = $(\"status\"); if (!e) return;\n"
        "  clearTimeout(HIDE_T);\n  e.style.display = \"grid\";\n"
        "  e.classList.remove(\"gone\");\n"
        "  e.classList.toggle(\"staged\", staged !== false && !U2_HOLD);   // the splash holds",
        "compare UI: showStatus under the hold")
    sub("function hideStatus() {\n  const e = $(\"status\"); if (!e) return;\n",
        "function hideStatus() {\n  const e = $(\"status\"); if (!e) return;\n"
        "  if (U2_HOLD) { U2_HIDE_PENDING = true; return; }   // after the splash, not under it\n",
        "compare UI: hideStatus under the hold")
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
        "const REPORT_3D = " + ("true" if REPORT_3D else "false") + ";\n"
        "// turns and the every-K brains only in the full mode (REPORT_3D = True in the generator)\n"
        "const REPORT_FULL = " + ("true" if REPORT_3D is True else "false") + ";\n", "report: the flag")
    sub("      const st = render(); await sleep(80);\n"
        "      const shots = [];\n"
        "      for (const [az, el] of VIEWS) shots.push(await captureView(az, el, CAPW, true));\n"
        "      const spinFrames = await captureSpin(0, SPIN_N, SPIN_W, true);\n",
        "      const st = render();          // the stats are read either way\n"
        "      const shots = [];\n"
        "      let spinFrames = [];\n"
        "      if (REPORT_FULL) {\n"
        "        // the probability stills and turn: painted-surface captures, which the glass-only\n"
        "        // page cannot make (render() paints no vertex in glass mode) - full mode only\n"
        "        await sleep(80);\n"
        "        for (const [az, el] of VIEWS) shots.push(await captureView(az, el, CAPW, true));\n"
        "        spinFrames = await captureSpin(0, SPIN_N, SPIN_W, true);\n"
        "      }\n", "report: per-cluster captures")

    # A LATERAL STILL IS ONE HEMISPHERE (2026-10-13, Lora: "remove the hemisphere behind when
    # screenshotting"; both laterals, surface and dots). contactsHemi() dropped only the far
    # side's dots, and only in the every-K pass: the glass stills of the report kept both
    # hemispheres and both sides' dots, which through the glass read as if they were all on
    # the near side. farHemi() hides the far hemisphere's mesh as well (Niivue skips a mesh
    # whose .visible is false); the stills loops in the glass pass call it per view, and the
    # dorsal view keeps both.
    sub('const HEMI_FOR_VIEW = ["lh", "rh", "both"];   // matches VIEWS\n',
        'const HEMI_FOR_VIEW = ["lh", "rh", "both"];   // matches VIEWS\n'
        '// The far hemisphere itself, not only its dots, for the lateral stills of the report: the\n'
        '// near side is drawn alone (Niivue skips a mesh whose .visible is false), its contacts\n'
        '// filtered by contactsHemi. "both" restores the scene.\n'
        'function farHemi(near) {\n'
        '  for (const h of ["lh", "rh"]) {\n'
        '    const m = hemis[h] && hemis[h].mesh;\n'
        '    if (m) m.visible = (near === "both" || h === near);\n'
        '  }\n'
        '  contactsHemi(near);          // re-emits the dots and redraws\n'
        '}\n', "report: farHemi")

    # the glass pass - stills, turns and every cluster at every K - skipped whole, so the
    # surface is never swapped and nothing needs restoring
    i = s.index("    const glassWas = { glass: state.glass, byCluster: state.elecByCluster,")
    j = s.index("    // ── whole-run figures", i)
    seg = s[i:j]
    # inside the pass, the turns and the every-K brains are the full mode's; the glass stills
    # of the clusters on screen are what "stills" keeps
    for old, new in (
        ("      glassCovSpin = await captureSpin(0, SPIN_N, SPIN_W, true, gSpinBox);\n",
         "      if (REPORT_FULL) glassCovSpin = await captureSpin(0, SPIN_N, SPIN_W, true, gSpinBox);\n"),
        ("        blocks[i].glassSpin = await captureSpin(0, SPIN_N, SPIN_W, true, gSpinBox);\n",
         "        blocks[i].glassSpin = REPORT_FULL ? await captureSpin(0, SPIN_N, SPIN_W, true, gSpinBox) : [];\n"),
        ("      if (SWEEP && SWEEP.ks && SWEEP.ks.length > 1) {\n",
         "      if (REPORT_FULL && SWEEP && SWEEP.ks && SWEEP.ks.length > 1) {\n"),
        # stills mode: the K on screen gets its table brains from the stills just captured,
        # so the per-K cluster table (chip, centroids, three views, composition) is drawn
        # at the exported cut without the every-K pass
        ("        blocks[i].glassSpin = REPORT_FULL ? await captureSpin(0, SPIN_N, SPIN_W, true, gSpinBox) : [];\n      }\n",
         "        blocks[i].glassSpin = REPORT_FULL ? await captureSpin(0, SPIN_N, SPIN_W, true, gSpinBox) : [];\n      }\n"
         "      { // table brains for the exported cut, keyed by the row's cluster id (the sweep column's).\n"
         "        // At the published K of a convex-NMF run the fit on screen (run_decomposition, 300\n"
         "        // iterations) and the sweep's fit at that K (1000) are different clusterings, so a\n"
         "        // sweep cluster takes the stills of the published cluster it overlaps most; for\n"
         "        // k-means, Ward and every re-cut K the two are the same partition and the id matches.\n"
         "        const kNow = String(KSEL), pubK = Number(RUN0.k);\n"
         "        const rowIds = ((SWEEP && SWEEP.stats && SWEEP.stats[kNow]) || {}).clusters || [];\n"
         "        if (rowIds.length) {\n"
         "          let match = (s) => Number(s);\n"
         "          const sw = SWEEP.labels && SWEEP.labels[kNow], pb = CONTACTS.runs && CONTACTS.runs[RUN0.id];\n"
         "          if (Number(KSEL) === pubK && sw && pb) {\n"
         "            const ov = {};\n"
         "            for (let i = 0; i < sw.length; i++) {\n"
         "              const a = sw[i], b = pb[i];\n"
         "              if (!Array.isArray(a) || !Array.isArray(b)) continue;\n"
         "              for (const s of a) { const row = ov[s] || (ov[s] = {}); for (const p of b) row[p] = (row[p] || 0) + 1; }\n"
         "            }\n"
         "            match = (s) => { const row = ov[s] || {}; const best = Object.keys(row).sort((x, y) => row[y] - row[x])[0];\n"
         "                             return best == null ? Number(s) : Number(best); };\n"
         "          }\n"
         "          const byId = {};\n"
         "          for (const s of rowIds) { const b = blocks.find(x => Number(x.d.cluster) === match(s)); byId[String(s)] = b ? b.glassShots : null; }\n"
         "          KBRAINS[KSEL] = byId;\n"
         "        } }\n"),
        # a lateral still is one hemisphere (farHemi above): set per view, restored after;
        # the same in the full mode's every-K pass and in the finally
        ("        for (let v = 0; v < VIEWS.length; v++) {\n"
         "          const [az, el] = VIEWS[v];\n"
         "          gs.push(gBox[v] ? await captureRaw(az, el, CAPW, gBox[v], true)\n"
         "                          : await captureView(az, el, CAPW, true));\n"
         "        }\n"
         "        blocks[i].glassShots = gs;\n",
         "        for (let v = 0; v < VIEWS.length; v++) {\n"
         "          const [az, el] = VIEWS[v];\n"
         "          farHemi(HEMI_FOR_VIEW[v]);   // a lateral still is one hemisphere; dorsal keeps both\n"
         "          await sleep(40);\n"
         "          gs.push(gBox[v] ? await captureRaw(az, el, CAPW, gBox[v], true)\n"
         "                          : await captureView(az, el, CAPW, true));\n"
         "        }\n"
         "        farHemi(\"both\");\n"
         "        blocks[i].glassShots = gs;\n"),
        ("              contactsHemi(HEMI_FOR_VIEW[v]);\n",
         "              farHemi(HEMI_FOR_VIEW[v]);\n"),
        ("            contactsHemi(\"both\");\n",
         "            farHemi(\"both\");\n"),
        ("      contactsHemi(\"both\");  // a pass that failed mid-view must not leave dots missing\n",
         "      farHemi(\"both\");  // a pass that failed mid-view must not leave dots or a hemisphere missing\n"),
    ):
        if seg.count(old) != 1:
            raise SystemExit(f"anchor inside the glass pass matches {seg.count(old)} times:\n{old}")
        seg = seg.replace(old, new)
    s = s[:i] + "    if (REPORT_3D) {\n" + seg.rstrip() + "\n    }\n\n" + s[j:]
    n_patch += 1

    # the per-cluster probability stills (full mode only: painted-surface captures the
    # glass-only page cannot make), then the glass stills with a caption that no longer
    # points at "the probability maps above" when there are none
    i = s.index('  <p class="sechint">P(cluster ${k} | labelled electrode within')
    m = s.index('  <div class="brains glass">', i)
    j = s.index('  <div style="margin-top:16px">\n    <div class="${raster', m)
    glass = s[m:j]
    c0 = glass.index('  <p class="sechint">The contacts themselves on a glass brain, captured from the viewer at')
    c1 = glass.index("</p>", c0) + len("</p>")
    cap_full = glass[c0:c1]
    cap_stills = ('  <p class="sechint">The contacts themselves on a glass brain, captured from the viewer at three '
                  'angles, with no ${frozen.radius}&nbsp;mm smoothing &mdash; this is where the electrodes are.</p>')
    glass = glass[:c0] + "${REPORT_FULL ? `" + cap_full + "` : `" + cap_stills + "`}" + glass[c1:]
    s = (s[:i] + '  ${!REPORT_FULL ? "" : `' + s[i:m].rstrip() + '`}\n'
         + '  ${!REPORT_3D ? "" : `' + glass.rstrip() + '`}\n' + s[j:])
    n_patch += 1

    # THE PER-K CLUSTER TABLE DOES NOT WAIT FOR BRAINS (2026-10-13, Lora: "the main cluster
    # card ... the first big figure that shows comparisons of all clusters like a table" was
    # missing). The K panel drew its rows - chip, centroids, three views, composition,
    # centre / language - only when the every-K brain captures existed, and those left with
    # the 3D flag on 2026-09-07, so the table had been gone since. It is drawn whenever the
    # cut has clusters; a view cell without a capture is simply empty.
    sub("      ${(B.brains && B.brains.some(Boolean)) ? `",
        "      ${(B.ids && B.ids.length) ? `",
        "K panel: the table does not wait for brains")

    # THE EXPORTED CUT'S ROWS ARE THE CLUSTERS ON SCREEN (2026-10-13, Lora: "c1 c2 c3 ...
    # need to match the glass brain screenshot"). The table took every K's ids from the
    # sweep column, but at the PUBLISHED K of a convex-NMF run the published fit and the
    # sweep's fit at that K are two different clusterings (run_decomposition at 300
    # iterations vs sweep_decomposition at 1000): the stills, the sections and the flat
    # chips are the published one's, so a sweep id of the same number sat next to another
    # cluster's brain. The exported cut now takes RUN's clusters, sizes and composition -
    # the published fit at the published K, the sweep cut elsewhere, exactly what the
    # sections use - and the published cut's chips are the flat cluster_NN.png.
    # The table's rows are the sweep column's ids in FIGURE ORDER (figureOrder above), and the
    # brains were a positional array in id order - row i took brain i, so after the first
    # reordered cluster every row showed another cluster's brain. Brains are keyed by the
    # row's cluster id now, in both passes.
    sub("          const g = (B.brains || [])[i];\n",
        "          const g = (B.brains || {})[String(c)] || null;   // keyed by the row's cluster id\n",
        "K panel: brains looked up by cluster id")
    sub("          KBRAINS[kk] = shots;\n",
        "          KBRAINS[kk] = Object.fromEntries(ids.map((c, j) => [String(c), shots[j]]));\n",
        "glass pass: every-K brains keyed by cluster id")

    # ... the per-cluster pair of turns ...
    a = '    <div style="margin-top:14px">\n      <h2 class="sec">Seen from every angle</h2>'
    b = '${spin(glassSpin, `Cluster ${k} contacts`)}\n      </div>\n    </div>'
    i = s.index(a)
    j = s.index(b, i) + len(b)
    s = s[:i] + '    ${!REPORT_FULL ? "" : `' + s[i:j].lstrip() + '`}\n' + s[j:]
    n_patch += 1

    # ... and the two whole-cohort turn sections
    a = '<section class="card">\n  <div class="three">\n'
    if s.count(a) != 1:
        raise SystemExit(f"anchor for the glass turn sections matches {s.count(a)} times")
    i = s.index(a)
    j = s.index("${resultsHTML}", i)
    s = s[:i] + '${!REPORT_FULL ? "" : `' + s[i:j].rstrip() + '`}\n\n' + s[j:]
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

    # ---- cohort make-up and native anatomy ---------------------------------------
    # Three bars the report did not have: the patients by recording centre, the patients
    # by task language, and each cluster's native anatomy. The first two are counted over
    # PATIENTS, so they do not depend on the coordinate file - EL051 and EL052 have no
    # rows in it and are absent from the rest of the report, but they still count here.
    # The regions come from contacts.json's vocabulary + per-contact index, written by
    # make_coverage_bundle.add_report_fields; a bundle without them simply draws nothing.
    sub("function stackbar(items, total, h, scaleMax) {",
        _REPORT_EXTRAS_JS + "\nfunction stackbar(items, total, h, scaleMax) {",
        "report extras: helpers")

    # the two cohort bars, under the donut row they belong with
    sub('  ${summary}\n  ${note("overview")}',
        '  ${cohortBarsHTML()}\n  ${summary}\n  ${note("overview")}',
        "report extras: centre and language bars")

    # Centre and language get a column of their own on both card grids. These run AFTER
    # the "K rows" patches above, so every anchor here is the already-patched text: the
    # grid is eight columns by now, not the six in the source file, and the K panel's
    # header already lists KROW_REPS rather than a single "centroid".
    sub("grid-template-columns:30px 1fr 1fr 1fr 1.25fr 1.25fr .85fr 1.05fr;gap:8px",
        "grid-template-columns:30px 1fr 1fr 1fr 1.25fr 1.25fr .85fr 1.05fr .72fr;gap:8px",
        "report extras: card grid gains a column")

    # The current-K header still names one centroid where its row draws three, so it was
    # already two cells short of its own grid and everything after the chip sat a column
    # to the left of what it labelled. It gets the K panel's KROW_REPS labels, and then
    # both headers get the new one.
    D = "·"
    tail_h = (f'${{VIEWS.map(v => `<div>${{esc(v[2])}}</div>`).join("")}}'
              f'<div>who is in it {D} samples</div></div>')
    reps = '${KROW_REPS.map(r => `<div>${REP_LABEL[r] || r}</div>`).join("")}'
    for ind, old_head, tag in (
        ("  ", "<div></div><div>centroid</div>", "cards"),
        ("        ", "<div></div>" + reps, "K panel"),
    ):
        sub(f'{ind}<div class="cardhead">{old_head}\n{ind}  {tail_h}',
            f'{ind}<div class="cardhead"><div></div>{reps}\n{ind}  '
            + tail_h[:-len("</div>")] + f'<div>centre {D} language</div></div>',
            f"report extras: centre/language header ({tag})")

    # The axis row under each grid is spacers plus one bar scale, so it needs as many
    # cells as the grid has columns: the current-K row carried five spacers against eight
    # columns, the K panel seven. Both go to eight, for nine columns.
    # The leading newline pins the indentation - without it the two-space form is a
    # substring of the eight-space one and the anchor matches both grids.
    cell = "<div></div>"
    for ind, n_old, tag in (("  ", 5, "cards"), ("        ", 7, "K panel")):
        sub(f'\n{ind}<div class="cardaxis">{cell * n_old}',
            f'\n{ind}<div class="cardaxis">{cell * 8}',
            f"report extras: centre/language axis spacer ({tag})")
    for tail, cid, ind, tag in (
        ('${RUN.cluster_patients[String(k)]} patients${\n'
         '            top ? " · " + esc(top[0]) + " " + (100 * top[1] / (tot || 1)).toFixed(0)'
         ' + "%" : ""}</div>\n        </div>', "k", "        ", "cards"),
        ('${(st.cluster_patients || {})[String(c)] || 0} patients${\n'
         '                top ? " · " + esc(top[0]) + " " + (100 * top[1] / (tot || 1)).toFixed(0)'
         ' + "%" : ""}</div>\n            </div>', "c", "            ", "K panel"),
    ):
        sub(tail, tail + f"\n{ind}${{centreLangCell({cid})}}",
            f"report extras: centre/language column ({tag})")

    # the anatomy track stays under each cluster's patient composition, in that cell
    # the two row builders name the cluster differently - `k` in the cards, `c` in the
    # K panel - so each patch passes its own variable rather than binding a shared name
    for anchor, cid, tag in (
        (("        <div><div class=\"sbtrack\">${stackbar(items, tot, 15, barMax)}</div>\n"
          "          <div class=\"sbcap\">${tot} samples"), "k", "cards"),
        (("            <div><div class=\"sbtrack\">${stackbar(items, tot, 15, kBarMax)}</div>\n"
          "              <div class=\"sbcap\">${tot} samples"), "c", "K panel"),
    ):
        head, cap = anchor.split("\n")
        pad = " " * (len(cap) - len(cap.lstrip()))
        sub(anchor,
            head + "\n" + pad + "${regionTrack(" + cid + ")}\n" + cap,
            f"report extras: region track ({tag})")

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
