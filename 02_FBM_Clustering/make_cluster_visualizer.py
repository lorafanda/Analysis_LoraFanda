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
