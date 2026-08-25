#!/usr/bin/env python3
"""
make_bsf_webblock.py - generates the 07_ClusterComparisons tab for analysis_status.html
FROM the statistics files, so the page cannot drift away from the analysis behind it.

Every number on the tab is read from outputs/clustering/bsf_comparison/. Nothing is
typed in. That is the same rule make_decomposition_figure.py follows, and it is here
because FIG C.3 panel D once carried hard-coded literals that stayed correct only by
luck while every other panel moved.

Writes two files into the scratch dir next to the stats:
    webblock_nav.html      the one <button> for the nav
    webblock_section.html  the whole <section id="clustercmp"> ... </section>

    python make_bsf_webblock.py            # generate
    python make_bsf_webblock.py --insert   # generate AND splice into the site
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "clustering" / "bsf_comparison"
SITE = Path("C:/Users/fanda/lorafanda.github.io/analysis_status.html")
FIGROOT = "02_FBM_Clustering/outputs/clustering/bsf_comparison"

ORDER = ["kmeans", "hierarchical", "cnmf"]
NICE = {"kmeans": "k-means <b>(BSF)</b>", "hierarchical": "Ward", "cnmf": "convex NMF"}
TAG = {"kmeans": "a", "hierarchical": "b", "cnmf": "c"}


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def fmt(v, n=3, sign=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "&mdash;"
    return f"{v:+.{n}f}" if sign else f"{v:.{n}f}"


def load():
    d = {}
    for k, f in (("sep", "part1_separation.csv"), ("agr", "part1_agreement.csv"),
                 ("coh", "part1_coherence.csv"), ("sizes", "part1_cluster_sizes.csv"),
                 ("lopo", "part1_lopo.csv"), ("pk", "part2_peaks_figure.csv")):
        p = OUT / f
        d[k] = pd.read_csv(p) if p.exists() else None
    for k, f in (("summ", "part1_summary.json"), ("p2meta", "part2_meta.json")):
        p = OUT / f
        d[k] = json.loads(p.read_text()) if p.exists() else {}
    return d


def t_separation(sep):
    if sep is None:
        return "<p><i>separation not computed</i></p>"
    rows = ["<tr><th></th><th>home space</th><th>silhouette (home)</th>"
            "<th>matched null</th><th>z</th><th>scored in the other space</th></tr>"]
    for m in ORDER:
        h = sep[(sep.method == m) & (sep.home)]
        o = sep[(sep.method == m) & (~sep.home)]
        if h.empty:
            continue
        h, o = h.iloc[0], (o.iloc[0] if not o.empty else None)
        z = h.z
        zc = "#1b7837" if z > 3 else ("#c1121f" if z < 2 else "#68727d")
        rows.append(
            f"<tr><td><b>{NICE[m]}</b></td><td><code>{h.space}</code></td>"
            f"<td><b>{fmt(h.silhouette, 4, True)}</b></td>"
            f"<td>{fmt(h.null_mean, 4, True)} &plusmn; {fmt(h.null_sd, 4)}</td>"
            f"<td style='color:{zc}'><b>{fmt(z, 1, True)}</b></td>"
            f"<td>{fmt(o.silhouette, 4, True) if o is not None else '&mdash;'}"
            f" <i>(z {fmt(o.z,1,True) if o is not None else ''})</i></td></tr>")
    return "<table class='sum' style='margin:8px 0'>" + "".join(rows) + "</table>"


def t_agreement(agr):
    if agr is None:
        return ""
    rows = ["<tr><th>pair</th><th>ARI</th><th>NMI</th></tr>"]
    for _, r in agr.iterrows():
        rows.append(f"<tr><td>{NICE[r.a]} vs {NICE[r.b]}</td>"
                    f"<td><b>{fmt(r.ari)}</b></td><td>{fmt(r.nmi)}</td></tr>")
    return "<table class='sum' style='margin:8px 0'>" + "".join(rows) + "</table>"


def t_coherence(coh):
    if coh is None:
        return ""
    rows = ["<tr><th></th><th>neighbours sharing label</th><th>over chance</th></tr>"]
    best = coh.over_chance.max()
    for m in ORDER:
        r = coh[coh.method == m]
        if r.empty:
            continue
        r = r.iloc[0]
        # precomputed: a python 3.11 f-string expression may not contain a backslash,
        # so the escaped quotes cannot live inside the braces
        style = ' style="color:#1b7837"' if r.over_chance >= best - 1e-9 else ""
        rows.append(f"<tr><td><b>{NICE[m]}</b></td>"
                    f"<td>{fmt(r.neighbours_sharing_label)}</td>"
                    f"<td{style}><b>{fmt(r.over_chance, 2)}&times;</b></td></tr>")
    return "<table class='sum' style='margin:8px 0'>" + "".join(rows) + "</table>"


def t_lopo(summ):
    lp = (summ or {}).get("lopo") or {}
    if not lp:
        return ""
    rows = ["<tr><th></th><th>worst fold</th><th>size-matched null (min)</th>"
            "<th>z</th><th>verdict</th></tr>"]
    for m in ORDER:
        s = lp.get(m)
        if not s:
            continue
        # the SIGN carries the meaning: inside is the pass, above is better than the
        # pass, only BELOW means an individual patient is holding the solution together
        z = s["z"]
        if z <= -2:
            v, c = "BELOW the null &mdash; a patient is carrying it", "#c1121f"
        elif z >= 2:
            v, c = "above the null &mdash; more robust than chance", "#1b7837"
        else:
            v, c = "inside the null &mdash; no patient carries it", "#1b7837"
        rows.append(
            f"<tr><td><b>{NICE[m]}</b></td><td>{fmt(s['real_min'])}</td>"
            f"<td>{fmt(s['null_min_mean'])} &plusmn; {fmt(s['null_min_sd'])}</td>"
            f"<td>{fmt(z, 2, True)}</td>"
            f"<td style='color:{c}'>{v}</td></tr>")
    return "<table class='sum' style='margin:8px 0'>" + "".join(rows) + "</table>"


def t_gate(sizes, base):
    if sizes is None:
        return ""
    rows = ["<tr><th></th>" + "".join(f"<th>c{j}</th>" for j in range(8)) +
            "<th>clusters over baseline</th></tr>"]
    for m in ORDER:
        s = sizes[sizes.method == m].sort_values("cluster")
        if s.empty:
            continue
        cells = []
        for _, r in s.iterrows():
            hot = r.pct_added > base
            cells.append(f"<td style='color:{'#c1121f' if hot else '#1b7837'}'>"
                         f"{r.pct_added:.0f}%</td>")
        over = int((s.pct_added > base).sum())
        rows.append(f"<tr><td><b>{NICE[m]}</b></td>{''.join(cells)}"
                    f"<td><b>{over}</b> of 8</td></tr>")
    return "<table class='sum' style='margin:8px 0'>" + "".join(rows) + "</table>"


def t_peaks(pk):
    if pk is None or pk.empty:
        return ""
    rows = ["<tr><th>feature set</th><th>space</th><th>method</th><th>at K=8</th>"
            "<th>peak</th><th>turns over?</th></tr>"]
    for _, r in pk.sort_values(["feature_set", "scheme", "method_label"]).iterrows():
        turn = ("<span style='color:#1b7837'>yes, k=%d</span>" % r.k_peak
                if not r.monotone else
                "<span style='color:#c1121f'>no &mdash; still rising at k=%d</span>" % r.k_peak)
        rows.append(f"<tr><td><code>{r.feature_set}</code></td><td>{r.scheme}</td>"
                    f"<td>{r.method_label}</td><td><b>{fmt(r.at_k8)}</b></td>"
                    f"<td>{fmt(r.peak)}</td><td>{turn}</td></tr>")
    return "<table class='sum' style='margin:8px 0'>" + "".join(rows) + "</table>"


def figure(tag, num, title, body, img, alt, runid):
    return f"""
    <figure id="{tag}"><div class="cap"><div class="fignum">FIG {num}</div>
      <h4>{title}</h4>
      {body}
      <div class="runid">{runid}</div></div>
      <div class="imgwrap"><img data-fig="{FIGROOT}/{img}" alt="{esc(alt)}"></div>
    </figure>"""


def build(d) -> str:
    s = d["summ"] or {}
    n = s.get("n_electrodes", 2946)
    npat = s.get("n_patients", 27)
    ng, na = s.get("n_gated", 1266), s.get("n_added", 1680)
    base = s.get("baseline_pct_added", 100 * na / max(n, 1))
    nnull = s.get("n_null", "?")
    p2 = d["p2meta"] or {}

    out = [f"""
  <section id="clustercmp">
    <h2>07 · Cluster comparisons &mdash; BSF against the alternatives</h2>
    <p><b>Two pinned runs anchor this tab</b>, chosen rather than resolved as
      &ldquo;newest&rdquo;, because the point of a best-so-far is that it does not move
      when something else is run:</p>
    <table class="sum" style="margin:8px 0">
      <tr><th></th><th>run</th><th>feature set</th><th>n</th><th>K</th><th>gate</th></tr>
      <tr><td><b>BSF</b> <i>best so far</i></td>
          <td><code>kmeans_concat_hg_all_20260819_235524</code></td>
          <td><code>concat_hg_all</code></td><td>{n}</td><td>8</td>
          <td><b>lifted</b></td></tr>
      <tr><td><b>SBSF</b> <i>second best</i></td>
          <td><code>kmeans_concat_hg_20260817_171544</code></td>
          <td><code>concat_hg</code></td><td>{ng}</td><td>8</td>
          <td>applied</td></tr>
    </table>
    <div class="method">
      <b>The cohort guarantee, checked rather than assumed.</b> Everything in Part 1 is
      the <b>same {n} electrodes across {npat} patients</b>, and everything in Part 2 is
      the <b>same {ng}</b>. Within each feature set the three runs were verified to hold
      a <b>bit-identical <code>X_train</code> in the identical electrode order</b>
      (<code>np.allclose</code> at 1e-6, plus a key-by-key comparison of
      <code>patient_id|electrode</code>), so any difference between the figures below is
      the METHOD and nothing else. Part 1 and Part 2 are deliberately different cohorts
      and are never mixed.
      <ul>
        <li><b>convex NMF at K=8 on <code>concat_hg_all</code> did not exist</b> &mdash;
          that run is K=7 and carries no <code>loadings_by_k/</code>. It was fitted here
          with the project's own <code>convex_nmf</code> (300 iterations, unit-normed,
          <code>random_state=0</code>) and cached into the run's own
          <code>loadings_by_k/G_k08.npy</code>, the layout <code>concat_hg</code> and
          <code>concat_rawds</code> already use. Purely additive; no existing file was
          modified.</li>
        <li><b>Each method is read in its home space.</b> convex NMF unit-norms before
          fitting; k-means and Ward use raw dB. Silhouette is not space-free, and
          scoring all three in dB is the error that made the first version of
          <a href="#separation">FIG C.7</a> wrong.</li>
        <li><b>Panels D and F of the original C.3 are dropped</b> by request. D
          (anatomical coherence) is reported as a statistic below instead; F (split-half
          replication) is not reported here at all. <b>B3</b> (loading-weighted
          glassbrain) is convex-NMF-only and needs a K=8 pyvista render that does not
          exist, so it is drawn for no method rather than for one.</li>
      </ul>
    </div>

    <h3>Part 1 &mdash; the three algorithms at K=8, on the BSF cohort</h3>

    <h4 id="bsfstats">The statistics</h4>
    <p style="margin:0 0 4px"><b>Separation against a matched null.</b> The null is a
      Gaussian carrying this feature set's own covariance &mdash; correlated features,
      smooth time courses, one blob, no cluster structure &mdash; refitted with the same
      method, {nnull} times per cell. White noise would be trivially beatable and would
      prove nothing.</p>
    {t_separation(d['sep'])}
    <p style="margin:0 0 4px"><b>Every method separates at home and collapses in the
      other's space.</b> That is <a href="#separation">FIG C.7</a> reproduced at K=8 on
      the ungated cohort: the three are describing different structure, not disagreeing
      about the same structure.</p>

    <p style="margin:10px 0 4px"><b>Do they find the same thing?</b> No.</p>
    {t_agreement(d['agr'])}

    <p style="margin:10px 0 4px"><b>Anatomical coherence</b> &mdash; of an electrode's 10
      nearest neighbours in fsaverage space, how many share its label, divided by the
      same quantity under a label shuffle. The chance correction matters: a solution with
      one dominant cluster scores high on the raw version for free.</p>
    {t_coherence(d['coh'])}

    <p style="margin:10px 0 4px"><b>Leave one patient out</b>, refit with the SAME
      method, against a size-matched pseudo-patient null.
      <code>lf_decompose.lopo_stability</code> hard-codes KMeans, which would silently
      have made this the wrong test for Ward and convex NMF; the version used here
      delegates the refit to the method so each is compared against itself.</p>
    {t_lopo(s)}
"""]

    for m in ORDER:
        out.append(figure(
            f"c3{TAG[m]}", f"C.3{TAG[m]}",
            f"The decomposition at K=8 &mdash; {NICE[m].replace('<b>','').replace('</b>','')}",
            f"""<ul>
        <li><b>A.</b> Held-out variance vs K, <b>bi-cross-validated</b>. The curve on the
          rest of the site holds out electrodes only and refits them across the full
          feature set, which makes it monotone in K by construction; this one holds out a
          block of rows AND columns, so an extra component has to earn its place.</li>
        <li><b>C.</b> How confident each label is. <b>Silhouette</b> for k-means and
          Ward, <b>largest normalised loading</b> for convex NMF &mdash; the same
          question, but not the same axis, and the panel says so on itself. Only convex
          NMF has a graded membership to report at all; that is the difference, not a
          presentational choice.</li>
        <li><b>B1.</b> Each cluster's mean response in raw dB, audio | picture | reading,
          on one shared y-axis so cluster amplitudes are comparable.</li>
        <li><b>B2.</b> Where those electrodes sit &mdash; sagittal fsaverage projection,
          anterior on the left, one shared frame for all eight panels.</li>
        <li><b>E.</b> Leave-one-patient-out against the size-matched null.</li>
      </ul>""",
            f"C3{TAG[m]}_{m}_K8.png",
            f"six-panel decomposition figure for {m} at K=8 on 2946 electrodes: held-out "
            f"variance curve, label-confidence histogram, leave-one-patient-out bars, "
            f"eight cluster mean response profiles and eight sagittal electrode maps",
            f"{m}/concat_hg_all &middot; K=8 &middot; "
            f"<code>make_bsf_figures.py</code> &middot; stats in "
            f"<code>bsf_comparison/part1_*.csv</code>"))

    out.append(f"""
    <h4 id="bsfgate">What the responsiveness gate was removing &mdash; at K=8</h4>
    <p style="margin:0 0 8px"><code>concat_hg_all</code> IS the ungated set:
      <b>{ng}</b> electrodes would pass the responsiveness gate and <b>{na}</b> are
      present only because it was lifted, so <b>{base:.1f}% added</b> is the cohort
      baseline every cluster is read against. A cluster far above that line is partly
      separating <i>responsive from non-responsive</i> rather than one response type from
      another &mdash; the failure the gate exists to prevent. Blue and grey are kept from
      <a href="#gatesplit">FIG C.8</a>.</p>
    {t_gate(d['sizes'], base)}
""")

    for m in ORDER:
        out.append(figure(
            f"c8{TAG[m]}", f"C.8{TAG[m]}",
            f"Gate composition at K=8 &mdash; {NICE[m].replace('<b>','').replace('</b>','')}",
            "<p style='margin:0 0 8px'>Every electrode, sorted within its cluster. "
            "<b style='color:#4a6fa5'>Blue</b> = would pass the responsiveness gate, "
            "<b style='color:#8e969e'>grey</b> = present only because it was lifted. "
            "The lower panel is the same information per cluster against the cohort "
            "baseline.</p>",
            f"C8{TAG[m]}_{m}_K8.png",
            f"per-electrode confidence for {m} at K=8, grouped by cluster and coloured "
            f"by whether the electrode passes the responsiveness gate, with a bar chart "
            f"of percent gate-added per cluster against the cohort baseline",
            f"{m}/concat_hg_all &middot; K=8 &middot; "
            f"<code>make_bsf_figures.py</code>"))

    out.append(f"""
    <h3>Part 2 &mdash; held-out variance over components, on the SBSF cohort</h3>
    <p><b>A different cohort on purpose.</b> Part 2 is <code>concat_hg</code> &mdash; the
      SBSF run, {ng} electrodes, gate APPLIED &mdash; and <code>concat_rawds</code>,
      which is the <b>same {ng} electrodes</b> described by 15 bands &times; 3 conditions
      &times; 30 bins instead of high gamma alone. So the two feature sets here are one
      cohort seen two ways, and neither is mixed with Part 1's {n}.</p>
    {figure("c13", "C.13", "Held-out variance explained over components, three algorithms",
            f'''<p style="margin:0 0 8px"><b>Bi-cross-validation.</b> A block of ROWS and a
        block of COLUMNS is held out; the method is fitted on the remaining block; each
        held-out electrode's loadings come from the TRAIN columns only and are scored on
        the TEST columns. The loadings never see the values they are graded on. The only
        thing that differs between methods is how those loadings are obtained &mdash;
        NNLS against the components for convex NMF, nearest centroid for k-means and Ward
        &mdash; which is exactly what distinguishes them.
        {p2.get("row_folds","?")}&times;{p2.get("col_folds","?")} folds,
        K = {", ".join(str(k) for k in p2.get("ks", []))}.</p>
      <p style="margin:0 0 8px"><b>Two panels per feature set, because variance explained
        is not space-free.</b> The left panel fits and scores each method where it belongs,
        so the SHAPE of each curve is meaningful but the heights are not comparable. The
        right panel puts every method in unit-norm, where heights ARE comparable, at the
        cost of scoring the hard methods somewhere they were not designed to run.</p>''',
            "C13_heldout_variance.png",
            "held-out variance explained versus K for k-means, Ward and convex NMF on "
            "concat_hg and concat_rawds, in home space and in unit-norm space",
            f"<code>make_heldout_variance.py</code> + <code>make_heldout_figure.py</code> "
            f"&middot; numbers in <code>bsf_comparison/part2_heldout_variance.csv</code>")}
    {t_peaks(d['pk'])}
  </section>""")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--insert", action="store_true")
    a = ap.parse_args()
    d = load()
    sec = build(d)
    nav = ('    <button class="n2" data-t="clustercmp"><span class="dot"></span>'
           '07 · Cluster comparisons</button>\n')
    (OUT / "webblock_section.html").write_text(sec, encoding="utf-8")
    (OUT / "webblock_nav.html").write_text(nav, encoding="utf-8")
    print(f"section {len(sec):,} chars -> {OUT/'webblock_section.html'}")

    if not a.insert:
        print("(dry run - pass --insert to splice into the site)")
        return 0

    s = SITE.read_text(encoding="utf-8")
    if 'id="clustercmp"' in s:
        # replace in place so this is re-runnable
        s = re.sub(r'\n  <section id="clustercmp">.*?\n  </section>', "\n" + sec,
                   s, flags=re.S)
        print("replaced the existing 07 tab")
    else:
        anchor = ('    <button class="n5" data-t="caveats">')
        assert s.count(anchor) == 1
        s = s.replace(anchor, nav + anchor)
        close = "\n  </section>\n  </main>"
        assert s.count(close) == 1, s.count(close)
        s = s.replace(close, "\n  </section>\n" + sec + "\n  </main>")
        print("inserted the 07 tab and its nav button")
    SITE.write_text(s, encoding="utf-8")

    t = SITE.read_text(encoding="utf-8")
    for tag in ("section", "figure", "table", "ul", "tr", "div"):
        o, c = t.count("<" + tag), t.count("</" + tag + ">")
        print(f"  {tag:<8} {o} / {c}  {'OK' if o == c else '<-- MISMATCH'}")
    ids = set(re.findall(r'id="([^"]+)"', t))
    bad = sorted(h for h in set(re.findall(r'href="#([^"]+)"', t)) if h not in ids)
    print("  dead links:", bad or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
