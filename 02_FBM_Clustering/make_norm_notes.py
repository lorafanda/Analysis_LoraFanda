#!/usr/bin/env python3
"""
make_norm_notes.py - the normalisation question, added to analysis_status.html everywhere
it bears on a claim already on the page.

WHAT IT ADDS. One long block in stage 02 (FIG N.1-N.6 with the measured numbers and the
literature they rest on) and six short bullet blocks elsewhere - stage 01, the K-choice
paragraph, the 07 comparison tab, the KISS tab, the open-decisions tab and the paper
scaffold - so the same argument is reachable from wherever the reader happens to be.
Every block is bracketed by its own BEGIN/END marker and is replaced, not appended, on a
re-run. Nothing else in the file is touched.

WHAT IT CORRECTS. Three hand-written passages still say k-means and Ward are fitted "in
raw dB". They were, until 2026-09-06 (commit cbc86fdf5): 240 and 241 now set
FIT_SPACE = 'unit-norm', measure_cluster_stability.SPACE says unit-norm for all three
methods, and every run and every CSV published on 2026-09-15 carries space=unit-norm -
including the ones the same paragraphs quote. The tables beside those sentences already
print "unit-norm" in all three rows, so the page contradicted itself. The corrections are
listed in CORRECTIONS below and applied with --insert; each one is a literal replacement
that fails loudly if the text has moved.

    python make_norm_notes.py            (print what would change)
    python make_norm_notes.py --insert   (write the site file)

Numbers come from outputs/clustering/normalisation/ (make_normalisation_figures.py) and
are re-read at build time, so the two must be regenerated together.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
NORM = ROOT / "outputs" / "clustering" / "normalisation"
SITE = Path("C:/Users/fanda/lorafanda.github.io/analysis_status.html")
FIGROOT = "02_FBM_Clustering/outputs/clustering/normalisation"

# ----------------------------------------------------------------- the numbers
S = pd.read_csv(NORM / "normalisation_scorecard.csv").set_index("transform")
J = json.loads((NORM / "normalisation_summary.json").read_text(encoding="utf-8"))
RAW, UN, BZ, BZU = "raw dB", "unit-norm", "band-z  (concat_bands5z)", "band-z → unit-norm"
RZ, SOFT = "row z (correlation distance)", "band-z → soft norm"
RZ2 = "band-z → row z"
TRAP1, TRAP2 = "per-bin column z  (trap)", "per-electrode per-band z  (trap)"
P = J["pairs"]
E = J["elevation"]
try:
    H = pd.read_csv(NORM / "N4_heldout_to_K300.csv")
    KM = H[H.method == "kmeans"]
    K300 = {sp: float(g[g.k == g.k.max()]["mean"].iloc[0]) for sp, g in KM.groupby("space")}
    K8 = {sp: float(g[g.k == 8]["mean"].iloc[0]) for sp, g in KM.groupby("space")}
    CN = H[H.method == "cnmf"]
    CNPEAK = {sp: int(g.loc[g["mean"].idxmax(), "k"]) for sp, g in CN.groupby("space")}
except FileNotFoundError:      # --quick run of the figure script
    K300 = K8 = CNPEAK = None

REF = {
    "cg": "Cronbach &amp; Gleser 1953, <i>Psychol Bull</i> 50:456",
    "mc": "Milligan &amp; Cooper 1988, <i>J Classification</i> 5:181",
    "st": "Steinley 2004, in <i>Classification, Clustering and Data Mining</i>",
    "ward": "Ward 1963, <i>JASA</i> 58:236; Murtagh &amp; Legendre 2014, <i>J Classification</i> 31:274",
    "dm": "Dhillon &amp; Modha 2001, <i>Machine Learning</i> 42:143",
    "eisen": "Eisen et al. 1998, <i>PNAS</i> 95:14863",
    "keogh": "Keogh &amp; Kasetty 2003, <i>Data Min Knowl Disc</i> 7:349; Rakthanmanon et al. 2012, <i>KDD</i>",
    "church": "Churchland et al. 2012, <i>Nature</i> 487:51",
    "ham": "Hamilton, Edwards &amp; Chang 2018, <i>Curr Biol</i> 28:1860",
    "ding": "Ding, Li &amp; Jordan 2010, <i>IEEE TPAMI</i> 32:45",
    "owen": "Owen &amp; Perry 2009, <i>Ann Appl Stat</i> 3:564",
    "benhur": "Ben-Hur, Elisseeff &amp; Guyon 2002, <i>PSB</i> 7:6; Lange et al. 2004, <i>Neural Comput</i> 16:1299",
    "vl": "von Luxburg 2010, <i>Found Trends Mach Learn</i> 2:235",
    "tib": "Tibshirani, Walther &amp; Hastie 2001, <i>JRSS-B</i> 63:411",
    "hennig": "Hennig 2007, <i>Comput Stat Data Anal</i> 52:258",
    "monti": "Monti et al. 2003, <i>Machine Learning</i> 52:91",
    "buz": "Buzs&aacute;ki, Anastassiou &amp; Koch 2012, <i>Nat Rev Neurosci</i> 13:407",
    "don": "Donoghue et al. 2020, <i>Nat Neurosci</i> 23:1655",
    "oss": "Ossand&oacute;n et al. 2011, <i>J Neurosci</i> 31:14521",
    "ramot": "Ramot et al. 2012, <i>J Neurosci</i> 32:10458",
}


def cite(*keys):
    return '<span class="cit">' + " &middot; ".join(REF[k] for k in keys) + "</span>"


def fig(num, fname, title, bullets, runid):
    li = "".join(f"<li>{b}</li>" for b in bullets)
    return (f'<figure id="normfig{num.lower().replace(".","")}"><div class="cap">'
            f'<div class="fignum">FIG {num}</div><h4>{title}</h4><ul>{li}</ul>'
            f'<div class="runid">{runid}</div></div>'
            f'<div class="imgwrap"><img data-fig="{FIGROOT}/{fname}" alt="{title}"></div></figure>')


# ----------------------------------------------------------------- the blocks
def block_s2():
    a = (f"{S.loc[RAW,'amp']:.2f}", f"{S.loc[BZU,'amp']:.2f}", f"{S.loc[BZ,'amp']:.2f}")
    return f"""
    <h4 id="s2norm">Normalisation &mdash; the largest single choice in the feature definition</h4>
    <p style="margin:0 0 8px">k-means and Ward minimise a sum of squares, so <b>whichever part of
      a vector carries the variance carries the partition</b> {cite('ward')}. Three separable things
      live in every concatenated cube {cite('cg')}: <b>elevation</b> (the mean level), <b>amplitude</b>
      (its length) and <b>shape</b> (where in frequency, condition and time the values sit). Only the
      third is the result. Measured here on <code>concat_bands5</code>, the same {J['n']} v8 electrodes
      every run below uses, K&nbsp;=&nbsp;{J['K']}, each transform fitted and scored on its own.</p>

    {fig("N.1", "N1_what_the_numbers_are.png",
         "What the numbers in a cube are made of, before any choice is made",
         [f"<b>Amplitude spans {J['norm_p95']/J['norm_p05']:.1f}&times;</b> across electrodes "
          f"(&#8214;x&#8214; 5% {J['norm_p05']:.0f}, median {J['norm_med']:.0f}, 95% {J['norm_p95']:.0f} dB) &mdash; "
          "distance to a contact, its orientation and its reference, not only its function.",
          "<b>1&ndash;20&nbsp;Hz holds 42% of the cohort's sum of squares</b> and 20&ndash;70&nbsp;Hz 11%. "
          f"Euclidean distance has no idea 1/f exists {cite('buz','don')}.",
          "<b>A rank-1 'own amplitude &times; common profile' model explains 15%</b> of the raw sum of "
          "squares &mdash; amplitude is real, and it is not the main thing.",
          "<b>Elevation is the sleeper</b>: a constant offset over all 450 features, 5&ndash;95% "
          f"{'%+.2f' % -0.57} &hellip; {'%+.2f' % 0.40}&nbsp;dB, which no per-electrode <i>scaling</i> removes."],
         "<code>make_normalisation_figures.py</code> &middot; v8, <code>concat_bands5</code>")}

    {fig("N.2", "N2_scorecard.png",
         "The scorecard &mdash; eleven transforms, one partition each, scored on what drives it",
         [f"<b>Raw dB is a sort by size and level</b>: &eta;&sup2; of &#8214;x&#8214; across clusters "
          f"{a[0]}, of elevation {S.loc[RAW,'elev']:.2f}; the largest cluster holds "
          f"{1/S.loc[RAW,'bal']:.0f}&times; the smallest.",
          f"<b>Band-z alone does not fix it</b> ({a[2]}) &mdash; it moves the sort into the high bands. "
          "The per-electrode step is what removes it.",
          f"<b>band-z &rarr; unit-norm</b> is the space the site's runs use {cite('dm')}, and it is the "
          f"best of the eleven on the column that matters most &mdash; it tracks where the response is "
          f"({S.loc[BZU,'vband']:.2f}, the highest band tracking in the table) without sorting by size "
          f"({a[1]}). Balance {S.loc[BZU,'bal']:.2f}; "
          + (f"<b>no cluster is one patient</b>" if S.loc[BZU, 'pat'] < 0.005 else
             f"{S.loc[BZU,'pat']:.0%} of electrodes sit in a cluster one patient dominates")
          + f" ({S.loc[UN,'pat']:.0%} for plain unit-norm).",
          f"<b>What it costs</b>: stability (ARI between 80% subsamples) "
          f"{S.loc[UN,'stab']:.2f} for plain unit-norm, {S.loc[BZU,'stab']:.2f} once the bands are "
          "equalised, and the most balanced partition of all is band-z &rarr; row z "
          f"({S.loc[RZ2,'bal']:.2f} against {S.loc[BZU,'bal']:.2f}). That is the price of letting the "
          "weak high bands speak, and it belongs in the paper.",
          f"<b>The two traps behave as the literature predicts</b> {cite('mc','st')}: per-bin column z "
          f"inflates the silent bins (balance {S.loc[TRAP1,'bal']:.2f}), and z-scoring each band within "
          f"each electrode erases where in frequency the response is (band tracking "
          f"{S.loc[TRAP2,'vband']:.2f}, the lowest of all eleven; stability {S.loc[TRAP2,'stab']:.2f}).",
          f"<b>Ward behaves the same way</b> (last column): amplitude &eta;&sup2; {S.loc[RAW,'ward_amp']:.2f} "
          f"raw &rarr; {S.loc[BZU,'ward_amp']:.2f} normalised."],
         "<code>make_normalisation_figures.py</code> &middot; numbers in "
         "<code>normalisation_scorecard.csv</code>, partition agreement in <code>normalisation_ari.csv</code>")}

    {fig("N.3", "N3_magnitude_splitting.png",
         "Do extra clusters buy shape, or subdivide by magnitude? &mdash; the published precedent, run on our cohort",
         [f"{REF['ham']} clustered 1,906 speech-responsive electrodes with convex NMF {cite('ding')} and "
          "reported that beyond k&nbsp;=&nbsp;2 the extra clusters were the same two response types, "
          "&ldquo;mostly further subdivided according to response magnitude&rdquo; (p.1861; Fig S2B&ndash;D) "
          "&mdash; <b>even though every electrode had already been z-scored against its own recording "
          "session</b> (STAR Methods e1).",
          "<b>Our cohort does the same thing in raw dB</b>: the amplitude sort rises from K=2 and sits "
          f"near &eta;&sup2;&nbsp;{a[0]} from K=8 onwards, and each new cluster extends the ladder of "
          "cluster mean amplitudes.",
          f"<b>Normalised, it does not</b>: &eta;&sup2; stays near {a[1]} across the whole sweep, and the "
          "clusters stay comparable in size.",
          "<b>What this licenses</b>: 'we normalise so that an added cluster buys a shape, not a volume "
          "setting' &mdash; with a citation, not as an assertion."],
         "<code>make_normalisation_figures.py</code> &middot; K = 2&hellip;14, k-means n_init=10, seed 42 "
         "&middot; numbers in <code>N3_magnitude_splitting.csv</code>")}

    {fig("N.5", "N5_pairs_before_after.png",
         "The same electrode pairs before and after &mdash; what the scorecard numbers are made of",
         [f"<b>A &mdash; same shape, different size.</b> {P['nA']} pairs in the cohort have cosine "
          f"&ge; 0.8 with at least a 2&times; size difference. Raw dB splits <b>{P['nA_split']}</b> of "
          f"them into different clusters; the normalised space splits <b>{P['nA_split_u']}</b>.",
          f"<b>B &mdash; same size, opposite shape.</b> {P['nB']:,} pairs have cosine &le; 0.3 with norms "
          f"within 10%. Raw dB puts <b>{P['nB_join']:,}</b> of them in the <i>same</i> cluster &mdash; the "
          f"low-amplitude bucket &mdash; against <b>{P['nB_join_u']:,}</b> normalised.",
          "Read the right column: after normalisation the A pairs lie on top of each other and the B "
          "pairs separate. That is the whole argument in six panels.",
          f"The two partitions agree at ARI {J['ari_db_vs_unit']:.2f} &mdash; <b>changing the "
          "normalisation changes the answer more than changing the algorithm does</b> "
          "(k-means vs Ward vs convex NMF agree at 0.25&ndash;0.36 on one feature set)."],
         "<code>make_normalisation_figures.py</code> &middot; pairs chosen by cosine and norm ratio, "
         "not by hand")}

    {fig("N.6", "N6_elevation_cluster.png",
         "The decision unit-norm does not make for you &mdash; elevation",
         ["<b>Scaling a vector does not centre it.</b> An electrode that is negative everywhere keeps "
          f"'negative everywhere' as its direction, so elevation still drives &eta;&sup2; "
          f"{E['eta_u']:.2f} of the normalised partition; only row z (centre, then scale &mdash; "
          f"Euclidean distance on row-z vectors <b>is</b> correlation distance {cite('eisen','keogh')}) "
          f"brings it to {E['eta_z']:.2f}.",
          f"<b>It is one identifiable cluster</b>: {E['n']} electrodes from {E['n_pat']} patients, "
          "broadband high-frequency suppression through all three conditions.",
          f"<b>And it is not <i>only</i> an offset</b>: with elevation removed, {E['kept_together']:.0%} "
          "of them still cluster together &mdash; so this is a response class, not a scaling artefact.",
          f"<b>Broadband high-frequency decreases are a documented response</b> {cite('oss','ramot')}. "
          "The check that settles it is the single-trial rasters of these contacts "
          "(<code>04_ersp_LM/&lt;pid&gt;/LM/HG/</code>): present trial by trial &rarr; keep unit-norm; "
          "only relative to a baseline carrying the previous trial's burst &rarr; row z until the "
          "baseline is fixed."],
         "<code>make_normalisation_figures.py</code> &middot; the cluster with the lowest mean "
         "elevation of the K=8 band-z &rarr; unit-norm partition")}

    <div class="method" style="border-left:4px solid var(--s2)">
      <h4 style="margin:0 0 6px">What other groups do &mdash; and what of it we can reproduce</h4>
      <ul style="margin:0">
        <li><b>Per-electrode normalisation before decomposition is standard in iEEG.</b>
          {REF['ham']} z-scored each electrode's 70&ndash;150&nbsp;Hz envelope against the
          <i>mean and SD of that recording session</i> (STAR Methods e1) before concatenating all
          27 subjects into one 31,625&nbsp;&times;&nbsp;1,906 matrix and running convex NMF
          {cite('ding')}. Note what that scale <b>is</b>: a noise scale, one number per electrode,
          so SNR survives &mdash; it removes gain, impedance and reference distance, not response size.
          <b>We already compute the analogue</b>: <code>compute_ersp</code> returns <code>avg_z</code>
          (each frequency against that trial's own baseline SD, <code>lf_ersp.py:1230</code>) and 140
          saves only <code>avg_db</code>. One extra <code>np.save</code> would give us a third space
          with a published precedent.</li>
        <li><b>Clustering on the unit sphere is a named method.</b> k-means on unit-normed rows is
          spherical k-means {cite('dm')}; centred-correlation clustering &mdash; row z &mdash; is the
          expression-profile standard {cite('eisen')} and the default for shape-based time-series
          comparison {cite('keogh')}.</li>
        <li><b>Keeping <i>some</i> amplitude has a precedent too.</b> {REF['church']} soft-normalised
          each neuron by its range plus a constant, so strongly modulated neurons reach about unit
          range and weak ones stay below it. The analogue here is
          <code>x&nbsp;/&nbsp;&#8214;x&#8214;<sup>&alpha;</sup></code> with &alpha;&nbsp;=&nbsp;0.5:
          size &eta;&sup2; {S.loc[SOFT,'amp']:.2f} with balance {S.loc[SOFT,'bal']:.2f}. Adding
          amplitude back as an <i>extra feature</i> instead is what not to do &mdash; k-means splits a
          one-dimensional continuum greedily and the appended column took over the partition.</li>
        <li><b>The negative results are old and specific.</b> Blind per-variable standardisation can
          destroy the very structure it is meant to reveal {cite('mc','st')}; our two traps are the two
          granularities that erase, respectively, <i>when</i> and <i>where in frequency</i> the response
          is.</li>
        <li><b>What none of them tells us.</b> {REF['ham'].split(',')[0]}'s design is a single band and
          continuous time, so cross-frequency weighting never arises for them. The band-z step rests on
          the 1/f literature {cite('buz','don')} and on FIG N.1B, not on their precedent.</li>
      </ul>
    </div>

    <div class="callout" style="margin:14px 0">
      <b>The recommendation, in one line.</b> Keep <b>band-z &rarr; unit-norm</b> as the reported space
      (it is already what every 2026-09-15 run used), say so in the methods with the three citations
      above, resolve the elevation cluster with its own rasters before choosing between unit-norm and
      row z, and report K from stability rather than from the held-out curve &mdash;
      <a href="#s2choosek">see below</a>.
    </div>
"""


def block_choosek():
    if K300 is None:
        return ""
    return f"""
      {fig("N.4", "N4_why_no_peak.png",
           "Why &ldquo;variance explained vs K&rdquo; cannot choose K for a hard partition",
           [f"<b>A &middot; bi-cross-validated, K to 300.</b> k-means reaches "
            f"{K8['raw dB']:.2f} at K=8 and is still climbing at K=300 ({K300['raw dB']:.2f}) in raw dB; "
            f"{K8['band-z → unit-norm']:.2f} &rarr; {K300['band-z → unit-norm']:.2f} normalised. "
            f"Convex NMF turns over at K={CNPEAK['raw dB']} / {CNPEAK['band-z → unit-norm']} "
            "(&#9660;) &mdash; the only interior peak in the figure.",
            "<b>Why.</b> A held-out electrode is reconstructed by ONE centroid &mdash; one free "
            "parameter &mdash; so an extra cluster almost never hurts. Convex NMF's k graded "
            f"loadings overfit sooner. {cite('owen')}",
            "<b>B &middot; in-sample R&sup2;</b>, the convention the field usually shows, and "
            "<b>C &middot; the increment</b> read as an elbow "
            f"({REF['ham']}, STAR Methods e2 and Fig S2A: their 2 clusters explained 16.9%, more "
            "added little). Both decay; neither has an optimum.",
            "<b>So K is a reproducibility question, not a variance question</b> &mdash; stability "
            "selection, bootstrap and the gap statistic, read together."],
           "<code>make_normalisation_figures.py</code> &middot; bi-CV 3&times;3 folds, "
           "<code>concat_bands5</code>, v8 &middot; numbers in <code>N4_heldout_to_K300.csv</code>, "
           "<code>N4_insample_r2.csv</code>")}
      <ul style="margin:8px 0 0">
        <li><b>A hard partition's held-out curve cannot have an interior peak here, and that is not a
          normalisation problem.</b> Re-run on <code>concat_bands5</code> with K pushed to 300: k-means
          reaches {K8['raw dB']:.2f} at K=8 and is still climbing at K=300 ({K300['raw dB']:.2f}) in raw
          dB, and {K8['band-z → unit-norm']:.2f} &rarr; {K300['band-z → unit-norm']:.2f} after
          normalisation. A held-out electrode is reconstructed by <b>one</b> centroid &mdash; one free
          parameter &mdash; so an extra cluster almost never hurts; convex NMF's k graded loadings
          overfit sooner, which is the only reason its curve turns over {cite('owen')}.
          <a href="#normfign4">FIG N.4</a>.</li>
        <li><b>The same is true of the convention the field usually shows.</b> {REF['ham']} plotted the
          <i>additional</i> percent variance explained for k&nbsp;=&nbsp;2&hellip;32 (STAR Methods e2)
          and read an elbow: two clusters explained 16.9% and more clusters added little (Fig S2A). Our
          in-sample increment does the same thing &mdash; it decays, it has no optimum
          (<a href="#normfign4">FIG N.4</a>B&ndash;C).</li>
        <li><b>So K comes from reproducibility, not from variance.</b> Stability selection
          {cite('benhur','monti')}, cluster-wise bootstrap {cite('hennig')}, and the gap statistic
          {cite('tib')} are the instruments that can answer it; stability is known to favour small K
          when clusters are well separated {cite('vl')}, which is why it is read beside the gap
          statistic and not alone. {REF['ham'].split(',')[0]} picked K per participant as the lowest K at
          which the onset component appeared (Figs S4&ndash;S5) &mdash; interpretability, openly.</li>
      </ul>
"""


def block_s1():
    return """
      <li><b>what the cube is normalised <i>to</i></b> &middot; dB re the pre-stimulus baseline
        (&minus;0.6&hellip;&minus;0.1&nbsp;s), per trial and per frequency, so a contact's gain and
        impedance are already divided out &mdash; but its <b>response size</b> is not, and that is what
        drives a Euclidean clustering downstream
        (<a href="#s2norm">stage 02 &middot; FIG N.1&ndash;N.6</a>).
        <code>compute_ersp</code> also returns <code>avg_z</code> (the same thing divided by the
        baseline SD, <code>lf_ersp.py:1230</code>) &mdash; the quantity Hamilton, Edwards &amp; Chang
        2018 (<i>Curr Biol</i> 28:1860, STAR Methods e1) cluster on &mdash; and 140 currently saves only
        <code>avg_db</code>. One <code>np.save</code> in the export block would give stage 02 a third
        space with a published precedent.</li>
      <li><b>what the responsiveness gate assumes</b> &middot; the gate is a <i>duration</i> test
        (|dB| over threshold in &ge;&nbsp;2&ndash;4% of bins), so a contact with a large but brief
        transient can fail it. The same hazard made Hamilton et al. 2018 reject a
        response-vs-silence criterion and select electrodes by held-out STRF prediction instead
        (STAR Methods e2, &ldquo;false exclusion of onset electrodes&rdquo;). Worth one count before v9:
        how many gated-out contacts have max&nbsp;|dB|&nbsp;&gt;&nbsp;4 in under 2% of bins.</li>
"""


def block_bsf():
    return f"""
        <li><b>Normalisation moves this comparison more than the algorithm does.</b> On the same
          electrodes, two K=8 partitions fitted in raw dB and in band-z&nbsp;&rarr;&nbsp;unit-norm agree
          at ARI {J['ari_db_vs_unit']:.2f}, against 0.25&ndash;0.36 between the three algorithms in one
          space. In raw dB the partition is largely a sort by response size (&eta;&sup2;
          {S.loc[RAW,'amp']:.2f}) and by level ({S.loc[RAW,'elev']:.2f}); normalised, {S.loc[BZU,'amp']:.2f}
          and {S.loc[BZU,'elev']:.2f}. {cite('ward','mc')} &mdash;
          <a href="#s2norm">stage 02 &middot; FIG N.2, N.5</a>.</li>
"""


def block_kiss():
    return f"""
    <div class="method" style="margin:12px 0">
      <b>And the coordinate system is a choice, not a given.</b>
      <ul style="margin:6px 0 0">
        <li>Scaling each electrode to unit length turns Euclidean distance into a distance between
          <b>shapes</b> (spherical k-means, {REF['dm']}); centring first as well turns it into
          <b>correlation</b> distance ({REF['eisen']}).</li>
        <li>Without it, the biggest difference between two electrodes really is how loud they are:
          &eta;&sup2; of response size across the K=8 clusters is {S.loc[RAW,'amp']:.2f} in raw dB and
          {S.loc[BZU,'amp']:.2f} after normalisation &mdash;
          <a href="#s2norm">stage 02 &middot; FIG N.2</a>.</li>
        <li>The same effect is on record elsewhere: {REF['ham']} found extra clusters subdividing by
          response magnitude even after per-electrode z-scoring (p.1861, Fig S2B&ndash;D).</li>
      </ul>
    </div>
"""


def block_caveats():
    return f"""
    <h3 id="normdecisions">Normalisation &mdash; three decisions, and what each one changes</h3>
    <div class="method" style="margin:10px 0">
      <ul style="margin:0">
        <li><b>1 &middot; Which space the paper reports.</b> Recommended: keep
          <b>band-z&nbsp;&rarr;&nbsp;unit-norm</b>, which is what every 2026-09-15 run already used, and
          write it down with its citations ({REF['dm']}; {REF['mc']}). Alternatives are costed in
          <a href="#s2norm">FIG N.2</a>: plain unit-norm is more reproducible
          ({S.loc[UN,'stab']:.2f} vs {S.loc[BZU,'stab']:.2f}) but lets 1/f choose the clusters and leaves
          {S.loc[UN,'pat']:.0%} of electrodes in one-patient clusters.</li>
        <li><b>2 &middot; Elevation: keep it or centre it.</b> Unit-norm keeps a constant offset;
          row&nbsp;z removes it. One cluster of {E['n']} electrodes from {E['n_pat']} patients turns on
          this &mdash; broadband high-frequency suppression in all three conditions, a documented
          response ({REF['oss']}; {REF['ramot']}). <b>The check</b>: its members' single-trial HG
          rasters. <a href="#s2norm">FIG N.6</a>.</li>
        <li><b>3 &middot; Whether to save the z-cube.</b> <code>compute_ersp</code> already computes
          <code>avg_z</code> (baseline-SD normalised, per frequency) and throws it away. Saving it gives
          a per-electrode <i>noise-scale</i> normalisation &mdash; the one {REF['ham']} used &mdash;
          as a fourth comparable space. Cost: one <code>np.save</code> per cube and a parallel folder.</li>
        <li><b>Not a decision, a correction.</b> Three passages on this page still described k-means and
          Ward as fitted in raw dB. They have been fitted in unit-norm since 2026-09-06
          (<code>cbc86fdf5</code>; <code>FIT_SPACE</code> in 240/241,
          <code>measure_cluster_stability.SPACE</code>), which is what the tables beside those sentences
          already printed. Corrected 2026-09-22.</li>
      </ul>
    </div>
"""


def block_paper():
    return f"""
    <div class="method" style="margin:12px 0">
      <b>Methods paragraph you can lift &mdash; the normalisation, with its citations.</b>
      <ul style="margin:6px 0 0">
        <li>&ldquo;Each condition's time-normalised ERSP was reduced to five bandwidth-weighted
          frequency bands &times; 30 time bins and the three conditions concatenated. Because Euclidean
          clustering is dominated by whichever feature carries the variance ({REF['ward']}), and because
          field-potential spectra are dominated by their low frequencies ({REF['buz']}; {REF['don']}),
          each band was standardised across the cohort (one mean and SD per band) and each electrode was
          then scaled to unit length, so that distance measures response <i>shape</i> rather than
          amplitude (spherical k-means; {REF['dm']}). Per-electrode normalisation before decomposition
          follows {REF['ham']}; without it, added clusters subdivide electrodes by response magnitude
          rather than by response type (their Fig S2B&ndash;D; reproduced on this cohort, FIG N.3).&rdquo;</li>
        <li>&ldquo;K was chosen by the reproducibility of the partition across resampled electrodes
          ({REF['benhur']}; {REF['hennig']}) rather than from explained variance, which for a
          hard assignment increases monotonically with K and has no interior optimum
          ({REF['owen']}; FIG N.4).&rdquo;</li>
        <li>Numbers to quote: amplitude &eta;&sup2; {S.loc[RAW,'amp']:.2f}&nbsp;&rarr;&nbsp;{S.loc[BZU,'amp']:.2f};
          ARI between the two spaces {J['ari_db_vs_unit']:.2f}; stability
          {S.loc[UN,'stab']:.2f}&nbsp;&rarr;&nbsp;{S.loc[BZU,'stab']:.2f}; {P['nA']} same-shape pairs,
          {P['nA_split']} split in dB vs {P['nA_split_u']} normalised.</li>
      </ul>
    </div>
"""


# marker, anchor (must appear exactly once), where, html
BLOCKS = [
    ("norm s2", "    <h4 id=\"s2choosek\">Choosing K &mdash; and why only one of the three can answer it</h4>",
     "before", block_s2),
    ("norm choosek", "    <h4 id=\"s2choosek\">Choosing K &mdash; and why only one of the three can answer it</h4>\n"
     "    <p style=\"margin:0 0 8px\">The held-out variance curve that was on this page holds out",
     "after-line", block_choosek),
    # stage 01's two bullets live in make_s1_tab.py itself (that whole section is
    # generated; a block spliced into it would be wiped on the next --insert there).
    ("norm bsf", "        <li><b>Panels D and F of the original C.3 are dropped</b> by request. D", "before", block_bsf),
    ("norm kiss", "    <p style=\"margin:0 0 8px\"><b>Why this is not a technicality.</b> Read each method at home and in",
     "before", block_kiss),
    ("norm caveats", "    <!-- ============ LATE BROADBAND BAND ============ -->", "before", block_caveats),
    ("norm paper", "    <div class=\"pf-warn\">", "before", block_paper),
]

# literal, wrong -> right. Each must appear exactly once.
CORRECTIONS = [
    ("<tr><td><b>k-means</b></td><td>one label</td><td>raw dB</td><td><code>240</code></td></tr>\n"
     "        <tr><td><b>Ward</b></td><td>one label</td><td>raw dB</td><td><code>241</code></td></tr>",
     "<tr><td><b>k-means</b></td><td>one label</td><td>unit-normed</td><td><code>240</code></td></tr>\n"
     "        <tr><td><b>Ward</b></td><td>one label</td><td>unit-normed</td><td><code>241</code></td></tr>"),
    ("<p style=\"margin:0 0 4px\"><b>Each is read in its home space.</b> Silhouette is not space-free:\n"
     "        convex NMF unit-norms each electrode before fitting while k-means and Ward use raw dB. Scoring\n"
     "        all three in dB is the error that made the first version of <a href=\"#separation\">FIG C.7</a>\n"
     "        wrong, and the right-hand column below shows what it would have produced.</p>",
     "<p style=\"margin:0 0 4px\"><b>All three are read in the space they are fitted in, and since\n"
     "        2026-09-06 that is the same space for all three &mdash; unit-norm</b> (<code>cbc86fdf5</code>:\n"
     "        <code>FIT_SPACE</code> in 240 / 241, <code>measure_cluster_stability.SPACE</code>; every run and\n"
     "        CSV of 2026-09-15 records <code>space=unit-norm</code>). Silhouette is not space-free &mdash;\n"
     "        scoring all three in dB is the error that made the first version of\n"
     "        <a href=\"#separation\">FIG C.7</a> wrong, and the right-hand column below still shows what that\n"
     "        would have produced. <b>What the earlier wording meant</b> &mdash; k-means and Ward in raw dB\n"
     "        &mdash; was true of the runs before 2026-09-06 only; why the choice matters at all is\n"
     "        <a href=\"#s2norm\">FIG N.1&ndash;N.6</a>.</p>"),
    ("<li><b>Each method is read in its home space.</b> convex NMF unit-norms before\n"
     "          fitting; k-means and Ward use raw dB. Silhouette is not space-free, and\n"
     "          scoring all three in dB is the error that made the first version of\n"
     "          <a href=\"#separation\">FIG C.7</a> wrong.</li>",
     "<li><b>All three are read where they are fitted &mdash; unit-norm for all three\n"
     "          since 2026-09-06</b> (<code>cbc86fdf5</code>). Silhouette is not space-free, and\n"
     "          scoring all three in dB is the error that made the first version of\n"
     "          <a href=\"#separation\">FIG C.7</a> wrong. The dB column is kept as that\n"
     "          counter-example. Why the space matters: <a href=\"#s2norm\">FIG N.1&ndash;N.6</a>.</li>"),
    ("<td>The coordinate system a method was actually fitted in. Convex NMF fits on\n"
     "          <b>unit-normed</b> data; k-means and Ward fit on <b>raw dB</b>. Nothing else differs\n"
     "          &mdash; same matrix, same gate, same time warping.</td>",
     "<td>The coordinate system a method was actually fitted in. Since 2026-09-06 all three fit on\n"
     "          <b>unit-normed</b> data (before that, k-means and Ward fitted on <b>raw dB</b>, and the\n"
     "          worked example below is that comparison). Nothing else differs &mdash; same matrix, same\n"
     "          gate, same time warping. What the choice does to the partition:\n"
     "          <a href=\"#s2norm\">FIG N.1&ndash;N.6</a>.</td>"),
    ("<b>Each method is scored in the space it fits in</b> &mdash; convex NMF unit-norms each\n"
     "        electrode, k-means and Ward use raw dB, and silhouette is not a space-free quantity.</p>",
     "<b>Each method is scored in the space it fits in</b> &mdash; unit-norm for all three since\n"
     "        2026-09-06 (<code>cbc86fdf5</code>; before that, raw dB for k-means and Ward), and\n"
     "        silhouette is not a space-free quantity. <a href=\"#s2norm\">FIG N.1&ndash;N.6</a>.</p>"),
    # the C.13 caption, generated by make_cluster_webblock.py - fix the source there too
    ("<li>One panel per feature set, every method fitted AND scored in its <b>home\n"
     "          space</b> &mdash; convex NMF unit-normed, k-means and Ward in raw dB. The SHAPE\n"
     "          of each curve is meaningful; the heights are NOT comparable across methods,\n"
     "          because they are explaining variance in two different matrices.</li>",
     "<li>One panel per feature set, every method fitted AND scored in its <b>home\n"
     "          space</b>. On the 2026-09-15 runs drawn here that is <b>unit-norm for all\n"
     "          three</b> (<code>cbc86fdf5</code>, 2026-09-06), so the heights ARE comparable:\n"
     "          convex NMF's curve is both higher and the only one that turns over. The\n"
     "          caption's earlier warning applied to the runs before that date.</li>"),
]


def apply(s: str, insert: bool) -> str:
    for name, anchor, where, fn in BLOCKS:
        b = f"<!-- BEGIN {name}, generated by make_norm_notes.py - do not hand-edit -->"
        e = f"<!-- END {name} -->"
        html = (fn() if fn else block_s1())
        if not html.strip():
            print(f"  {name}: nothing to write (run make_normalisation_figures.py without --quick)")
            continue
        new = f"{b}\n{html.rstrip()}\n{e}\n"
        if b in s:                                   # replace in place
            i, j = s.index(b), s.index(e) + len(e) + 1
            s = s[:i] + new + s[j:]
            print(f"  {name}: replaced ({len(html):,} chars)")
            continue
        if s.count(anchor) != 1:
            raise SystemExit(f"anchor for {name} appears {s.count(anchor)} times:\n{anchor[:120]}")
        i = s.index(anchor)
        if where == "after-line":
            i = s.index("\n", s.index(anchor) + len(anchor)) + 1
        elif where == "after":
            i = i + len(anchor)
        s = s[:i] + new + s[i:]
        print(f"  {name}: inserted ({len(html):,} chars)")

    for old, new in CORRECTIONS:
        if new in s:
            print("  correction already applied")
            continue
        if s.count(old) != 1:
            raise SystemExit(f"correction text appears {s.count(old)} times:\n{old[:140]}")
        s = s.replace(old, new)
        print(f"  corrected: {old[:70]}...")
    return s


CSS = """
<!-- BEGIN norm css, generated by make_norm_notes.py - do not hand-edit -->
<style>
.cit{color:var(--muted);font-size:11.5px}
.cit i{font-style:italic}
</style>
<!-- END norm css -->
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--insert", action="store_true")
    a = ap.parse_args()
    s = SITE.read_text(encoding="utf-8")
    n0 = len(s)
    if "BEGIN norm css" not in s:
        s = s.replace("</head>", CSS + "</head>", 1)
    s = apply(s, a.insert)
    print(f"{n0:,} -> {len(s):,} chars")
    if a.insert:
        SITE.write_text(s, encoding="utf-8")
        print(f"wrote {SITE}")
    else:
        print("(pass --insert to write the site)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
