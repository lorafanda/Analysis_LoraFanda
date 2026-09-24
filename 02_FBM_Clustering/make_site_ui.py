#!/usr/bin/env python3
"""
make_site_ui.py - the reading layer of analysis_status.html: grouped navigation with the
retired tabs at the bottom, sub-tabs inside every tab, collapsible sections, a figure
lightbox, heading search, a phone layout, print styling.

    python make_site_ui.py            write the blocks to stdout-size report, change nothing
    python make_site_ui.py --insert   splice them into the site (idempotent)

Everything is a runtime layer: one CSS block in <head> and one script at the end of
<body>, both between BEGIN/END markers, plus the nav rebuilt into groups. No section's
content is touched, so the generators (make_*_tab.py, make_*_webblock.py,
retire_sections.py, make_site_gate.py) keep working: their BEGIN/END markers and the
`<button class="n5" data-t="caveats">` anchor survive inside the grouped nav, and the
sub-tabs / collapsibles are built in the browser from each section's own <h3>s
(or its .nb-day / .mtg cards), never written into the file.

Tab order and groups are the GROUPS table below; retired buttons (class "retired") go
under a collapsed "Retired" header whatever their position was; a button not listed
in GROUPS lands in "Other" before Retired.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

SITE = Path("C:/Users/fanda/lorafanda.github.io/analysis_status.html")

GROUPS = [   # (header or None, [data-t ...]) in display order
    (None, ["overview"]),
    ("Pipeline", ["s1", "s2", "lana"]),
    ("Side studies", ["mm", "microepi"]),
    ("Writing", ["paper", "outline", "kiss"]),
    ("Log &amp; plans", ["notebook", "meetings", "caveats"]),
]

# ---------------------------------------------------------------------------------------
# THE REPORT SHAPE. Every analysis tab reads Answer -> Evidence -> Method -> Open -> History.
# ANSWERS carries the tab's answer (written here, injected at build); ROLES maps each of the
# tab's groups (the runtime keys: an h3's id or the slug of its text, "intro" for what precedes
# the first heading) to a role; a group with no entry lands in History. Tabs without an entry
# keep the plain strip (KISS, the notebook, the Paper 2 tab). The Overview's answer is the
# list of every tab's one-liner, built at runtime from ANSWERS in nav order.
# ---------------------------------------------------------------------------------------
ANSWERS = {
    "overview": dict(
        one="the cohort and where every tab stands",
        html=("<p>Cohort <b>v8</b>: 1680 gated electrodes from 27 patients (2026-09-15). PAT_6684 (G-05) has since been "
              "removed from the dataset; <b>v9</b> is built once the MicroEPI reruns and EL051 are on disk. One line per tab, "
              "the tab's own Answer behind each:</p><!--TABLIST-->"),
        status="v8 on every stage-02 product · the five S-figures and stages 03/04 are the 1027-electrode fit"),
    "outline": dict(
        one="the paper as bullets, preprocessing → clustering outputs, darkness = confidence; source paper2_outline.md (make_outline_tab.py)",
        html=("<p>Every section in Cell Reports order as bullet points, each line shaded by how sure it is (black = read from the code or a "
              "run, lightest = a slot for a citation's section / paragraph, a number or a decision). Edit "
              "<code>02_FBM_Clustering/paper2_outline.md</code> and re-run <code>make_outline_tab.py --insert</code>.</p>"),
        status="built 2026-09-24 for the lab presentation; numbers from the v9 cache and the 2026-09-24 audit"),
    "s1": dict(
        one="one ERSP cube per electrode × condition: 103 × 300 (0–400 Hz, warped), WM-referenced, notched, trial-filtered; halves beside; 31 of 31 patients ok (run 2026-09-18, reruns 09-21 / 22)",
        html=("<ul>"
              "<li><b>product</b>: cube 103 freq × 300 warped bins, dB re baseline (−0.4, −0.1 s), GO at bin 150 · odd / even halves beside every cube</li>"
              "<li><b>chain</b>: load → channels out (aux, Unknown, bad) → WM reference (grid / whole CAR where no WM) → notch per block (z ≥ 3; iir or spectrum interpolation) → trials (accuracy, stimulus ≥ 0.5 s, post 1–10 s, IQR 1.5) → STFT 1 kHz, bins ≤ 400 Hz → warp → mean</li>"
              "<li><b>run</b>: 2026-09-18, <code>140_ersp_pipeline.py</code> per patient, 31 patients, fmax 400 Hz (was 500) · audit table + review mode per contact</li>"
              "<li><b>done since</b>: EL043 on its two recordings joined (3 conditions), G-04 / G-06 with the reference without bad contacts, PAT_6953; no stale cube in the tree · <b>open</b>: 137 contacts without anatomy link (alias decision), QC pass in review mode</li>"
              "</ul>"),
        status="04_ersp_LM_RAWONLY = 2026-09-18 run (fmax 400) + reruns EL043 / PAT_6704 / PAT_6854 (09-22), PAT_6953 (09-21) · 31 / 31 ok · previous tree = _old"),
    "s2": dict(
        one="no reproducible hard partition; a graded convex-NMF description; the representation matters more than the algorithm",
        html=("<p>Three algorithms (k-means, Ward, convex NMF) on four feature sets of the same 1680 electrodes, K = 5…30. "
              "K is read where convex NMF's bi-cross-validated curve peaks (v8: 9 / 12 / 13 / 14, v9: 10 / 13 / 13 / 14 for HFA / 15 bands / 5 bands /"
              "5 bands z) because it is the only method whose curve turns over; the cross-method figures are cut at K = 8. "
              "The durable result is negative, then graded: hard partitions of these data are not reproducible across "
              "preprocessing, most electrodes have no majority component, and two algorithms on one feature set agree on "
              "~60% of electrodes where two feature sets under one algorithm agree on ~40% (chance 12%). The graded "
              "decomposition is the analysis; an argmax map is only ever shown beside its loadings.</p>"
              "<ul><li>Still v7 renders: FIG 2 on 5 bands z, the K=8 FIG 1 slides, the weighted FIG 4, the native stability sweep.</li>"
              "<li>K=8 (largest K with no one-patient cluster on HFA) or the held-out peaks — which one the paper reports.</li>"
              "<li>v9 once the MicroEPI reruns land.</li></ul>"),
        status="v8 · twelve runs 20260914–15 · statistics at the peaks · Paper 2 figures on v8"),
    "lana": dict(
        one="language-network proximity predicts more HFA while hearing the prompt and while speaking, less during visual encoding",
        html=("<p>Against LanA (an 806-subject probabilistic atlas of the language network), being closer to the network "
              "predicts more high-frequency activity while the spoken prompt is heard and during production in all three "
              "conditions, and less while a visual stimulus is on screen — a within-trial dissociation a BOLD atlas cannot "
              "express. Effects are small (|ρ| ≤ 0.16) and clear FDR because n = 2644; the map's structure is the result, not "
              "the coefficients. A hard in/out split depends on the threshold (86% in at P ≥ 0.05, 39% at P ≥ 0.10), which is "
              "the argument for the continuous map.</p>"
              "<ul><li>Per-parcel maps (IFG, IFGorb, MFG, AntTemp, PostTemp, AngG): six more volumes through the same function.</li>"
              "<li>Restricting the atlas to typically-lateralised subjects would raise the correlation ceiling.</li>"
              "<li>Not yet on v8 — the maps are on the 2644 / 2724-contact set of August.</li></ul>"),
        status="correlation maps 2026-08-02 · membership figures on the corrected coordinates · pre-v8"),
    "meetings": dict(
        one="the through-line and four logged meetings, June–July 2026",
        html=("<p>Can we find sites that represent language rather than stimulus or response processing? Clustering said no "
              "(weak, condition-driven structure); classifying networks works but sidesteps the question; the atlas route (3a) "
              "gave the first positive answer — a time–frequency signature of language-network proximity. Four meetings are "
              "logged below with the status of every decision. Nothing after 2026-07-28 is recorded here; the lab notebook "
              "carries it.</p>"
              "<ul><li>3b — the electrophysiological ROI — was never defined.</li>"
              "<li>Either log meetings after July here, or retire this tab into the notebook.</li></ul>"),
        status="last entry 2026-07-28"),
    "mm": dict(
        one="the speech-time broadband burst on G-05's contacts is muscle and the hardware reference, not cortex",
        html=("<p>Mouth movement produces a broadband high-frequency increase on G-05's macro contacts that starts after the cue "
              "with a reaction time, lasts as long as the movement, is tissue-blind and strongest at the lateral temporal skull "
              "entries: muscle, volume-conducted, not a cortical mouth-motor response. It is absent from the microwires and "
              "from an unconnected input. In the language task the same burst rides on every white-matter reference contact "
              "at the end of each spoken answer (+6 dB, gain 1.00), so it is the Micromed hardware reference: the WM "
              "re-reference halves it, a bipolar pair removes it.</p>"
              "<ul><li>Does the LM production map follow the mouth map across contacts?</li>"
              "<li>G-01, G-02, G-03 on the same protocol. (G-05 is out of the dataset; this stays as the mechanism.)</li></ul>"),
        status="report of 2026-09-13/14 · G-05 only"),
    "microepi": dict(
        one="references audited on all five; timing verified on G-01, G-04, G-06, measured on G-02 (+6 ms, 14 trials off); G-05 out",
        html=("<p>Of the six micro–macro patients: G-01, G-04 and G-06 are time-aligned to a few ms and their references are "
              "audited (the bad-channel lists are in config); G-02's photodiode sits +6 ms late with a 0–16 ms sawtooth and "
              "14 of 157 trials are 20–60 ms off; G-03's timing cannot be checked from its export; G-05 is out (seizure). "
              "Nothing has been re-run yet.</p>"
              "<ul><li>140 on G-01 / G-04 / G-06 with the new lists; a decision on G-02's 14 trials; G-03's raw files.</li>"
              "<li>IAD2–5, IAD12 and IMG1 are still in G-06's reference as configured.</li></ul>"),
        status="audits 2026-09-15/16 · reruns pending"),
    "caveats": dict(
        one="what is open — the reruns, G-02's 14 trials, G-03's raw check, one cohort-changing regex, and two paper decisions",
        html=("<p>Open, in order: the MicroEPI reruns and EL051, then cohort v9; G-02's 14 mis-timed trials — correct, drop or "
              "leave; G-03's raw timing check; <code>lf_dataset.py:113</code>, which drops every shaft whose name ends in M "
              "(PAT_3415's TM strip) — one regex, cohort-changing; notebook 150 and <code>build_timing_table.py</code> still key "
              "on G-05; the editorial choice between opening A and B (FIG C.4a/b) and whether HFA stays the headline; pooling "
              "(460/465) five rebuilds behind v8, so S1's role side and cluster side come from different samples.</p>"
              "<p>Everything under a <i>Done</i> heading below is duplicated in the lab notebook and is proposed for deletion.</p>"),
        status="open list re-read 2026-09-17"),
}

# group key -> role; keys are the h3 id when it has one, else the slug of its text (the runtime's rule)
_H = "history"; _E = "evidence"; _M = "method"; _O = "open"; _A = "answer"
ROLES = {
    "overview": {"intro": _H, "the-scientific-arc": _H, "headline-figures-one-per-argument": _H, "how-it-was-built": _M,
                 "how-the-stages-relate-compare": _H, "status-at-a-glance": _E},
    "s1": {"intro": _M, "s1-status": _E, "s1-chain": _M, "s1-params": _M, "s1-outputs": _M, "s1-checks": _E,
           "s1-open": _O, "s1-history": _H},
    "s2": {"intro": _M, "paper2figs": _E, "s2gallery": _E, "rtcompare": _H, "method": _H, "results-hfa-k-means-k-9": _H,
           "concatenated-clustering-a-second-sample-unit": _H, "response-timing-when-does-each-cluster-come-on": _H,
           "across-the-k-sweep-and-why-the-obvious-version-o": _H, "supporting-checks-reproducibility-k-choice-and-m": _H,
           "cross-correlation-do-the-clusters-lead-and-lag-e": _H, "cvmethods": _M, "gradedvspartition": _M},
    "lana": {"intro": _A, "what-the-atlas-is": _M, "what-was-done-step-by-step": _M, "how-to-read-these-figures": _M, "results": _H,
             "rebuilt-on-the-corrected-coordinates-two-cohorts": _E, "3-a-2-how-this-one-actually-works": _E,
             "how-this-relates-to-the-lana-fedorenko-work": _E, "honest-caveats": _O, "next-go-deeper": _O, "where-it-lives": _M},
    "meetings": {},          # every meeting card is History; the answer carries the through-line
    "mm": {"intro": _A, "where-things-stand": _A, "1-getting-the-trials": _M, "2-the-first-macro-analysis-and-why-i-stopped-tru": _H,
           "3-testing-the-timing-two-problems-both-fixed": _M, "4-the-macro-contacts-correctly-aligned": _E, "5-white-matter-contacts-as-data": _E,
           "6-the-micro-contacts-and-the-analog-inputs-as-co": _E, "7-what-i-think-it-means": _A},
    "microepi": {"intro": _A, "status-now-against-cohort-v8-2026-09-15": _A, "how-to-read-the-checks": _M,
                 "g-01-pat-5515-clean-aligned": _E, "g-02-pat-5533-clean-reference-a-6-ms-placement-w": _E,
                 "g-03-pat-6619-audited-timing-not-yet-checkable": _E, "g-04-pat-6704-one-noise-contact-in-the-reference": _E,
                 "g-05-pat-6684-removed": _H, "g-06-pat-6854-a-noisy-shaft-timing-aligned": _E},
    "caveats": {"intro": _H, "latebroadband": _H, "g05loose": _H, "clusterdiag": _H, "clusterplan": _H, "real-time-05-open-items": _O,
                "decisions-only-you-can-make": _O, "still-to-run": _O, "done-2026-09-08-09": _H, "done-2026-08-01-evening": _H,
                "done-2026-07-31-08-01": _H, "done-2026-07-19": _H, "recently-fixed": _H},
}

# PROPOSED DELETIONS. (section, group key, text prefix or None for the whole group, why). Struck through
# on the page, nothing removed; each block keeps a note with the reason. Prefix = the start of the
# element's text, whitespace-normalised, matched case-insensitively against the group's direct children.
DEL = [
    # the 41 blocks proposed on 2026-09-17 were deleted for real by site_delete_blocks.py the same
    # day (their HTML is in site_delete_blocks.removed.txt and in git); add entries here to propose more
]

# CORNELL. Per notebook day: (date as printed, first words of its tag) -> (cue, one-line summary).
CORNELL = {
    ("2026-09-12 → 15", "G-05 on its TRC"):        ("G-05 from the TRC; v8 fitted", "G-05 re-processed from its Micromed TRC with photodiode times through the raw clock fit; cohort v8 built and every stage-02 product rerun on it — and G-05 has since been removed."),
    ("2026-09-10", "cohort v7"):                    ("v7 cohort; repo cleanup; native stability", "Trial rejection and the HG-trials figure fixed in 140, v7 built into a new cache, 866 MB of stale figures untracked, and the convex-NMF stability found understated by half."),
    ("2026-09-07", "the cluster visualizer"):       ("visualizer as a comparison tool", "The cluster visualizer became a feature-set × algorithm matrix with overlay and grid views; 291 undrawn electrodes recovered by rebuilding the bundle."),
    ("2026-08-21", "a bootstrap ceiling"):          ("bootstrap ceiling 0.632; C.11; half-cubes", "Per-cluster stability had been capped at 0.632 by an in-bag bootstrap and reversed once corrected; component anatomy tested per voxel; odd/even half-cubes added to stage 01."),
    ("2026-08-19", "interpreting the decomposition"): ("own-space separation; gate lifted", "FIG C.7 corrected (each method separates only in its own space), the responsiveness gate made a testable feature set, rasters added beside centroids."),
    ("2026-08-18", "the 27-patient cohort"):        ("27-patient cohort; centroids publish", "Six concat runs on the 27-patient / 1266-electrode cohort, K=7 defended on concat_hg only, every centroid published, PAT_6953 unblocked with a manual WM reference."),
    ("2026-08-16", "silent data failures"):         ("silent data failures; aux leak", "G-02/G-03's flat 2 s responses traced to a misspelled log column and repaired; auxiliary channels found ERSP'd as cortex in both trees and purged; 464 site claims re-audited."),
    ("2026-08-14", "real-time re-run"):             ("HUG id bug; RT re-run", "The HUG patients' failure was two id-spelling bugs; the real-time re-run reached 21 of 30 and the ERSP_clean renders were made readable."),
    ("2026-08-11", "real time, GO-locked"):         ("notebook 150; four fixes", "GO-locked pipeline (150) written; its checks found MicroEPI trials double-counted and a condition-name bug that also reach 140."),
    ("2026-08-08", "K=7 published"):                ("K=7 published; onsets NaN; xcorr fixed", "Fabricated onsets became NaN, K=7 published as a run, 18 stale concat runs deleted, the cross-correlation estimator made polarity-aware after three false lags."),
    ("2026-08-07", "visualizer, diagnostics"):      ("four claims withdrawn; pivot to cNMF", "Controls withdrew four claims about the K=5 partition and surfaced three published defects; the pivot to convex NMF recorded with citations."),
    ("2026-08-03", "re-run + story figures"):       ("1027-electrode re-run; S1–S5", "Full re-run on 1027 electrodes (K 4→5), the auditory reference re-chosen by profile, story figures S1–S5 built."),
    ("2026-08-03", "a third unnormalised join"):    ("anatomy join normalised", "Anatomy labels were 45% 'unknown' from an unnormalised join — the third instance of that bug class; purity now reported against chance."),
    ("2026-08-02", "clean cohort + lead/lag"):      ("grid at contact level; lead/lag", "PAT_3415's grid excluded at contact level, the site repointed at the clean run, the lead/lag analysis (215) added with its bootstrap CI."),
    ("2026-08-02", "cluster timing"):               ("cluster onsets; the across-K confound", "Cluster onset ladders built with the pooling definition, split into stimulus and response windows; the across-K figure rebuilt around the cluster-size confound."),
    ("2026-08-02", "atlas + K selection"):          ("K on four criteria; LanA as runs", "Every K scored on four criteria; the LanA splits registered as clustering runs so the glassbrain renders them; inside-out laterals fixed."),
    ("2026-08-01", "recon payoff"):                 ("recon for 23/25 runs; pooling gap closed", "252 run on the fixed coordinates, phantom index entries pruned, 460's export gap closed."),
    ("2026-07-31", "the recon fix"):                ("recon gap = naming", "126 unplottable contacts were a homoglyph and four naming conventions, not missing localisation; audit_coverage.py written."),
    ("2026-07-29", "atlas as a run"):               ("atlas as a run", "The LanA in/out split registered as a run with a Desikan-Killiany breakdown."),
    ("2026-07-28", "language atlas"):               ("language atlas; concat wins", "The Fedorenko/LanA analysis built (sweep, correlation maps, split); concatenated clustering shown cleaner than per-condition."),
    ("2026-07-19", "full refresh"):                 ("full refresh", "Clustering, decoding and pooling all re-run; the status site built."),
    ("2026-07-06", ""):                             ("HFA beats the full band", "Paired per-patient test: HFA decodes better than the full spectrum (0.596 vs 0.535)."),
    ("2026-07-03", ""):                             ("centroid writer", "One centroid writer per K, with SEM shading."),
    ("2026-07-01", ""):                             ("knife plot; poolv2 gradient", "Silhouette knife plot and PCA projection; poolv2's dB gradient wired onto electrodes."),
    ("2026-06-30", "convergence"):                  ("poolv2 convergence", "poolv2 and the notebook made bit-identical; the premotor-planning role restored."),
}

ROLE_ORDER = ["answer", "evidence", "method", "open", "history"]
ROLE_LABEL = {"answer": "Answer", "evidence": "Evidence", "method": "Method", "open": "Open", "history": "History"}


def ui_data() -> str:
    import json
    return json.dumps({
        "answers": ANSWERS, "roles": ROLES, "roleOrder": ROLE_ORDER, "roleLabel": ROLE_LABEL,
        "del": [dict(sec=s, key=k, prefix=p, why=w) for s, k, p, w in DEL],
        "cornell": [dict(date=d, tag=t, cue=c, sum=m) for (d, t), (c, m) in CORNELL.items()],
    }, ensure_ascii=False)


B_NAV, E_NAV = "<!-- BEGIN ui nav, generated by make_site_ui.py - do not hand-edit -->", "<!-- END ui nav -->"
B_CSS, E_CSS = "<!-- BEGIN ui css, generated by make_site_ui.py - do not hand-edit -->", "<!-- END ui css -->"
B_JS, E_JS = "<!-- BEGIN ui js, generated by make_site_ui.py - do not hand-edit -->", "<!-- END ui js -->"

# one nav entry: a button, optionally wrapped in a generator's BEGIN/END nav markers
ENTRY = re.compile(
    r'(?P<wrapped><!-- BEGIN (?P<gen>[\w-]+) nav -->\s*(?P<wbtn><button class="[^"]*" data-t="[^"]+">.*?</button>)\s*<!-- END (?P=gen) nav -->)'
    r'|(?P<btn><button class="(?P<cls>[^"]*)" data-t="(?P<t>[^"]+)">.*?</button>)', re.S)

CSS = r"""<style id="ui-css">
  /* ---- nav groups, search, retired ---- */
  .ui-navh{font-size:10.5px;font-weight:700;letter-spacing:.9px;text-transform:uppercase;color:var(--muted);
    margin:14px 10px 3px;display:flex;align-items:center;gap:8px}
  .ui-navh.ui-ret{margin-top:18px;padding-top:12px;border-top:1px solid var(--line)}
  .ui-rett{background:none;border:0;padding:0;margin:0;font:inherit;color:var(--muted);cursor:pointer;
    display:flex;align-items:center;gap:6px;width:auto;letter-spacing:inherit;text-transform:inherit;font-weight:inherit}
  .ui-rett:hover{background:none;color:var(--ink)}
  .ui-rett .ui-n{background:var(--chip);border-radius:999px;padding:0 7px;font-size:10.5px;color:var(--muted)}
  .ui-rett .ui-car{display:inline-block;transition:transform .15s;font-size:9px}
  .ui-rett[aria-expanded="true"] .ui-car{transform:rotate(90deg)}
  .ui-retired[hidden]{display:none}
  .ui-search{position:relative;margin:0 6px 10px}
  .ui-search input{width:100%;font:inherit;font-size:13px;padding:7px 10px 7px 30px;border:1px solid var(--line);
    border-radius:8px;background:#fbfcfd;color:var(--ink);outline:none}
  .ui-search input:focus{border-color:var(--accent);background:#fff;box-shadow:0 0 0 3px rgba(31,119,180,.10)}
  .ui-search::before{content:"";position:absolute;left:10px;top:9px;width:12px;height:12px;border:2px solid #9aa5b1;
    border-radius:50%;box-sizing:border-box}
  .ui-search::after{content:"";position:absolute;left:19px;top:19px;width:6px;height:2px;background:#9aa5b1;transform:rotate(45deg)}
  #ui-qres{position:absolute;left:0;right:0;top:36px;z-index:40;background:#fff;border:1px solid var(--line);border-radius:10px;
    box-shadow:0 10px 30px rgba(20,30,45,.14);max-height:60vh;overflow:auto;padding:4px}
  #ui-qres[hidden]{display:none}
  #ui-qres button{display:block;width:100%;text-align:left;background:none;border:0;border-radius:7px;padding:7px 9px;
    font:inherit;font-size:12.5px;color:var(--ink);cursor:pointer;line-height:1.35}
  #ui-qres button:hover,#ui-qres button.sel{background:var(--chip)}
  #ui-qres .ui-qt{display:block;font-size:10.5px;color:var(--muted);letter-spacing:.4px;text-transform:uppercase}
  #ui-qres .ui-qk{color:var(--muted);font-size:11px;margin-left:6px}
  #ui-qres .ui-qnone{padding:8px 10px;font-size:12.5px;color:var(--muted)}
  #ui-qres mark{background:#fff1b8;padding:0 1px;border-radius:2px}
  nav .ui-navtog{display:none}
  /* ---- sub-tab strip ---- */
  .ui-strip{position:sticky;top:0;z-index:20;display:flex;align-items:center;gap:6px;flex-wrap:wrap;
    background:rgba(246,247,249,.92);backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);
    padding:8px 0 8px;margin:6px 0 18px;border-bottom:1px solid var(--line)}
  .ui-strip .ui-lbl{font-size:10.5px;font-weight:700;letter-spacing:.9px;text-transform:uppercase;color:var(--muted);margin-right:2px}
  .ui-strip .ui-pills{display:flex;gap:5px;flex-wrap:wrap;flex:1 1 auto;min-width:0}
  .ui-pill{background:var(--panel);border:1px solid var(--line);border-radius:999px;padding:3px 10px;font:inherit;font-size:12px;
    color:#39434e;cursor:pointer;white-space:nowrap;line-height:1.4;max-width:260px;overflow:hidden;text-overflow:ellipsis}
  .ui-pill:hover{border-color:#b9c3cd;background:#fff}
  .ui-pill.on{background:var(--ink);color:#fff;border-color:var(--ink)}
  .ui-pill.closed{color:var(--muted);border-style:dashed}
  .ui-tools{display:flex;gap:4px;align-items:center;margin-left:auto}
  .ui-tb{background:none;border:1px solid transparent;border-radius:7px;padding:3px 8px;font:inherit;font-size:11.5px;color:var(--muted);cursor:pointer;white-space:nowrap}
  .ui-tb:hover{background:var(--panel);border-color:var(--line);color:var(--ink)}
  .ui-tb.on{background:var(--panel);border-color:var(--line);color:var(--ink);font-weight:600}
  .ui-tb[disabled]{opacity:.4;cursor:default}
  /* ---- collapsible groups ---- */
  .ui-group>.ui-head{cursor:pointer;position:relative;user-select:none}
  .ui-group>h3.ui-head{padding-right:26px}
  .ui-chev{position:absolute;right:2px;top:0;width:18px;height:18px;border-radius:5px;display:inline-flex;align-items:center;
    justify-content:center;color:var(--muted);font-size:10px;transition:transform .15s;pointer-events:none}
  .ui-group>h3.ui-head:hover .ui-chev{background:var(--chip);color:var(--ink)}
  .ui-group.ui-closed>.ui-head .ui-chev{transform:rotate(-90deg)}
  .ui-group.ui-closed>*:not(.ui-head){display:none!important}
  .ui-group.ui-closed>h3.ui-head{margin-bottom:4px;border-bottom-style:dashed}
  .ui-group.ui-closed>h3.ui-head::after{content:attr(data-ui-count);position:absolute;right:28px;top:0;font-weight:500;
    letter-spacing:0;text-transform:none;font-size:11px;color:var(--muted)}
  .nb-day.ui-group>.nb-date.ui-head{cursor:pointer}
  .nb-day.ui-group>.nb-date .ui-chev,.mtg.ui-group>h4 .ui-chev{position:static;margin-left:8px;display:inline-flex;vertical-align:middle}
  .mtg.ui-group>h4.ui-head{cursor:pointer}
  .mtg.ui-group.ui-closed>*:not(.ui-head):not(.mtg-date){display:none!important}
  .mtg.ui-group.ui-closed>.mtg-date{display:block!important}
  section.ui-focus .ui-group:not(.ui-cur){display:none!important}
  .ui-flash{animation:uiflash 1.4s ease-out}
  @keyframes uiflash{0%{background:#fff1b8}100%{background:transparent}}
  h3.ui-syn{color:#9aa5b1}
  .ui-sel{font:inherit;font-size:12px;padding:3px 8px;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:#39434e;max-width:300px}
  .ui-tw{overflow-x:auto;margin:8px 0 4px}
  .ui-tw>table{margin:0}
  /* ---- report shape: role tabs, the answer block ---- */
  .ui-roles{flex:1 0 100%;display:flex;gap:2px;margin:-2px 0 6px;border-bottom:1px solid var(--line)}
  .ui-role{background:none;border:0;border-bottom:2px solid transparent;margin-bottom:-1px;padding:6px 12px 7px;font:inherit;font-size:13px;
    font-weight:600;color:var(--muted);cursor:pointer;display:inline-flex;align-items:center;gap:6px}
  .ui-role:hover{color:var(--ink)}
  .ui-role.on{color:var(--ink);border-bottom-color:var(--ink)}
  .ui-role .ui-rn{font-size:10.5px;font-weight:600;color:var(--muted);background:var(--chip);border-radius:999px;padding:0 6px}
  .ui-role.on .ui-rn{background:var(--ink);color:#fff}
  .ui-rolehide{display:none!important}
  .ui-strip.ui-nopills .ui-lbl,.ui-strip.ui-nopills .ui-pills{display:none}
  .ui-answer{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--ink);border-radius:12px;padding:16px 20px 12px;margin:0 0 22px}
  .ui-answer[hidden]{display:none}
  .ui-alab{font-size:10.5px;font-weight:700;letter-spacing:.9px;text-transform:uppercase;color:var(--muted);margin:0 0 6px}
  .ui-answer p{margin:0 0 10px;font-size:15px;line-height:1.55;max-width:78ch}
  .ui-answer ul{margin:4px 0 10px;padding-left:20px;max-width:78ch} .ui-answer li{margin:3px 0;font-size:13.5px;color:#39434e}
  .ui-answer .ui-astat{font-size:11.5px;color:var(--muted);border-top:1px solid var(--line);padding-top:8px;margin-top:4px;
    font-family:ui-monospace,Menlo,Consolas,monospace}
  .ui-tablist{list-style:none;padding:0!important;margin:6px 0 10px!important;max-width:none!important}
  .ui-tablist li{padding:6px 0;border-top:1px solid var(--line);font-size:13.5px!important}
  .ui-tablist li:first-child{border-top:0}
  /* ---- proposed deletions ---- */
  .ui-delnote{font-size:11px;font-weight:600;color:#c0392b;letter-spacing:.2px;margin:14px 0 -8px;padding-left:10px;border-left:3px solid #c0392b}
  .ui-del,.ui-del *{text-decoration:line-through;color:#9aa5b1!important;text-decoration-color:#c0392b}
  .ui-del img{opacity:.35}
  .ui-del.ui-group>.ui-head{text-decoration:line-through;color:#9aa5b1}
  .ui-del.ui-group>.ui-delnote,.ui-del.ui-group>.ui-delnote *{text-decoration:none;color:#c0392b!important}
  .ui-del.ui-group>.ui-head .ui-chev{text-decoration:none}
  .ui-pill.del{text-decoration:line-through;border-color:#e3b3ad}
  section.ui-hidedel .ui-del,section.ui-hidedel .ui-delnote{display:none!important}
  .ui-deltog{color:#c0392b}
  /* ---- Cornell notebook cards ---- */
  .nb-day.ui-cornell{display:grid;grid-template-columns:150px 1fr}
  .nb-day.ui-cornell>.nb-date{grid-column:1/-1}
  .nb-day.ui-cornell>.ui-cue{grid-column:1;grid-row:2/span var(--rows,4);padding:9px 14px;border-right:1px solid #eef1f4;border-top:1px solid #eef1f4;
    background:#fbfcfd;font-size:12.5px;line-height:1.4;color:var(--ink);font-weight:600}
  .nb-day.ui-cornell>.nb-row{grid-column:2}
  .nb-day.ui-cornell>.nb-row .nb-stage{border-right:1px solid #eef1f4}
  .nb-day.ui-cornell>.ui-sum{grid-column:1/-1;border-top:1px solid var(--line);background:#f4f6f8;padding:8px 14px;font-size:12.5px;color:#39434e}
  .ui-cuelab{display:block;font-size:9.5px;font-weight:700;letter-spacing:.8px;text-transform:uppercase;color:var(--muted);margin:0 0 3px}
  .ui-sum .ui-cuelab{display:inline;margin:0 8px 0 0}
  @media(max-width:700px){.nb-day.ui-cornell{grid-template-columns:1fr}.nb-day.ui-cornell>.ui-cue{grid-column:1;grid-row:auto;border-right:0}.nb-day.ui-cornell>.nb-row{grid-column:1}}
  /* ---- lightbox ---- */
  #ui-lb{position:fixed;inset:0;z-index:1000;background:rgba(16,22,30,.86);display:flex;flex-direction:column;
    animation:fade .15s ease}
  #ui-lb[hidden]{display:none}
  #ui-lb .ui-lbbar{display:flex;align-items:center;gap:12px;padding:10px 16px;color:#e8edf2;font-size:13px;flex:0 0 auto}
  #ui-lb .ui-lbbar b{font-weight:600;flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  #ui-lb .ui-lbbar button,#ui-lb .ui-lbbar a{background:rgba(255,255,255,.12);border:0;color:#fff;font:inherit;font-size:12.5px;
    padding:5px 11px;border-radius:7px;cursor:pointer;text-decoration:none}
  #ui-lb .ui-lbbar button:hover,#ui-lb .ui-lbbar a:hover{background:rgba(255,255,255,.22)}
  #ui-lb .ui-lbin{flex:1 1 auto;overflow:auto;padding:0 16px 16px;text-align:center;cursor:zoom-in}
  #ui-lb .ui-lbin img{max-width:100%;height:auto;border-radius:6px;background:#fff}
  #ui-lb.ui-full .ui-lbin{cursor:zoom-out;text-align:left}
  #ui-lb.ui-full .ui-lbin img{max-width:none}
  figure .imgwrap img{cursor:zoom-in}
  /* ---- progress, back to top ---- */
  #ui-prog{position:fixed;left:0;top:0;height:3px;width:0;background:var(--accent);z-index:900;transition:width .1s linear}
  #ui-top{position:fixed;right:18px;bottom:18px;z-index:800;width:38px;height:38px;border-radius:50%;border:1px solid var(--line);
    background:var(--panel);color:var(--ink);cursor:pointer;box-shadow:0 4px 14px rgba(20,30,45,.14);font-size:16px;opacity:0;
    pointer-events:none;transition:opacity .2s}
  #ui-top.show{opacity:1;pointer-events:auto}
  /* ---- phone ---- */
  @media(max-width:820px){
    .wrap{flex-direction:column}
    /* the base sheet's flex:0 0 246px becomes a 246 px HEIGHT once the column stacks - undo it */
    nav{width:auto;height:auto;flex:0 0 auto;position:sticky;top:0;z-index:30;border-right:0;border-bottom:1px solid var(--line);
      display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px;padding:10px 12px;max-height:100vh;overflow:auto}
    nav h1{flex:1 1 auto;width:auto;margin:0;font-size:14px}
    nav .sub{display:none}
    nav .ui-navtog{display:inline-flex;flex:0 0 auto;width:auto;margin:0;padding:6px 10px;border:1px solid var(--line);
      border-radius:8px;font-size:12.5px;gap:6px;background:var(--panel)}
    nav .ui-navtog .ui-cur{font-weight:600;max-width:46vw;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    nav .ui-search{flex:1 0 100%;margin:2px 0 0}
    nav .ui-navlist{display:none;flex:1 0 100%}
    nav.open .ui-navlist{display:block;padding-bottom:6px}
    nav .foot{display:none;flex:1 0 100%;width:auto}
    nav.open .foot{display:block}
    main{padding:16px 12px 70px}
    .ui-strip{top:var(--ui-navh,0px);flex-wrap:nowrap;overflow-x:auto;-webkit-overflow-scrolling:touch;padding:8px 0}
    .ui-strip .ui-lbl{display:none}
    .ui-strip .ui-pills{flex-wrap:nowrap;flex:0 0 auto}
    .ui-pill{flex:0 0 auto;max-width:200px}
    .ui-tools{margin-left:8px;flex:0 0 auto}
    h2.title{font-size:22px}
    figure .imgwrap{padding:8px}
  }
  /* ---- print: the open tab, everything expanded, no chrome ---- */
  @media print{
    nav,.ui-strip,#ui-top,#ui-prog,#ui-lb,.ui-chev{display:none!important}
    .wrap{display:block} main{max-width:none;padding:0}
    .ui-group.ui-closed>*:not(.ui-head){display:block!important}
    section.ui-focus .ui-group:not(.ui-cur){display:block!important}
    .ui-rolehide{display:block!important} .ui-answer[hidden]{display:block!important}
    section.ui-hidedel .ui-del,section.ui-hidedel .ui-delnote{display:none!important}
    figure,table,.method,.callout{break-inside:avoid}
    figure .imgwrap{overflow:visible}
    a{color:inherit}
  }
</style>"""

JS = r"""<script id="ui-js">
(function(){
"use strict";
const D=__UI_DATA__;
const $=(s,r)=>(r||document).querySelector(s), $$=(s,r)=>Array.from((r||document).querySelectorAll(s));
const LS={get(k){try{return localStorage.getItem("fbm.ui."+k);}catch(e){return null;}},
          set(k,v){try{v==null?localStorage.removeItem("fbm.ui."+k):localStorage.setItem("fbm.ui."+k,v);}catch(e){}}};
const text=el=>(el.textContent||"").replace(/\s+/g," ").trim();
const figNo=n=>{if(!n)return "";const c=n.cloneNode(true);c.querySelectorAll(".runid").forEach(r=>r.remove());return text(c).split(" ").slice(0,2).join(" ");};
const slug=t=>t.toLowerCase().replace(/[^a-z0-9]+/g,"-").replace(/^-+|-+$/g,"").slice(0,48)||"x";
const shortT=(t,n)=>{n=n||38;t=t.replace(/^\d+(\.\d+)?\s*[\u00b7.\-\u2013]\s*/,"");return t.length>n?t.slice(0,n-1).replace(/\s+\S*$/,"")+"\u2026":t;};
const nav=$("nav"), main=$("main"), secs=$$("main section");
const navBtn=id=>$('nav button[data-t="'+CSS.escape(id)+'"]');
const tabLabel=id=>{const b=navBtn(id);return b?text(b):id;};
const curSec=()=>secs.find(s=>s.classList.contains("on"))||null;
const showTab=id=>{if(typeof show==="function")show(id);};

/* ---------- nav: retired group, phone toggle ---------- */
const ret=$(".ui-retired"), rett=$(".ui-rett");
function setRet(open){if(!ret||!rett)return;ret.hidden=!open;rett.setAttribute("aria-expanded",String(open));LS.set("nav.retired",open?"1":"0");}
if(ret&&rett){setRet(LS.get("nav.retired")==="1");rett.addEventListener("click",()=>setRet(ret.hidden));}
const navlist=$("nav .ui-navlist");
if(navlist){
  const tog=document.createElement("button");tog.type="button";tog.className="ui-navtog";
  tog.innerHTML='<span class="ui-cur"></span><span class="ui-car">&#9662;</span>';
  nav.insertBefore(tog,nav.querySelector("h1").nextSibling);
  tog.addEventListener("click",()=>nav.classList.toggle("open"));
  $$("button[data-t]",navlist).forEach(b=>b.addEventListener("click",()=>nav.classList.remove("open")));
}
function syncNav(){
  const s=curSec(); const cur=$("nav .ui-navtog .ui-cur"); if(cur)cur.textContent=s?tabLabel(s.id):"";
  if(s&&ret&&ret.contains(navBtn(s.id)||ret)&&navBtn(s.id))setRet(true);
}

/* ---------- groups: sub-tabs, collapsibles ---------- */
const REG={};   // section id -> {groups:[{el,key,head,pill}], strip, mode}
function mkChev(){const c=document.createElement("span");c.className="ui-chev";c.textContent="\u25BC";return c;}
function groupsOf(sec){
  if(sec.querySelector(".pf-bar")) return [];              // the tab has sub-tabs of its own
  const kids=Array.from(sec.children); let out=[];
  const heads=kids.filter(k=>k.tagName==="H3");
  if(heads.length>=2){
    heads.forEach(h=>{
      const g=document.createElement("div"); g.className="ui-group";
      sec.insertBefore(g,h); g.appendChild(h);
      let n=g.nextSibling;
      while(n&&!(n.nodeType===1&&n.tagName==="H3")){const nx=n.nextSibling;g.appendChild(n);n=nx;}
      const label=text(h); const key=h.id||slug(label); if(!h.id)h.id=sec.id+"-"+key;
      h.classList.add("ui-head"); h.appendChild(mkChev()); g.dataset.key=key;
      out.push({el:g,key,head:h,label});
    });
    // everything before the first heading, except the tab's own header (eyebrow, title, lead), becomes an "Intro" group
    const first=out[0].el; const isHdr=n=>n.nodeType===1&&(n.tagName==="H2"||n.classList.contains("eyebrow")||n.classList.contains("lead")||n.classList.contains("retired-note"));
    const pre=[]; for(let n=sec.firstChild;n&&n!==first;n=n.nextSibling){if(!isHdr(n))pre.push(n);}
    if(pre.some(n=>n.nodeType===1)){
      const hdr=Array.from(sec.children).filter(isHdr); const after=hdr.length?hdr[hdr.length-1]:null;
      const g=document.createElement("div"); g.className="ui-group ui-intro"; g.dataset.key="intro";
      const h=document.createElement("h3"); h.className="ui-head ui-syn"; h.id=sec.id+"-intro"; h.textContent="Intro"; h.appendChild(mkChev());
      if(after)sec.insertBefore(g,after.nextSibling); else sec.insertBefore(g,first);
      g.appendChild(h); pre.forEach(n=>g.appendChild(n));
      out.unshift({el:g,key:"intro",head:h,label:"Intro"});
    }
    out.forEach(o=>{const g=o.el,h=o.head; const nf=g.querySelectorAll("figure").length, nt=g.querySelectorAll("table").length, nw=text(g).split(" ").length;
      h.setAttribute("data-ui-count",[nw+" words",nf?nf+" fig":"",nt?nt+" tbl":""].filter(Boolean).join(" \u00b7 "));});
    return out;
  }
  const days=$$(":scope > .nb-day",sec);
  if(days.length>=2){
    days.forEach((d,i)=>{const h=d.querySelector(".nb-date"); if(!h)return; d.classList.add("ui-group"); h.classList.add("ui-head");
      const first=h.firstChild; const tag=h.querySelector(".tag");
      const lab=(first&&first.nodeType===1&&!first.classList.contains("tag"))?text(first):(first&&first.nodeType===3&&first.textContent.trim())?first.textContent.trim():text(h).replace(tag?text(tag):"","").trim();
      const tagText=tag?text(tag):"";
      const key=slug(lab)+"-"+i; d.dataset.key=key; if(!d.id)d.id=sec.id+"-"+key; h.appendChild(mkChev());
      out.push({el:d,key,head:h,label:lab+(tagText?" · "+shortT(tagText,70):"")});});
    return out;
  }
  const mtgs=$$(".mtg",sec);
  if(mtgs.length>=2){
    mtgs.forEach((m,i)=>{const h=m.querySelector("h4"), d=m.querySelector(".mtg-date"); if(!h)return; m.classList.add("ui-group"); h.classList.add("ui-head");
      const lab=(d?text(d):"")||text(h); const key=slug(lab)+"-"+i; m.dataset.key=key; if(!m.id)m.id=sec.id+"-"+key; h.appendChild(mkChev());
      out.push({el:m,key,head:h,label:shortT(lab,22)});});
    return out;
  }
  return [];
}
function stateKey(sec){return sec.id+".closed";}
function loadClosed(sec){try{return new Set(JSON.parse(LS.get(stateKey(sec))||"[]"));}catch(e){return new Set();}}
function saveClosed(sec){const c=REG[sec.id].groups.filter(g=>g.el.classList.contains("ui-closed")).map(g=>g.key);LS.set(stateKey(sec),c.length?JSON.stringify(c):null);}
function setClosed(sec,g,closed){g.el.classList.toggle("ui-closed",closed);if(g.pill&&g.pill.classList)g.pill.classList.toggle("closed",closed);}
function build(sec){
  const groups=groupsOf(sec); if(!groups.length)return;
  const many=groups.length>14;
  const strip=document.createElement("div"); strip.className="ui-strip";
  strip.innerHTML='<span class="ui-lbl">In this tab</span>'+(many?'<select class="ui-sel" title="jump to"></select>':'<div class="ui-pills"></div>')+'<div class="ui-tools">'
    +'<button type="button" class="ui-tb" data-act="prev" title="previous section">&#8249;</button>'
    +'<button type="button" class="ui-tb" data-act="next" title="next section">&#8250;</button>'
    +'<button type="button" class="ui-tb" data-act="focus" title="show one section at a time">one at a time</button>'
    +'<button type="button" class="ui-tb" data-act="expand">expand all</button>'
    +'<button type="button" class="ui-tb" data-act="collapse">collapse all</button>'
    +'<button type="button" class="ui-tb" data-act="print" title="print this tab (everything expanded)">print</button></div>';
  const pills=strip.querySelector(".ui-pills"), sel=strip.querySelector(".ui-sel");
  const closed=loadClosed(sec);
  groups.forEach((g,i)=>{
    if(sel){const o=document.createElement("option");o.value=String(i);o.textContent=g.label;sel.appendChild(o);g.pill=o;}
    else{const p=document.createElement("button"); p.type="button"; p.className="ui-pill"; p.textContent=shortT(g.label); p.title=g.label;
      p.addEventListener("click",()=>goTo(sec,g,true)); pills.appendChild(p); g.pill=p;}
    if(closed.has(g.key))setClosed(sec,g,true);
    g.head.addEventListener("click",ev=>{if(ev.target.closest("a,button,input,textarea,select"))return;
      setClosed(sec,g,!g.el.classList.contains("ui-closed")); saveClosed(sec);});
  });
  if(sel){sel.insertAdjacentHTML("afterbegin",'<option value="">jump to\u2026</option>');sel.addEventListener("change",()=>{const g=groups[+sel.value];if(g)goTo(sec,g,true);sel.value="";});}
  let anchor=groups[0].el; while(anchor.parentElement!==sec)anchor=anchor.parentElement;
  sec.insertBefore(strip,anchor);
  REG[sec.id]={groups,strip,mode:LS.get(sec.id+".mode")==="focus"?"focus":"jump",cur:null};
  strip.addEventListener("click",ev=>{
    const b=ev.target.closest("button[data-act]"); if(!b)return; const act=b.dataset.act, R=REG[sec.id];
    if(act==="expand"||act==="collapse"){R.groups.forEach(g=>setClosed(sec,g,act==="collapse"));saveClosed(sec);}
    else if(act==="focus"){setMode(sec,R.mode==="focus"?"jump":"focus");}
    else if(act==="prev"||act==="next"){const V=R.groups.filter(vis);const i=V.indexOf(R.cur||V[0]);const j=Math.min(V.length-1,Math.max(0,i+(act==="next"?1:-1)));if(V[j])goTo(sec,V[j],true);}
    else if(act==="print"){window.print();}
  });
  setMode(sec,REG[sec.id].mode,true);
  if(D.roles[sec.id])applyRoles(sec);
  applyDel(sec);
  if(sec.id==="notebook")applyCornell(sec);
}
/* ---------- the report shape: Answer -> Evidence -> Method -> Open -> History ---------- */
const vis=g=>!g.el.classList.contains("ui-rolehide");
function tabList(){
  const out=[]; $$("nav .ui-navlist button[data-t]").forEach(b=>{const t=b.dataset.t; const A=D.answers[t]; if(!A||t==="overview"||b.classList.contains("retired"))return;
    out.push('<li><a href="#'+t+'"><b>'+esc(text(b))+'</b></a> — '+esc(A.one)+'</li>');});
  return '<ul class="ui-tablist">'+out.join("")+'</ul>';
}
function applyRoles(sec){
  const R=REG[sec.id], map=D.roles[sec.id]||{}; R.roles=true;
  R.groups.forEach(g=>{g.role=map[g.key]||"history"; if(!(g.key in map)&&Object.keys(map).length)console.warn("ui roles: "+sec.id+" has no role for "+g.key); g.el.dataset.role=g.role;});
  const A=D.answers[sec.id]; let ans=null;
  if(A){ans=document.createElement("div"); ans.className="ui-answer"; ans.dataset.role="answer";
    ans.innerHTML='<div class="ui-alab">Answer</div>'+A.html.replace("<!--TABLIST-->",tabList())+(A.status?'<div class="ui-astat">'+esc(A.status)+'</div>':'');
    sec.insertBefore(ans,R.strip.nextSibling);}
  R.answer=ans;
  // groups in role order (a stable sort), directly under the strip
  if(R.groups.every(g=>g.el.parentElement===sec)){
    const order=D.roleOrder; R.groups=[...R.groups].sort((a,b)=>order.indexOf(a.role)-order.indexOf(b.role));
    let cursor=ans||R.strip; R.groups.forEach(g=>{sec.insertBefore(g.el,cursor.nextSibling);cursor=g.el;});
    const pills=R.strip.querySelector(".ui-pills"); if(pills)R.groups.forEach(g=>{if(g.pill&&g.pill.parentElement===pills)pills.appendChild(g.pill);});
  }
  const bar=document.createElement("div"); bar.className="ui-roles";
  D.roleOrder.forEach(r=>{const n=R.groups.filter(g=>g.role===r).length+(r==="answer"&&ans?1:0); if(!n)return;
    const b=document.createElement("button"); b.type="button"; b.className="ui-role"; b.dataset.role=r; b.innerHTML=esc(D.roleLabel[r])+'<span class="ui-rn">'+n+'</span>';
    b.addEventListener("click",()=>setRole(sec,r)); bar.appendChild(b);});
  R.strip.insertBefore(bar,R.strip.firstChild); R.strip.classList.add("ui-hasroles");
  const saved=LS.get(sec.id+".role"); setRole(sec,(saved&&bar.querySelector('[data-role="'+saved+'"]'))?saved:(ans?"answer":bar.firstChild.dataset.role),true);
}
function setRole(sec,role,silent){
  const R=REG[sec.id]; if(!R.roles)return; R.role=role; LS.set(sec.id+".role",role==="answer"?null:role);
  $$(".ui-role",R.strip).forEach(b=>b.classList.toggle("on",b.dataset.role===role));
  if(R.answer)R.answer.hidden=role!=="answer";
  R.groups.forEach(g=>{const off=g.role!==role; g.el.classList.toggle("ui-rolehide",off); if(g.pill&&g.pill.tagName==="BUTTON")g.pill.hidden=off;});
  R.cur=null; R.groups.forEach(g=>{if(g.pill.classList)g.pill.classList.remove("on");});
  R.strip.classList.toggle("ui-nopills",!R.groups.some(g=>vis(g)&&g.pill&&g.pill.tagName==="BUTTON"));
  if(R.mode==="focus"){const g=R.groups.find(vis); if(g)setCur(sec,g);}
  updateArrows(sec); if(!silent)window.scrollTo({top:Math.max(0,R.strip.getBoundingClientRect().top+window.scrollY-6)});
}
/* ---------- proposed deletions: struck through, with the reason; nothing removed ---------- */
function applyDel(sec){
  const R=REG[sec.id]; if(!R)return; let n=0;
  D.del.filter(d=>d.sec===sec.id).forEach(d=>{
    const g=R.groups.find(x=>x.key===d.key); if(!g){console.warn("ui del: no group "+sec.id+"/"+d.key);return;}
    const note=document.createElement("div"); note.className="ui-delnote"; note.textContent="proposed deletion — "+d.why;
    if(!d.prefix){g.el.classList.add("ui-del"); g.head.insertAdjacentElement("afterend",note); if(g.pill&&g.pill.classList)g.pill.classList.add("del"); n++; return;}
    const want=d.prefix.toLowerCase(); const el=Array.from(g.el.children).find(c=>c!==g.head&&!c.classList.contains("ui-delnote")&&text(c).toLowerCase().startsWith(want));
    if(!el){console.warn("ui del: not found "+sec.id+"/"+d.key+" “"+d.prefix+"”");return;}
    el.classList.add("ui-del"); el.insertAdjacentElement("beforebegin",note); n++;
  });
  R.ndel=n; if(!n)return;
  const b=document.createElement("button"); b.type="button"; b.className="ui-tb ui-deltog"; b.dataset.act="deltog";
  const hide=LS.get(sec.id+".hidedel")==="1"; sec.classList.toggle("ui-hidedel",hide);
  const label=()=>{b.textContent=(sec.classList.contains("ui-hidedel")?"show ":"hide ")+n+" struck";}; label();
  b.addEventListener("click",()=>{sec.classList.toggle("ui-hidedel");LS.set(sec.id+".hidedel",sec.classList.contains("ui-hidedel")?"1":null);label();});
  R.strip.querySelector(".ui-tools").appendChild(b);
}
/* ---------- the notebook as Cornell notes: cue column, notes, one-line summary ---------- */
function applyCornell(sec){
  const R=REG[sec.id]; if(!R)return;
  R.groups.forEach(g=>{
    const h=g.head; const first=h.firstElementChild; const date=first&&!first.classList.contains("tag")?text(first):(h.firstChild&&h.firstChild.nodeType===3?h.firstChild.textContent.trim():"");
    const tag=h.querySelector(".tag"); const tagText=tag?text(tag):"";
    const e=D.cornell.find(x=>x.date===date&&(!x.tag||tagText.startsWith(x.tag))); if(!e)return;
    const rows=$$(":scope > .nb-row",g.el); g.el.classList.add("ui-cornell"); g.el.style.setProperty("--rows",String(rows.length));
    const cue=document.createElement("div"); cue.className="ui-cue"; cue.innerHTML='<span class="ui-cuelab">Cue</span>'+esc(e.cue);
    const sum=document.createElement("div"); sum.className="ui-sum"; sum.innerHTML='<span class="ui-cuelab">Summary</span>'+esc(e.sum);
    h.insertAdjacentElement("afterend",cue); g.el.appendChild(sum);
    if(g.pill)g.pill.textContent=date+" · "+e.cue;
  });
}
function setMode(sec,mode,silent){
  const R=REG[sec.id]; R.mode=mode; LS.set(sec.id+".mode",mode==="focus"?"focus":null);
  sec.classList.toggle("ui-focus",mode==="focus");
  const fb=R.strip.querySelector('[data-act="focus"]'); fb.classList.toggle("on",mode==="focus");
  if(mode==="focus"){const k=LS.get(sec.id+".cur"); const g=R.groups.find(x=>x.key===k&&vis(x))||R.groups.find(vis)||R.groups[0]; setCur(sec,g); if(!silent)window.scrollTo({top:0});}
  else{R.groups.forEach(g=>g.el.classList.remove("ui-cur"));}
  updateArrows(sec);
}
function setCur(sec,g){const R=REG[sec.id];R.cur=g;R.groups.forEach(x=>{x.el.classList.toggle("ui-cur",x===g);if(x.pill.classList)x.pill.classList.toggle("on",x===g);});
  if(R.mode==="focus"){LS.set(sec.id+".cur",g.key);setClosed(sec,g,false);} updateArrows(sec);}
function updateArrows(sec){const R=REG[sec.id];const V=R.groups.filter(vis);const i=V.indexOf(R.cur);
  R.strip.querySelector('[data-act="prev"]').disabled=i<=0; R.strip.querySelector('[data-act="next"]').disabled=i<0||i>=V.length-1;}
function goTo(sec,g,scroll){
  const R=REG[sec.id]; if(!R)return;
  if(R.roles&&g.role&&g.role!==R.role)setRole(sec,g.role,true);
  if(g.el.classList.contains("ui-closed")){setClosed(sec,g,false);saveClosed(sec);}
  setCur(sec,g);
  if(R.mode==="focus"){window.scrollTo({top:Math.max(0,R.strip.getBoundingClientRect().top+window.scrollY-6),behavior:"auto"});return;}
  if(scroll){const y=g.el.getBoundingClientRect().top+window.scrollY-R.strip.offsetHeight-14;window.scrollTo({top:y,behavior:"auto"});}
}
secs.forEach(sec=>{try{build(sec);}catch(e){console.error("ui layer: "+sec.id+": "+e.message+" @ "+String(e.stack||"").split(/\r?\n/).slice(1,3).join(" | "));}});

/* the pill that is "current" while reading: the last group whose top has passed the strip */
let tick=false;
function track(){
  tick=false; const s=curSec(); const R=s&&REG[s.id]; if(!R||R.mode==="focus")return;
  const line=R.strip.getBoundingClientRect().bottom+16; let cur=null;
  for(const g of R.groups){if(!vis(g))continue;if(g.el.getBoundingClientRect().top<=line)cur=g;else break;}
  if(cur!==R.cur){R.cur=cur;R.groups.forEach(x=>{if(x.pill.classList)x.pill.classList.toggle("on",x===cur);});updateArrows(s);}
}
window.addEventListener("scroll",()=>{if(!tick){tick=true;requestAnimationFrame(track);}
  const p=$("#ui-prog"); if(p){const h=document.documentElement;const m=h.scrollHeight-h.clientHeight;p.style.width=(m>0?100*h.scrollTop/m:0)+"%";}
  const t=$("#ui-top"); if(t)t.classList.toggle("show",window.scrollY>900);},{passive:true});

/* ---------- deep links into a section ---------- */
function openTo(id,flash){
  const el=document.getElementById(id); if(!el)return false;
  const sec=el.closest("main section"); if(!sec)return false;
  if(!sec.classList.contains("on")){showTab(sec.id);history.replaceState(null,"","#"+id);}
  const R=REG[sec.id]; const g=R&&R.groups.find(x=>x.el===el||x.el.contains(el));
  if(g){goTo(sec,g,false);}
  const pane=el.closest(".pf-pane"); if(pane){const b=$('.pf-bar button[data-pt="'+CSS.escape(pane.id.slice(8))+'"]');if(b)b.click();}
  setTimeout(()=>{const top=el.getBoundingClientRect().top+window.scrollY-(R?R.strip.offsetHeight:0)-14;window.scrollTo({top:Math.max(0,top),behavior:"auto"});
    if(flash){el.classList.remove("ui-flash");void el.offsetWidth;el.classList.add("ui-flash");}},30);
  return true;
}
function onHash(){const h=location.hash.slice(1); if(!h)return; const el=document.getElementById(h); if(!el)return;
  if(el.tagName==="SECTION"&&el.parentElement===main){syncNav();return;} openTo(h,true);}
window.addEventListener("hashchange",()=>setTimeout(onHash,0));
$$("nav button[data-t]").forEach(b=>b.addEventListener("click",()=>setTimeout(syncNav,0)));
setTimeout(()=>{onHash();syncNav();},0);

/* ---------- tables scroll instead of overflowing on narrow screens ---------- */
$$("main table").forEach(t=>{if(t.closest(".ui-tw,.imgwrap"))return;const w=document.createElement("div");w.className="ui-tw";t.parentNode.insertBefore(w,t);w.appendChild(t);});

/* ---------- search over headings ---------- */
const q=$("#ui-q"), qres=$("#ui-qres"); const INDEX=[];
secs.forEach(sec=>{
  const tab=tabLabel(sec.id); const h2=sec.querySelector("h2");
  INDEX.push({tab,id:sec.id,kind:"tab",label:h2?text(h2):tab});
  const R=REG[sec.id]; if(R)R.groups.forEach(g=>INDEX.push({tab,id:g.el.id||g.head.id,kind:"section",label:g.label}));
  $$(".pf-pane",sec).forEach(p=>{const h=p.querySelector("h3");if(h)INDEX.push({tab,id:p.id,kind:"part",label:text(h)});});
  $$("figure",sec).forEach((f,i)=>{const h=f.querySelector("h4"),n=f.querySelector(".fignum");if(!h)return;if(!f.id)f.id=sec.id+"-fig-"+(i+1);
    INDEX.push({tab,id:f.id,kind:"figure",label:(n?figNo(n)+" ":"")+text(h)});});
  $$(".method h3, .callout h3",sec).forEach((h,i)=>{if(h.closest(".ui-group"))return;if(!h.id)h.id=sec.id+"-box-"+(i+1);INDEX.push({tab,id:h.id,kind:"box",label:text(h)});});
});
let qsel=-1;
function esc(s){return s.replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
function mark(label,words){let h=esc(label);words.forEach(w=>{if(!w)return;h=h.replace(new RegExp("("+w.replace(/[.*+?^${}()|[\]\\]/g,"\\$&")+")","ig"),"<mark>$1</mark>");});return h;}
function runSearch(){
  const v=(q.value||"").trim().toLowerCase(); qsel=-1;
  if(v.length<2){qres.hidden=true;qres.innerHTML="";return;}
  const words=v.split(/\s+/).filter(Boolean);
  const hits=INDEX.filter(e=>{const l=e.label.toLowerCase(), t=e.tab.toLowerCase();return words.every(w=>l.includes(w)||t.includes(w));}).slice(0,40);
  qres.innerHTML=hits.length?hits.map((e,i)=>'<button type="button" data-i="'+i+'"><span class="ui-qt">'+esc(e.tab)+' \u203a '+e.kind+'</span>'+mark(e.label,words)+'</button>').join("")
    :'<div class="ui-qnone">nothing in the headings for \u201c'+esc(v)+'\u201d</div>';
  qres.hidden=false;
  $$("button",qres).forEach(b=>b.addEventListener("click",()=>{const e=hits[+b.dataset.i];qres.hidden=true;q.blur();nav.classList.remove("open");
    if(e.kind==="tab"){showTab(e.id);syncNav();}else{openTo(e.id,true);history.replaceState(null,"","#"+e.id);}}));
}
if(q&&qres){
  q.addEventListener("input",runSearch); q.addEventListener("focus",()=>{if(q.value.trim().length>=2)qres.hidden=false;});
  q.addEventListener("keydown",ev=>{const bs=$$("button",qres);
    if(ev.key==="Escape"){qres.hidden=true;q.blur();}
    else if(ev.key==="ArrowDown"||ev.key==="ArrowUp"){if(!bs.length)return;ev.preventDefault();qsel=(qsel+(ev.key==="ArrowDown"?1:-1)+bs.length)%bs.length;bs.forEach((b,i)=>b.classList.toggle("sel",i===qsel));bs[qsel].scrollIntoView({block:"nearest"});}
    else if(ev.key==="Enter"){if(bs.length){(bs[qsel>=0?qsel:0]).click();}}});
  document.addEventListener("click",ev=>{if(!ev.target.closest(".ui-search"))qres.hidden=true;});
  document.addEventListener("keydown",ev=>{if(ev.key==="/"&&!ev.target.closest("input,textarea,[contenteditable]")){ev.preventDefault();q.focus();q.select();}});
}

/* ---------- figure lightbox ---------- */
const lb=document.createElement("div"); lb.id="ui-lb"; lb.hidden=true;
lb.innerHTML='<div class="ui-lbbar"><b></b><button type="button" data-lb="fit">fit / actual size</button><a target="_blank" rel="noopener">open file</a><button type="button" data-lb="close">close &#10005;</button></div><div class="ui-lbin"><img alt=""></div>';
document.body.appendChild(lb);
const lbImg=lb.querySelector("img"), lbCap=lb.querySelector("b"), lbA=lb.querySelector("a");
function lbOpen(img){const f=img.closest("figure");const h=f&&f.querySelector("h4"),n=f&&f.querySelector(".fignum");
  lbImg.src=img.currentSrc||img.src;lbA.href=lbImg.src;lbCap.textContent=(n?figNo(n)+" \u00b7 ":"")+(h?text(h):img.alt||"");
  lb.classList.remove("ui-full");lb.hidden=false;document.body.style.overflow="hidden";}
function lbClose(){lb.hidden=true;document.body.style.overflow="";}
document.addEventListener("click",ev=>{const img=ev.target.closest("figure .imgwrap img, .imgwrap img");if(img&&!ev.target.closest("#ui-lb")){ev.preventDefault();lbOpen(img);}});
lb.addEventListener("click",ev=>{if(ev.target.closest('[data-lb="close"]')||ev.target===lb){lbClose();return;}
  if(ev.target.closest('[data-lb="fit"]')||ev.target.closest(".ui-lbin")){lb.classList.toggle("ui-full");}});
document.addEventListener("keydown",ev=>{if(ev.key==="Escape"){if(!lb.hidden)lbClose();else if(qres&&!qres.hidden){qres.hidden=true;}}});

/* ---------- progress bar, back to top ---------- */
const prog=document.createElement("div");prog.id="ui-prog";document.body.appendChild(prog);
const topBtn=document.createElement("button");topBtn.id="ui-top";topBtn.type="button";topBtn.title="back to top";topBtn.innerHTML="&#8593;";document.body.appendChild(topBtn);
topBtn.addEventListener("click",()=>window.scrollTo({top:0,behavior:"auto"}));
/* the phone nav is sticky: the strip sits under it */
const setNavH=()=>document.documentElement.style.setProperty("--ui-navh",(window.innerWidth<=820?nav.offsetHeight:0)+"px");
window.addEventListener("resize",setNavH); setNavH(); if(navlist)$(".ui-navtog").addEventListener("click",()=>setTimeout(setNavH,0));
})();
</script>"""


def rebuild_nav(s: str) -> str:
    """Group the nav buttons; keep every generator marker and button intact."""
    if B_NAV in s and E_NAV in s:
        i, j = s.index(B_NAV), s.index(E_NAV) + len(E_NAV)
        region = s[i:j]
        if s[j:j + 1] == "\n":      # the newline the first insert added after the marker
            j += 1
    else:
        i = s.index('    <button class="n0', s.index("<nav>"))
        j = s.index('    <div class="foot">', i)
        region = s[i:j]
    entries = []   # (data-t, is_retired, html)
    for m in ENTRY.finditer(region):
        if m.group("wrapped"):
            b = m.group("wbtn"); t = re.search(r'data-t="([^"]+)"', b).group(1); cls = re.search(r'class="([^"]*)"', b).group(1)
            entries.append((t, "retired" in cls.split(), m.group("wrapped").strip()))
        else:
            entries.append((m.group("t"), "retired" in m.group("cls").split(), m.group("btn").strip()))
    seen = {t for t, _, _ in entries}
    by = {t: h for t, _, h in entries}
    retired = [(t, h) for t, r, h in entries if r]
    placed = set()
    out = [B_NAV, '    <div class="ui-search"><input id="ui-q" type="search" placeholder="Search headings\u2026  ( / )" autocomplete="off" spellcheck="false"><div id="ui-qres" hidden></div></div>',
           '    <div class="ui-navlist">']
    for head, ids in GROUPS:
        ids = [t for t in ids if t in seen and t not in {r for r, _ in retired}]
        if not ids:
            continue
        if head:
            out.append(f'    <div class="ui-navh">{head}</div>')
        for t in ids:
            out.append("    " + by[t]); placed.add(t)
    other = [t for t, r, _ in entries if not r and t not in placed]
    if other:
        out.append('    <div class="ui-navh">Other</div>')
        for t in other:
            out.append("    " + by[t]); placed.add(t)
    if retired:
        out.append('    <div class="ui-navh ui-ret"><button type="button" class="ui-rett" aria-expanded="false">'
                   f'<span class="ui-car">&#9656;</span>Retired <span class="ui-n">{len(retired)}</span></button></div>')
        out.append('    <div class="ui-retired" hidden>')
        for t, h in retired:
            out.append("    " + h)
        out.append("    </div>")
    out.append("    </div>")
    out.append("    " + E_NAV)
    return s[:i] + "\n".join(out) + "\n" + s[j:]


def splice(s: str, begin: str, end: str, block: str, anchor: str, before: bool = True) -> str:
    payload = begin + "\n" + block + "\n" + end
    if begin in s and end in s:
        i, j = s.index(begin), s.index(end) + len(end)
        return s[:i] + payload + s[j:]
    k = s.index(anchor)
    return (s[:k] + payload + "\n" + s[k:]) if before else (s[:k + len(anchor)] + "\n" + payload + s[k + len(anchor):])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--insert", action="store_true")
    a = ap.parse_args()
    s = SITE.read_text(encoding="utf-8")
    n0 = len(s)
    s = rebuild_nav(s)
    s = splice(s, B_CSS, E_CSS, CSS, "<!-- BEGIN site gate" if "<!-- BEGIN site gate" in s else "</head>")
    s = splice(s, B_JS, E_JS, JS.replace("__UI_DATA__", ui_data()), "</body>")
    s = re.sub(r"(status of the analysis\. Updated )[0-9\u2011-]+", lambda m: m.group(1) + dt.date.today().isoformat().replace("-", "\u2011"), s, count=1)
    nav = s[s.index("<nav>"):s.index("</nav>")]
    n_btn, n_hdr, n_ret = nav.count("data-t="), nav.count('class="ui-navh'), nav.count(" retired")
    print(f"nav: {n_btn} buttons, {n_hdr} group headers, {n_ret} retired")
    print(f"css {len(CSS):,} chars, js {len(JS):,} chars; page {n0:,} -> {len(s):,} chars")
    if not a.insert:
        print("(pass --insert to write the site)")
        return 0
    SITE.write_text(s, encoding="utf-8")
    print(f"wrote {SITE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
