#!/usr/bin/env python
"""140 with the trials 150 marked REMOVED FOR EVERY CHANNEL, written to a TEST tree - nothing in
outputs/03_ERSP changes.

    python .\150_trial_crosschannel.py --patient PAT_3455       (first: the marks and the trial times)
    python .\140_crosschannel_test.py --patient PAT_3455        (then: 140 without those trials)
    python .\140_crosschannel_test.py --patient PAT_3455 --k 2.5
                                      (mark again at another k from the 150 table - arithmetic, no recomputation)
    python .\140_crosschannel_test.py --patient PAT_3455 --r 0.4
                                      (a fixed cut on r_mean instead, the same for every condition)
    python .\140_crosschannel_test.py --patient PAT_3455 --trials audio:7,14 picture:35,2 reading:38
                                      (your own list, the HFA figure's trial numbers)
    python .\140_crosschannel_test.py --patient PAT_3455 --no-psd   (anything after the patient goes to 140)

WHAT (2026-10-15, Lora): the ERSPs and the HFA rasters with the trials that look the same on
every channel taken out - for all channels at once, not channel by channel as the rule in use
does. The trials come from outputs/_trial_crosschannel/<pid>/<pid>_<cond>_trial_crosschannel.tsv
(150's run): the ones it marked, or marked again at --k (r_mean > median + k * 1.4826 * MAD over
the condition's trials), or the list you give.

HOW: nothing new in 140. Each trial to remove becomes a 20 ms span just after its onset in
cfg.bad_time_spans[<pid>], the rule that already removes trials over a discharge: collect_trials
drops a trial whose [onset - baseline, trial_end] touches a span, the HFA figure draws it at the
bottom tagged "discharge: cross-channel r_mean 0.77", and the ERSP, the cubes, the halves and
the trial scores never see it. Every span is checked against the other trials of the table so
no neighbour is caught. The whole output tree goes to --out (default outputs/_crosschannel_test),
the reference-contact (NOREF) figures are skipped, and the per-channel rejection in use stays
on top of this. Run it with the same Python as 140.
"""
import argparse
import os
import runpy
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from functions import config as cfg          # noqa: E402  (the same module object 140 imports)

CONDS = ("audio", "picture", "reading")

ap = argparse.ArgumentParser()
ap.add_argument("--patient", required=True)
ap.add_argument("--k", type=float, default=None,
                help="mark again at this k (r_mean > median + k MAD) instead of taking 150's marks")
ap.add_argument("--r", type=float, default=None,
                help="a fixed cut instead: remove every trial with r_mean above this (e.g. 0.4), same for all conditions")
ap.add_argument("--trials", nargs="*", default=None, metavar="COND:N,N",
                help="your own list instead, e.g. audio:7,14 picture:35 (HFA figure numbers)")
ap.add_argument("--src", default=os.path.join("outputs", "_trial_crosschannel"), help="where 150 wrote its tables")
ap.add_argument("--out", default=os.path.join("outputs", "_crosschannel_test"),
                help="test output root; the tree is <out>/<cfg.ERSP_TREE>/<pid>/LM/ (default outputs/_crosschannel_test)")
a, passthrough = ap.parse_known_args()
pid = a.patient

# ---- the 150 tables
tabs = {}
for cond in CONDS:
    f = os.path.join(a.src, pid, f"{pid}_{cond}_trial_crosschannel.tsv")
    if os.path.exists(f):
        tabs[cond] = pd.read_csv(f, sep="\t")
if not tabs:
    sys.exit(f"no 150 tables for {pid} under {a.src} - run 150_trial_crosschannel.py --patient {pid} first")
for cond, t in tabs.items():
    if not {"onset_s", "tend_s", "r_mean", "trial_label"} <= set(t.columns):
        sys.exit(f"{cond}: the 150 table has no trial times (made before 2026-10-15 17:45) - rerun 150 first")

# ---- which trials go
chosen = {}
if a.trials:
    for item in a.trials:
        cond, _, nums = item.partition(":")
        if cond not in tabs or not nums:
            sys.exit(f"--trials: '{item}' is not cond:N,N with cond among {sorted(tabs)}")
        want = [int(x) for x in nums.split(",") if x.strip()]
        t = tabs[cond]
        missing = sorted(set(want) - set(t.trial_label.astype(int)))
        if missing:
            sys.exit(f"{cond}: trial(s) {missing} are not in the 150 table (kept trials only, HFA numbers)")
        chosen[cond] = t[t.trial_label.isin(want)]
    how = "your list"
elif a.r is not None:
    for cond, t in tabs.items():
        chosen[cond] = t[t.r_mean > float(a.r)]
    how = f"r_mean > {a.r:g}"
elif a.k is not None:
    for cond, t in tabs.items():
        r = t.r_mean.to_numpy(float)
        ok = np.isfinite(r)
        med = float(np.median(r[ok]))
        mad = 1.4826 * float(np.median(np.abs(r[ok] - med)))
        thr = med + float(a.k) * mad if mad > 1e-9 else np.inf
        chosen[cond] = t[ok & (r > thr)]
    how = f"r_mean > median + {a.k:g} MAD"
else:
    for cond, t in tabs.items():
        chosen[cond] = t[t.marked == 1]
    how = "150's marks"

# ---- one 20 ms span just after each onset; no other trial of the table may touch it
pre = abs(float(cfg.baseline_w[0]))            # collect_trials counts a trial from onset - baseline to its end
spans = []
for cond, rows in chosen.items():
    t = tabs[cond]
    for r in rows.itertuples():
        s0, s1 = float(r.onset_s) + 0.01, float(r.onset_s) + 0.03
        others = t[t.trial_label != r.trial_label]
        hit = others[(others.onset_s - pre < s1) & (others.tend_s > s0)]
        if len(hit):
            sys.exit(f"{cond} trial {int(r.trial_label)}: the span {s0:.3f}-{s1:.3f} s would also take trial(s) "
                     f"{[int(x) for x in hit.trial_label]} (their windows overlap here) - tell me")
        spans.append((round(s0, 3), round(s1, 3), f"cross-channel r_mean {float(r.r_mean):.2f}"))

print(f"[crosschannel test] {pid}: trials removed for every channel ({how}):")
for cond in CONDS:
    if cond in chosen:
        rows = chosen[cond].sort_values("trial_label")
        print(f"   {cond:<8} {len(rows):>2} of {len(tabs[cond])}:  "
              + (" ".join(f"{int(x.trial_label)} (r {x.r_mean:.2f})" for x in rows.itertuples()) or "none"))
if not spans:
    print("[crosschannel test] nothing to remove - this run would equal the real one; stopping")
    sys.exit(0)

existing = list((getattr(cfg, "bad_time_spans", {}) or {}).get(pid, []))
if not hasattr(cfg, "bad_time_spans") or cfg.bad_time_spans is None:
    cfg.bad_time_spans = {}
cfg.bad_time_spans[pid] = existing + spans
out_root = os.path.abspath(a.out)
os.makedirs(out_root, exist_ok=True)
cfg.outputs_root = out_root
cfg.write_wm_noref_figures = False
print(f"[crosschannel test] {len(spans)} spans added to cfg.bad_time_spans['{pid}'] for this process "
      f"({len(existing)} were there); outputs_root -> {out_root}   (the real tree is not written)")

sys.argv = ["140_ersp_pipeline.py", "--patient", pid, *passthrough]
runpy.run_path(os.path.join(HERE, "140_ersp_pipeline.py"), run_name="__main__")
