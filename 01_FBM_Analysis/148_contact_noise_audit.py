#!/usr/bin/env python3
"""148_contact_noise_audit.py - how noisy is each contact, measured where it is visible.

    python 148_contact_noise_audit.py --patients PAT_3455
    python 148_contact_noise_audit.py --all
    python 148_contact_noise_audit.py --all --report        # re-read the TSVs, no loading

Writes outputs/contact_noise/<pid>_contact_noise.tsv and prints what is over threshold.

WHERE IT IS VISIBLE. Not in the cubes: those are dB re baseline, so a contact that is
equally noisy in the baseline and in the trial normalises to look ordinary - PAT_3455's
IPD14 scored 0.91 on a cube metric while being one of the hairiest traces on the shaft.
Not in the raw either: before re-referencing every channel reads the same because the
shared reference dominates (PAT_3455: sd 957 uV on all 116 channels, to three digits,
because its photodiode was wired against ground). And not in the trial scores: score_hg
is a within-channel z for trial rejection, so every contact sits at 1.00x by construction.

It is visible in the WM-REFERENCED time domain, which is what the Signal rasters draw.
So that is what this measures: broadband sd and 70-150 Hz sd per contact over the task
window, each as a multiple of that patient's own montage median.

READING THE NUMBERS. On PAT_3965 the genuinely broken contacts separate themselves by an
order of magnitude - cmd11 and cmd12 at 28x the median in high gamma, the x/y connector
block at 15-18x, and then nothing until 2.5x. On PAT_3455 the worst is 3.4x. So there is
no single threshold: --thresh only decides what gets PRINTED, the TSV has every contact,
and a contact is worth removing when it stands apart from its own montage, not when it
clears a number chosen here.

The pipeline's own loader is used (140's _load_signals_and_prep_for_patient), so EDF, H5,
TRC, multi-file concatenation and the renames all behave exactly as they do in a real run.
That is also why this has to run where mne is installed.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "outputs" / "contact_noise"
HG_BAND = (70.0, 150.0)


def load_140():
    spec = importlib.util.spec_from_file_location("p140", HERE / "140_ersp_pipeline.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["p140"] = m
    spec.loader.exec_module(m)
    return m


def audit(m, pid):
    from scipy.signal import butter, sosfiltfilt
    cfg = m.cfg
    out = m._load_signals_and_prep_for_patient(pid)
    patient_id, signals, names, fs, prep_dir, etsv, _is_micro = out
    signals = np.asarray(signals)

    wm_idx = m.io.wm_indices_for_patient(patient_id, names, electrodes_tsv_pattern=etsv)
    if not wm_idx:
        print(f"  [{patient_id}] no WM channels - skipped")
        return None
    bad = list(getattr(cfg, "bad_channels_manual", {}).get(patient_id, []) or [])
    ref = signals[:, list(wm_idx)].mean(axis=1)

    # the task window, from the trials the pipeline itself would use; falling back to the
    # whole recording would let a long quiet tail dominate every contact's sd
    try:
        groups = m.tr.collect_trials(prep_dir, fs, outlier_method="IQR", iqr_k=cfg.iqr_k,
                                     patient_id=patient_id, max_post_s=cfg.max_post_s,
                                     min_stim_s=cfg.min_stim_s, min_post_s=cfg.min_post_s)
        on = np.concatenate([np.asarray(v[0]) for v in groups.values()]) if groups else None
        te = np.concatenate([np.asarray(v[2]) for v in groups.values()]) if groups else None
        a, b = (int(max(0, on.min())), int(min(len(signals), te.max()))) if on is not None \
            else (0, len(signals))
    except Exception as e:
        print(f"  [{patient_id}] no trial table ({type(e).__name__}); whole recording")
        a, b = 0, len(signals)

    sos = butter(4, list(HG_BAND), btype="band", fs=fs, output="sos")
    rows = []
    for j, ch in enumerate(names):
        x = signals[a:b, j] - ref[a:b]
        rows.append(dict(patient=patient_id, channel=ch,
                         is_wm=int(j in set(wm_idx)),
                         is_bad=int(m.io.normalize_label(ch) in
                                    {m.io.normalize_label(z) for z in bad}),
                         sd=float(np.std(x)),
                         hg=float(np.std(sosfiltfilt(sos, x)))))
    sd = np.array([r["sd"] for r in rows]); hg = np.array([r["hg"] for r in rows])
    msd, mhg = float(np.median(sd)), float(np.median(hg))
    for r in rows:
        r["sd_x"] = round(r["sd"] / msd, 3) if msd else float("nan")
        r["hg_x"] = round(r["hg"] / mhg, 3) if mhg else float("nan")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{patient_id}_contact_noise.tsv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader(); w.writerows(rows)
    print(f"  [{patient_id}] {len(rows)} contacts, {len(wm_idx)} in the WM mean, "
          f"window {(b-a)/fs:.0f} s | median sd {msd:.1f} uV, HG {mhg:.2f} uV -> {p.name}")
    return rows


def show(rows, thresh):
    rows = sorted(rows, key=lambda r: -max(r["sd_x"], r["hg_x"]))
    hits = [r for r in rows if max(r["sd_x"], r["hg_x"]) >= thresh]
    if not hits:
        print(f"      nothing over {thresh}x")
        return
    for r in hits[:25]:
        tag = (" [WM ref]" if r["is_wm"] else "") + (" [bad-listed]" if r["is_bad"] else "")
        print(f"      {r['channel']:10s} sd {r['sd_x']:6.2f}x  HG {r['hg_x']:6.2f}x{tag}")
    if len(hits) > 25:
        print(f"      ... and {len(hits)-25} more over {thresh}x")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--thresh", type=float, default=2.0, help="what gets printed")
    ap.add_argument("--report", action="store_true",
                    help="re-read the TSVs already written, load nothing")
    a = ap.parse_args()

    if a.report:
        for p in sorted(OUT.glob("*_contact_noise.tsv")):
            rows = [{k: (float(v) if k in ("sd", "hg", "sd_x", "hg_x")
                         else int(v) if k in ("is_wm", "is_bad") else v)
                     for k, v in r.items()}
                    for r in csv.DictReader(open(p, newline=""), delimiter="\t")]
            print(f"\n=== {rows[0]['patient']}  ({len(rows)} contacts)")
            show(rows, a.thresh)
        return 0

    m = load_140()
    pats = a.patients or []
    if a.all:
        pats = [str(x) for x in m.cfg.patient_ids]
    if not pats:
        ap.error("give --patients, --all or --report")
    for pid in pats:
        print(f"\n=== {pid}")
        try:
            rows = audit(m, pid)
        except Exception as e:
            print(f"  [{pid}] FAILED: {type(e).__name__}: {e}")
            continue
        if rows:
            show(rows, a.thresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
