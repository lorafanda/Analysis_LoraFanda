"""
el038_rebuild_triggers.py - EL038's LM trigger tables from the FULL psychopy log and DC6.

WHY. The events file under task_FBM/data_LM/raw is a pruned copy of the session log: 141
of 160 rows. The 19 missing rows are the 3 practice pictures, 14 auditory trials (log
onsets 452-579 s = recording 556-684 s) and the first 2 reading trials (843, 851 s). They
were pruned because DC6 drifts to its bright level during 607-640 s and 663-677 s, the
2025-02 detector lost the pulses there, and the preset then dropped the whole stretch
(fake_trials 77-93, invalid_trials 104-106) so that pulses and log rows lined up
positionally. The full log (the parent folder's sub_EL038_..._events.tsv) has every
trial with its latency and accuracy, and DC6 still carries 10 of the 14 audio pulses
with exemplar-exact lengths.

WHAT IT WRITES. One table per condition, same columns as before (onset, onset_duration,
sample, sample_offsets, trial_end, condition_name, resp_accuracy, trial_idx) plus
`source`:
    DC6            onset and offset are DC6 edges (a pulse within 0.3 s of the mapped log onset)
    log+exemplar   no usable DC6 pulse (the drift stretch): onset = log onset mapped to the
                   recording (linear fit on the DC6-matched trials, residual < 25 ms),
                   offset = onset + the exemplar's sound length measured on DC6 in
                   EL046/EL043/EL034/EL045 (they agree to 20 ms), or 1.00 s / 3.50 s for
                   picture / reading
trial_end = sample_offsets + log duration * fs, as the parser does. The 3 practice pictures
(before the block, no DC6 pulse) stay out.

    python el038_rebuild_triggers.py            # dry run: report + check figure, nothing written
    python el038_rebuild_triggers.py --apply    # old tables -> prep0/prep0-bad/, new tables written

Then `python 140_ersp_pipeline.py --patient EL038` (NOT --pd: the preset's fake / invalid lists
describe the old detection and would regress this).
"""
import argparse
import glob
import os
import shutil
import sys
import time

import h5py
import numpy as np
import pandas as pd

BERN = r"\\nasac-m2.unige.ch\m-HumanNeuronLab\DATARAW\SEEG_EXPERIMENTS_BERN"
LOG = os.path.join(BERN, "EL038", "sub_EL038_task_LanguageMapping_timestamp_12_11_2024_9h30m19s_lang_GER_events.tsv")
H5 = os.path.join(BERN, "EL038", "task_FBM", "data_LM", "raw", "EL038_20241112_HUG_LM.h5")
PREP0 = os.path.join(BERN, "EL038", "task_FBM", "data_LM", "prep0")
REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "04_ersp_LM", "EL038", "LM", "Report")
REF_PATIENTS = ("EL046", "EL043", "EL034", "EL045")
FS = 1024.0
MATCH_S = 0.3
WIN_S = 40.0
COND = {"picture_naming": "picture", "auditory_naming_GER": "audio", "reading_completion": "reading"}
FIXED_LEN = {"picture": 1.00, "reading": 3.50}


def bright_segments(x, fs):
    """(start_s, end_s) of DC6-bright stretches, threshold per WIN_S window so a DC drift
    does not hide a whole block; a flat window gets the global threshold."""
    n, w = len(x), int(WIN_S * fs)
    g5, g95 = np.percentile(x, [5, 95])
    hi = np.zeros(n, dtype=bool)
    for s in range(0, n, w):
        seg = x[s:s + w]
        p5, p95 = np.percentile(seg, [5, 95])
        thr = 0.5 * (p5 + p95) if (p95 - p5) > 0.2 * (g95 - g5) else 0.5 * (g5 + g95)
        hi[s:s + w] = seg > thr
    d = np.diff(np.r_[0, hi.astype(int), 0])
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return st / fs, en / fs


def exemplar_lengths():
    cols = []
    for p in REF_PATIENTS:
        t = pd.concat([pd.read_csv(f, sep="\t") for f in glob.glob(os.path.join(BERN, p, "task_FBM", "data_LM", "prep0", f"{p}_LM_*__DC6_*.tsv"))])
        t = t[t.condition_name.str.contains("audit", case=False)].copy()
        t["len"] = (t.sample_offsets - t["sample"]) / 1024.0
        cols.append(t.groupby("trial_idx")["len"].median().rename(p))
    return pd.concat(cols, axis=1).median(axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    ev = pd.read_csv(LOG, sep="\t", skiprows=1)
    ev["cond"] = ev.category.map(COND)
    assert ev.cond.notna().all(), ev.category.unique()
    practice = ev.onset < 40                      # 3 pictures before the block, no DC6 pulse
    print(f"log: {len(ev)} rows {ev.cond.value_counts().to_dict()}; {int(practice.sum())} practice pictures left out")
    ev = ev[~practice].reset_index(drop=True)

    with h5py.File(H5, "r") as f:
        x = f["traces/raw/DC6"][()].astype(np.float32)
        dur = float(f["meta"].attrs.get("duration", len(x) / FS))
    fs = len(x) / dur
    assert abs(fs - FS) < 1, fs
    st, en = bright_segments(x, FS)

    # log -> recording map from the trials with a clean pulse, refined once
    a_, b_ = 1.0, 104.578
    for _ in range(2):
        rec = a_ * ev.onset.values + b_
        j = np.array([np.argmin(np.abs(st - r)) for r in rec])
        ok = np.abs(st[j] - rec) < MATCH_S
        a_, b_ = np.polyfit(ev.onset.values[ok], st[j][ok], 1)
    rec = a_ * ev.onset.values + b_
    j = np.array([np.argmin(np.abs(st - r)) for r in rec])
    ok = np.abs(st[j] - rec) < MATCH_S
    resid = st[j][ok] - rec[ok]
    print(f"map: recording = {a_:.6f} * log + {b_:.3f}; {int(ok.sum())} of {len(ev)} trials on a DC6 pulse, "
          f"residual median {np.median(np.abs(resid)) * 1000:.0f} ms, max {np.max(np.abs(resid)) * 1000:.0f} ms")

    ref = exemplar_lengths()
    rows = []
    for i, r in ev.iterrows():
        if ok[i]:
            on_s, off_s, src = st[j[i]], en[j[i]], "DC6"
            if r.cond == "audio" and abs((off_s - on_s) - ref.get(int(r.exemplar), np.nan)) > 0.1:
                # a pulse cut by the drift: keep the DC6 onset, take the exemplar length
                off_s, src = on_s + ref[int(r.exemplar)], "DC6+exemplar"
        else:
            on_s = rec[i]
            length = ref[int(r.exemplar)] if r.cond == "audio" else FIXED_LEN[r.cond]
            off_s, src = on_s + length, "log+exemplar"
        on, off = int(round(on_s * FS)), int(round(off_s * FS))
        rows.append(dict(onset=round(on / FS, 6), onset_duration=round((off - on) / FS, 6), sample=on, sample_offsets=off,
                         trial_end=off + int(round(float(r.duration) * FS)), condition_name=r.category,
                         resp_accuracy=r.response_type, trial_idx=int(r.exemplar), source=src))
    new = pd.DataFrame(rows)
    print("\nsource of the rows:", new.groupby(["condition_name", "source"]).size().to_dict())
    print(new[new.source != "DC6"][["condition_name", "trial_idx", "onset", "onset_duration", "resp_accuracy", "source"]].round(2).to_string(index=False))

    # the 2025 tables' rows stay as they are (their edge convention differs from this
    # detector by a few ms); the old-minus-new difference measured on them calibrates the
    # edges of the rows that are new
    old = pd.concat([pd.read_csv(f, sep="\t") for f in glob.glob(os.path.join(PREP0, "*.tsv"))])
    old["t"] = old["sample"] / FS
    d_on, d_off, hit = [], [], np.zeros(len(new), dtype=bool)
    for i, r in new.iterrows():
        cand = old[(old.condition_name == r.condition_name) & ((old.t - r.onset).abs() < 0.05)]
        if len(cand) == 1:
            o = cand.iloc[0]
            d_on.append(o["sample"] - r["sample"]); d_off.append(o.sample_offsets - r.sample_offsets)
            new.loc[i, ["sample", "sample_offsets", "trial_end"]] = [int(o["sample"]), int(o.sample_offsets), int(o.trial_end)]
            new.loc[i, "source"] = "2025 table"; hit[i] = True
    print(f"\nold rows kept verbatim: {int(hit.sum())} of {len(old)}; their onset edge sits {np.median(d_on):+.0f} samples "
          f"(IQR {np.percentile(d_on, 25):+.0f}..{np.percentile(d_on, 75):+.0f}) and the offset edge {np.median(d_off):+.0f} samples "
          f"from this detector's -> applied to the new DC6 rows")
    sh_on, sh_off = int(round(np.median(d_on))), int(round(np.median(d_off)))
    for i, r in new[~hit].iterrows():
        if r.source.startswith("DC6"):
            new.loc[i, "sample"] = int(r["sample"] + sh_on)
            new.loc[i, "sample_offsets"] = int(r.sample_offsets + (sh_off if r.source == "DC6" else sh_on))
            new.loc[i, "trial_end"] = int(new.loc[i, "sample_offsets"] + int(round(float(ev.duration.iloc[i]) * FS)))
    new["onset"] = (new["sample"] / FS).round(6); new["onset_duration"] = ((new.sample_offsets - new["sample"]) / FS).round(6)
    assert (new.sample_offsets > new["sample"]).all() and (new.trial_end >= new.sample_offsets).all()
    print("rows that are new:"); print(new[~hit][["condition_name", "trial_idx", "onset", "onset_duration", "resp_accuracy", "source"]].round(2).to_string(index=False))
    print(f"\nnew tables: {new.groupby('condition_name').size().to_dict()}  (old: {old.groupby('condition_name').size().to_dict()})")

    # check figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(REPORT, exist_ok=True)
        t = np.arange(len(x)) / FS
        fig, axes = plt.subplots(3, 1, figsize=(22, 12))
        axes[0].plot(t[::8], x[::8], lw=0.3, color="k")
        for c, col in (("audio", "tab:red"), ("picture", "tab:blue"), ("reading", "tab:green")):
            g = new[new.condition_name.map(COND) == c]
            axes[0].vlines(g.onset, x.min(), x.max(), color=col, lw=0.5, alpha=0.7, label=f"{c} ({len(g)})")
        axes[0].legend(loc="upper right", fontsize=8); axes[0].set_title("EL038 DC6 with the rebuilt onsets", loc="left")
        for ax, (lo, hi) in zip(axes[1:], ((540, 700), (830, 940))):
            a0, b0 = int(lo * FS), int(hi * FS)
            ax.plot(t[a0:b0], x[a0:b0], lw=0.5, color="k")
            for _, r in new[(new.onset > lo) & (new.onset < hi)].iterrows():
                col = "tab:red" if r.source == "DC6" else "tab:purple"
                ax.axvline(r.onset, color=col, lw=1.3); ax.axvline(r.onset + r.onset_duration, color=col, lw=1.3, ls="--")
                ax.text(r.onset + 0.1, x.max(), f"ex{r.trial_idx} {r.resp_accuracy}\n{r.source}", fontsize=7, color=col, va="top")
            ax.set_title(f"{lo}-{hi} s: red = DC6 edges, purple = log-mapped onset + exemplar length (dashed = offset)", loc="left", fontsize=9)
        p = os.path.join(REPORT, "EL038_triggers_rebuilt_PDcheck.png")
        fig.savefig(p, dpi=110, facecolor="white", bbox_inches="tight"); print(f"\ncheck figure -> {p}")
    except Exception as e:
        print(f"(no figure: {e})")

    if not a.apply:
        print("\nDRY RUN - nothing written. Re-run with --apply.")
        return 0
    bad = os.path.join(PREP0, "prep0-bad")
    os.makedirs(bad, exist_ok=True)
    for f in glob.glob(os.path.join(PREP0, "*.tsv")):
        shutil.move(f, os.path.join(bad, os.path.basename(f)))
        print(f"parked {os.path.basename(f)} -> prep0-bad/")
    stamp = time.strftime("%Y-%m-%d")
    for cat, short in COND.items():
        g = new[new.condition_name == cat].sort_values("sample")
        out = os.path.join(PREP0, f"EL038_LM_{short}__DC6_{stamp}.tsv")
        g.to_csv(out, sep="\t", index=False)
        print(f"wrote {os.path.basename(out)}: {len(g)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
