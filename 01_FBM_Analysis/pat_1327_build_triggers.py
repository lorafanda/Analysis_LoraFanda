#!/usr/bin/env python3
"""
pat_1327_build_triggers.py - PAT_1327's trial triggers, from the split photodiode.

    python pat_1327_build_triggers.py            # write the three prep0 tables
    python pat_1327_build_triggers.py --check    # detect and report, write nothing
    python pat_1327_build_triggers.py --plot     # also draw the diagnostic PNG

THE RECORDING. Two Micromed files, joined in this order (cfg.RAW_CONCAT["PAT_1327"]):

    EEG_2935940.TRC   4,440,448 samples @ 2048 Hz = 36.1 min
    EEG_2935950.TRC   1,185,984 samples            =  9.7 min

236 channels, identical in both. The task sits in the LAST THIRD of file 40 and runs to
the end of file 50: the diode is flat for the first 23 minutes and starts at 23.6 min.
Every sample number written here is on the JOINED axis, which is the axis 140 sees once
RAW_CONCAT is set - file 50 begins at sample 4,440,448.

THE PHOTODIODE IS A DIFFERENCE, AND AN ENVELOPE. The diode is split over X1 and X2. Both
channels carry the same fast carrier, which cancels in X1 - X2; what survives is that
carrier AMPLITUDE-MODULATED by the diode, as a step that decays back to zero over 1-4 s
because the input is AC-coupled. So:

  * the diode state is the SIGN of the smoothed difference, not its level;
  * a screen change is a sign flip;
  * running a detector on the raw difference finds the carrier and nothing else - the
    first attempt did exactly that and produced 269 "steps" of pure noise.

A 25 ms mean turns the envelope into a clean two-level signal (steps of 250-900 uV
against a derivative noise floor of ~14 uV), and a Schmitt trigger at +-120 uV reads it.

DOWN IS STIMULUS ON. Each trial is exactly one down edge then one up edge, and the
interval between them is the stimulus duration - which comes out at 1.01 s (sd 0.011) for
picture naming and 3.50 s (sd 0.010) for reading completion. Those are the task's own
constants, the same ones PAT_3415 and PAT_3975 show, so the pairing is not a guess.

A SECOND IS MISSING AT THE FILE JOIN, and that is the whole of the "1 s step". Matched to
the events table the diode agrees to about 1 ms for picture naming and the first 35
auditory trials, and is then a flat -1.000 s off for everything that follows. The step
begins at trial 89 - which is the FIRST TRIAL AFTER THE FILE BOUNDARY - and the headers
say why: file 1 lasts 2168.188 s and file 2 starts 2169 s after it, so Micromed dropped
about 0.8 s (the header clock has 1 s resolution) when it split the recording. The joined
axis is short by that, so everything after the join sits a second early against the
stimulus computer's clock. Nothing moved in the log and nothing drifted (a rate fit gives
0.5 ppm, i.e. nothing).

It costs nothing here, because every onset written below is the diode edge read off the
JOINED axis - the same axis 140 loads through RAW_CONCAT - so the tables are self
consistent whatever the wall clock says. Matching is still done per trial to the nearest
down edge within +-1.5 s, which absorbs the step wherever it falls.

THE ONE TRIAL THE JOIN COSTS. A join is a discontinuity in every channel, and one
auditory trial (onset 2167.61 s) has it inside its epoch. It is dropped, with a line
saying so, rather than left to carry a step into its ERSP.

WHAT trial_end MEANS HERE, the same as everywhere else in this cohort: the end of the
trial window, not the end of the response. In PAT_3415 / PAT_3455 / PAT_3975 it sits a
constant ~1.0-1.2 s before the next onset, so that is the rule used - next onset minus
ITI_S - and the last trial of each block takes its block's median instead.
"""
from __future__ import print_function

import argparse
import datetime as _dt
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "functions"))
import config as cfg                                                  # noqa: E402

PID = "PAT_1327"
RAW = os.path.join(r"\\nasac-m2.unige.ch\m-HumanNeuronLab\DATARAW\SEEG_EXPERIMENTS_HUG",
                   PID, "task_FBM", "data_LM", "raw")
PREP0 = os.path.join(os.path.dirname(RAW), "prep0")
EVENTS = "sub-1327_task-LanguageMapping_datetime-15-9-2026(14h18m24s)_language-FRE_events.tsv"

PD_CHANNELS = ("X1", "X2")       # the diode is the difference of these two
SMOOTH_S = 0.025                 # the mean that turns the modulated carrier into a level
THR_UV = 120.0                   # Schmitt threshold on it; steps are 250-900, noise ~14
MIN_GAP_S = 0.30                 # two screen changes cannot be closer than this
MATCH_WIN_S = 1.5                # how far a log onset may sit from its edge
ITI_S = 1.0                      # trial_end = next onset - this (the cohort's convention)
STRIP_S = 120.0                  # seconds per row of the --plot figure

# the events table's category -> (the pipeline's condition_name, the prep0 file's tag)
COND = {
    "picture_naming":      ("picture_naming", "picture"),
    "auditory_naming_FRE": ("auditory_naming", "audio"),
    "reading_completion":  ("reading_completion", "reading"),
}


def read_pd():
    """X1 - X2 over the joined recording, plus fs and the file boundaries."""
    from neo.rawio import MicromedRawIO
    files = (getattr(cfg, "RAW_CONCAT", {}) or {}).get(PID)
    if not files:
        raise SystemExit("cfg.RAW_CONCAT['%s'] is not set - the two TRC files must be "
                         "declared there, in order, or 140 will read only the first" % PID)
    parts, bounds, fs, off = [], [], None, 0
    for fn in files:
        path = os.path.join(RAW, fn)
        r = MicromedRawIO(filename=path)
        r.parse_header()
        chans = [str(c["name"]).strip() for c in r.header["signal_channels"]]
        missing = [c for c in PD_CHANNELS if c not in chans]
        if missing:
            raise SystemExit("%s: no channel %s" % (fn, ", ".join(missing)))
        idx = np.array([chans.index(c) for c in PD_CHANNELS])
        n = int(r.get_signal_size(block_index=0, seg_index=0))
        raw = r.get_analogsignal_chunk(block_index=0, seg_index=0, i_start=0, i_stop=n,
                                       channel_indexes=idx)
        sig = r.rescale_signal_raw_to_float(raw, dtype="float32", channel_indexes=idx)
        fs_f = float(r.get_signal_sampling_rate())
        if fs is None:
            fs = fs_f
        elif fs_f != fs:
            raise SystemExit("%s: %g Hz, but %s is %g Hz" % (fn, fs_f, files[0], fs))
        parts.append(sig)
        bounds.append((fn, off, off + n))
        off += n
        print("  %-20s %10s samples  (joined axis starts at %s)"
              % (fn, "{:,}".format(n), "{:,}".format(off - n)))
    X = np.vstack(parts)
    return (X[:, 0] - X[:, 1]).astype(np.float64), fs, bounds


def transitions(pd_, fs):
    """(sample, sign) of every screen change, read off the envelope."""
    w = max(1, int(SMOOTH_S * fs))
    sm = np.convolve(pd_, np.ones(w) / w, mode="same")
    state = np.zeros(len(sm), dtype=np.int8)
    state[sm > THR_UV] = 1
    state[sm < -THR_UV] = -1
    idx = np.flatnonzero(state != 0)
    if idx.size == 0:
        raise SystemExit("no photodiode transitions above %g uV - wrong channels?" % THR_UV)
    s_at = state[idx]
    first = np.r_[True, s_at[1:] != s_at[:-1]]
    ev_i, ev_s = idx[first], s_at[first]
    keep = [0]
    for j in range(1, len(ev_i)):
        if (ev_i[j] - ev_i[keep[-1]]) / fs >= MIN_GAP_S:
            keep.append(j)
    return ev_i[keep], ev_s[keep]


def build(check=False, plot=False):
    print("[pd] reading %s and %s" % PD_CHANNELS)
    pd_, fs, bounds = read_pd()
    n_tot = len(pd_)
    print("  joined: {:,} samples @ {:g} Hz = {:.1f} min".format(n_tot, fs, n_tot / fs / 60))

    ev_i, ev_s = transitions(pd_, fs)
    t = ev_i / fs
    dn_i, up_i = ev_i[ev_s < 0], ev_i[ev_s > 0]
    print("[pd] %d transitions (%d down, %d up), %.1f -> %.1f min"
          % (len(ev_i), len(dn_i), len(up_i), t[0] / 60, t[-1] / 60))

    log = pd.read_csv(os.path.join(RAW, EVENTS), sep="\t", skiprows=1)
    log = log.sort_values("onset").reset_index(drop=True)
    print("[log] %d trials: %s" % (len(log), dict(log.category.value_counts())))

    # the constant that puts the log on the recording clock: the offset that lands the
    # most onsets on a down edge. Scanned, not assumed.
    dn_t = dn_i / fs
    best = (None, -1)
    for cand in dn_t[:80] - log.onset.values[0]:
        j = np.clip(np.searchsorted(dn_t, log.onset.values + cand), 1, len(dn_t) - 1)
        res = np.minimum(np.abs(dn_t[j] - (log.onset.values + cand)),
                         np.abs(dn_t[j - 1] - (log.onset.values + cand)))
        hits = int((res < 0.05).sum())
        if hits > best[1]:
            best = (cand, hits)
    offset = best[0]
    print("[fit] log -> recording offset %.3f s (%d/%d onsets within 50 ms of a down edge)"
          % (offset, best[1], len(log)))

    rows, unmatched = [], []
    for i, r in log.iterrows():
        want = r.onset + offset
        j = int(np.argmin(np.abs(dn_t - want)))
        if abs(dn_t[j] - want) > MATCH_WIN_S:
            unmatched.append(i)
            continue
        on = int(dn_i[j])
        k = int(np.searchsorted(up_i, on))
        if k >= len(up_i):
            unmatched.append(i)
            continue
        cond_name, tag = COND[r.category]
        rows.append(dict(sample=on, sample_offsets=int(up_i[k]), condition_name=cond_name,
                         tag=tag, resp_accuracy=str(r.response_type).strip(),
                         trial_idx=int(r.exemplar), res_s=dn_t[j] - want))
    tr = pd.DataFrame(rows).sort_values("sample").reset_index(drop=True)
    if unmatched:
        print("[warn] %d log trials with no photodiode edge: %s" % (len(unmatched), unmatched))
    if tr["sample"].duplicated().any():          # tr.sample is the DataFrame method
        raise SystemExit("the same edge was used for two trials - matching is wrong")

    tr["onset"] = tr["sample"] / fs
    tr["onset_duration"] = (tr.sample_offsets - tr["sample"]) / fs
    print("[chk] residual to the log: median %.1f ms | the 1 s step starts at trial %s"
          % (np.median(tr.res_s) * 1000,
             int(np.argmax(tr.res_s.values < -0.5)) if (tr.res_s < -0.5).any() else "-"))
    print("[chk] stimulus duration per block (the task's own constants):")
    for c, g in tr.groupby("condition_name"):
        print("        %-20s n=%3d  min %.2f  med %.2f  max %.2f  sd %.3f s"
              % (c, len(g), g.onset_duration.min(), g.onset_duration.median(),
                 g.onset_duration.max(), g.onset_duration.std()))

    # trial_end: the next onset of the same block, less the inter-trial interval
    out = []
    for cond_name, g in tr.groupby("condition_name", sort=False):
        g = g.sort_values("sample").reset_index(drop=True)
        nxt = g["sample"].shift(-1)
        end = (nxt - ITI_S * fs).round()
        post = (end - g.sample_offsets) / fs
        end.iloc[-1] = round(g.sample_offsets.iloc[-1] + np.nanmedian(post) * fs)
        g["trial_end"] = end.astype(np.int64)
        bad = g.trial_end < g.sample_offsets
        if bad.any():                      # a block gap or a very long stimulus
            g.loc[bad, "trial_end"] = g.loc[bad, "sample_offsets"] + int(np.nanmedian(post) * fs)
            print("[fix] %s: %d trial_end(s) fell before the stimulus offset, set to the "
                  "block median instead" % (cond_name, int(bad.sum())))
        out.append(g)
    tr = pd.concat(out, ignore_index=True).sort_values("sample").reset_index(drop=True)

    # A FILE JOIN IS A STEP IN EVERY CHANNEL. A trial whose epoch contains one would carry
    # that step into its ERSP, so it goes - named, not silently.
    joins = [b[1] for b in bounds[1:]]
    pre = abs(float(getattr(cfg, "baseline_w", (-0.6, -0.1))[0])) * fs
    spans = np.zeros(len(tr), dtype=bool)
    for j in joins:
        spans |= ((tr["sample"] - pre) < j) & (tr.trial_end > j)
    if spans.any():
        for _, r in tr[spans].iterrows():
            print("[drop] %s trial at %.2f s: its epoch spans the join at %.2f s"
                  % (r.condition_name, r.onset, joins[0] / fs))
        tr = tr[~spans].reset_index(drop=True)

    post_s = (tr.trial_end - tr.sample_offsets) / fs
    print("[chk] post-stimulus window: min %.2f  med %.2f  max %.2f s   "
          "(the filters keep %.1f-%g s)"
          % (post_s.min(), post_s.median(), post_s.max(), cfg.min_post_s, cfg.max_post_s))
    keep = ((tr.onset_duration >= cfg.min_stim_s) & (post_s >= cfg.min_post_s) &
            (post_s <= cfg.max_post_s) & (tr.resp_accuracy.str.lower() == "correct"))
    print("[chk] would survive 140's hard filters (before the IQR rule): %d of %d"
          % (int(keep.sum()), len(tr)))
    for c, g in tr.assign(keep=keep).groupby("condition_name"):
        print("        %-20s %d of %d" % (c, int(g.keep.sum()), len(g)))

    cols = ["onset", "onset_duration", "sample", "sample_offsets", "trial_end",
            "condition_name", "resp_accuracy", "trial_idx"]
    stamp = _dt.date.today().isoformat()
    if check:
        print("\n--check: nothing written")
        return tr
    if not os.path.isdir(PREP0):
        os.makedirs(PREP0)
    for tag, g in tr.groupby("tag"):
        p = os.path.join(PREP0, "%s_LM_%s__X1minusX2_%s.tsv" % (PID, tag, stamp))
        g[cols].sort_values("sample").to_csv(p, sep="\t", index=False)
        print("wrote %s  (%d trials)" % (p, len(g)))

    if plot:
        _plot(pd_, fs, tr, bounds)
    return tr


def _plot(pd_, fs, tr, bounds, strip_s=STRIP_S):
    """The whole concatenated experiment, in 140's own style.

    Same reading as the notebook's PD figure: a dashed line and a labelled marker at each
    onset, and the onset-to-offset span shaded, so a trial whose duration is wrong shows
    up as a span the wrong width. Drawn in strips of strip_s seconds because 160 trials
    over 21 minutes cannot be read on one axis, and the join between the two TRC files is
    marked where it falls.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    w = max(1, int(SMOOTH_S * fs))
    sm = np.convolve(pd_, np.ones(w) / w, mode="same")
    colour = {"picture_naming": "tab:green", "auditory_naming": "tab:blue",
              "reading_completion": "tab:purple"}
    shade = {"picture_naming": (0.8, 1.0, 0.8, 0.30), "auditory_naming": (0.8, 0.88, 1.0, 0.30),
             "reading_completion": (0.92, 0.84, 1.0, 0.30)}

    t0_all = tr.onset.min() - 8.0
    t1_all = (tr.trial_end.max() / fs) + 8.0
    n = int(np.ceil((t1_all - t0_all) / strip_s))
    join_s = [b[1] / fs for b in bounds[1:]]

    fig, axes = plt.subplots(n, 1, figsize=(30, 3.2 * n))
    if n == 1:
        axes = [axes]
    for k, ax in enumerate(axes):
        a, b = t0_all + k * strip_s, t0_all + (k + 1) * strip_s
        s0, s1 = int(a * fs), min(int(b * fs), len(sm))
        ax.plot(np.arange(s0, s1) / fs, sm[s0:s1], lw=0.9, color="#444444")
        y0, y1 = np.nanmin(sm[s0:s1]), np.nanmax(sm[s0:s1])
        span = (y1 - y0) or 1.0
        top = y1 + 0.06 * span
        for j in join_s:
            if a <= j < b:
                ax.axvline(j, color="k", lw=2.0)
                ax.text(j, top, " file 2 starts", fontsize=9, va="bottom", ha="left")
        for i, r in tr.iterrows():
            on_t, off_t = r["sample"] / fs, r.sample_offsets / fs
            if not (a <= on_t < b):
                continue
            c = colour.get(r.condition_name, "tab:red")
            if str(r.resp_accuracy).strip().lower() != "correct":
                c = "red"
            ax.axvline(on_t, ls="--", lw=1.25, color=c, alpha=0.9)
            ax.axvline(off_t, ls=":", lw=1.0, color=c, alpha=0.7)
            ax.plot([on_t], [top], marker="v", ms=6, color=c)
            ax.text(on_t, top, "%d:%s" % (i, r.condition_name.split("_")[0][:4]),
                    ha="center", va="bottom", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=c, lw=0.8, alpha=0.9))
            ax.axvspan(on_t, min(off_t, b), color=shade.get(r.condition_name), lw=0)
        ax.set_xlim(a, b)
        ax.set_ylim(y0 - 0.06 * span, top + 0.16 * span)
        ax.set_ylabel("X1-X2, %g ms mean (uV)" % (SMOOTH_S * 1000), fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.locator_params(axis="x", nbins=40)
    axes[0].legend(handles=[Line2D([0], [0], color=v, lw=2, ls="--", label=k_)
                            for k_, v in colour.items()]
                   + [Line2D([0], [0], color="red", lw=2, ls="--", label="incorrect response"),
                      Line2D([0], [0], color="k", lw=2, ls=":", label="offset (stimulus end)")],
                   loc="upper right", ncol=5, framealpha=0.9, fontsize=9)
    axes[0].set_title("%s - the full concatenated experiment: %d trials, onsets (dashed), "
                      "offsets (dotted), stimulus shaded" % (PID, len(tr)), fontsize=12)
    axes[-1].set_xlabel("time on the joined axis (s)")
    fig.tight_layout()
    p = os.path.join(PREP0, "%s_LM_photodiode_check.png" % PID)
    fig.savefig(p, dpi=100)
    plt.close(fig)
    print("wrote %s  (%d strips of %g s)" % (p, n, strip_s))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true", help="report, write nothing")
    ap.add_argument("--plot", action="store_true", help="also write the diagnostic PNG")
    a = ap.parse_args()
    build(check=a.check, plot=a.plot)
