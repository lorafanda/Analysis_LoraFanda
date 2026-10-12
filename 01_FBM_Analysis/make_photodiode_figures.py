#!/usr/bin/env python3
"""make_photodiode_figures.py - the trigger channel of a patient, and the onsets cut from it.

    python make_photodiode_figures.py --patients PAT_3455 PAT_3965 PAT_3975
    python make_photodiode_figures.py --all

Writes outputs/photodiode_check/<pid>_photodiode.png, which 01_FBM_LM_report.py puts in
the overview PDF as page 2. Two panels: the whole session with the channel's own rail
drawn in, so saturation is visible, and sixty seconds with the prep0 onsets on top, so a
misplaced extraction shows itself.

WHY. PAT_3455 clusters on its own because every one of its channels carries the same
signal. Its diode turned out to rail 37.4% of the time AND to put a -882 uV step on all
116 contacts at every onset. PAT_3975's diode also rails, at 31.1%, and leaks -2.3 uV
into 0 of 218 contacts - so saturation on its own is harmless, and the number that
matters is the step on the BRAIN channels and how much of it varies between them. A step
that is identical everywhere is removed by the white-matter reference; only the spread
survives.

Readers: Micromed .TRC through neo, EDF through a reader written out here. pyedflib is
not installed and is not worth installing (see the 2026-10-09 entry in LORA_ENV2 notes).
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RAW = Path("//nasac-m2.unige.ch/m-HumanNeuronLab/DATARAW")
OUT = HERE / "outputs" / "photodiode_check"
INK, MUTED = "#22272b", "#7c858e"

# Where each patient's recording and trigger table are, and which channel carries the
# trigger. `diff` draws the difference of two channels - PAT_3975's diode is E1-E2 and a
# single channel of it understates the signal.
PATIENTS = {
    "PAT_3455": dict(
        rec="SEEG_EXPERIMENTS_HUG/PAT_3455/task_FBM/data_LM/prep0/"
            "PAT_3455_EEG_950310_FLM&VisCatLoc_anon.edf",
        chans=["photo"], prep="SEEG_EXPERIMENTS_HUG/PAT_3455/task_FBM/data_LM/prep0",
        tsv="PAT_3455_LM_*__triggerPD*.tsv"),
    "PAT_3965": dict(
        rec="SEEG_EXPERIMENTS_HUG/PAT_3965/task_FBM/data_LM/raw/"
            "PAT_3965_EEG_2030033_LM_CatLocVis.TRC",
        chans=["x1"], prep="SEEG_EXPERIMENTS_HUG/PAT_3965/task_FBM/data_LM/prep0",
        tsv="PAT_3965_LM_*__x1_*.tsv"),
    "PAT_3975": dict(
        rec="SEEG_EXPERIMENTS_HUG/PAT_3975/task_FBM/data_LM/raw/"
            "PAT_3975_EEG_1631437_FLM_anon.TRC",
        chans=["E1", "E2"], diff=True,
        prep="SEEG_EXPERIMENTS_HUG/PAT_3975/task_FBM/data_LM/prep0",
        tsv="PAT_3975_LM_*__E2_*.tsv"),
}


def _edf_header(fh):
    fh.seek(0)
    h = fh.read(256)
    n_rec, rec_dur, ns = int(h[236:244]), float(h[244:252]), int(h[252:256])

    def field(size):
        return [fh.read(size).decode("latin-1").strip() for _ in range(ns)]
    labels = field(16)
    fh.read(80 * ns); fh.read(8 * ns)
    pmin = np.array([float(x) for x in field(8)]); pmax = np.array([float(x) for x in field(8)])
    dmin = np.array([float(x) for x in field(8)]); dmax = np.array([float(x) for x in field(8)])
    fh.read(80 * ns)
    nsamp = np.array([int(x) for x in field(8)])
    fh.read(32 * ns)
    return dict(n_rec=n_rec, rec_dur=rec_dur, labels=labels, nsamp=nsamp, pmin=pmin,
                pmax=pmax, dmin=dmin, dmax=dmax, data_off=256 + 256 * ns)


def read_edf(path, want):
    """{label: trace}, fs, all labels - from the header and the one channel asked for."""
    with open(path, "rb") as fh:
        H = _edf_header(fh)
        out = {}
        per_rec = int(H["nsamp"].sum())
        for c in want:
            hit = next((k for k in H["labels"] if k.lower() == c.lower()), None)
            if hit is None:
                continue
            j = H["labels"].index(hit)
            off, n = int(H["nsamp"][:j].sum()), int(H["nsamp"][j])
            fh.seek(H["data_off"])
            raw = np.fromfile(fh, dtype="<i2", count=per_rec * H["n_rec"])
            raw = raw.reshape(H["n_rec"], per_rec)[:, off:off + n].ravel().astype("float64")
            g = (H["pmax"][j] - H["pmin"][j]) / (H["dmax"][j] - H["dmin"][j])
            out[hit] = (raw - H["dmin"][j]) * g + H["pmin"][j]
            fs = n / H["rec_dur"]
    return out, fs, H["labels"]


def read_trc(path, want):
    from neo.rawio import MicromedRawIO
    r = MicromedRawIO(filename=str(path))
    r.parse_header()
    sig = r.header["signal_channels"]
    names = [str(x).strip() for x in sig["name"]]
    fs = float(sig["sampling_rate"][0])
    n = r.get_signal_size(0, 0)
    out = {}
    for c in want:
        hit = next((k for k in names if k.lower() == c.lower()), None)
        if hit is None:
            continue
        j = np.array([names.index(hit)])
        x = r.get_analogsignal_chunk(block_index=0, seg_index=0, i_start=0, i_stop=n,
                                     channel_indexes=j)
        out[hit] = r.rescale_signal_raw_to_float(x, dtype="float64", channel_indexes=j)[:, 0]
    return out, fs, names


def onsets(prep: Path, pattern: str):
    out = []
    for f in sorted(prep.glob(pattern)):
        if f.name.endswith(".bak"):
            continue
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                out.append(float(r["onset"]))
    return np.array(sorted(out))


def figure_for(pid, J):
    rec = RAW / J["rec"]
    if not rec.exists():
        print(f"{pid}: no recording at {rec}")
        return None
    sigs, fs, names = (read_edf if rec.suffix.lower() == ".edf" else read_trc)(rec, J["chans"])
    if not sigs:
        print(f"{pid}: none of {J['chans']} among {names[:6]} ...")
        return None
    if J.get("diff") and len(sigs) == 2:
        k = list(sigs)
        sigs = {f"{k[0]} - {k[1]}": sigs[k[0]] - sigs[k[1]]}
    lab, x = next(iter(sigs.items()))
    on = onsets(RAW / J["prep"], J["tsv"])
    if not len(on):
        print(f"{pid}: no onsets matching {J['tsv']}")
        return None

    t = np.arange(len(x)) / fs
    amax = float(np.abs(x).max())
    rail = float((np.abs(x) >= 0.98 * amax).mean())
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 3.1), dpi=170,
                               gridspec_kw=dict(width_ratios=[2, 1.25]))
    step = max(1, len(x) // 90000)
    a.plot(t[::step], x[::step], lw=0.4, color=MUTED)
    for s in (amax, -amax):
        a.axhline(s, color="#c1121f", lw=0.7, ls=":", alpha=.8)
    a.set_ylabel(f"{pid}\n{lab}", fontsize=9, color=INK)
    a.set_title(f"whole session · {len(x)/fs/60:.0f} min · {100*rail:.1f}% of samples "
                f"at the rail (dotted)", fontsize=9, color=INK, loc="left")
    t0 = float(on[9] - 4) if len(on) > 10 else float(t[0])
    m = (t >= t0) & (t <= t0 + 60)
    b.plot(t[m], x[m], lw=0.7, color=INK)
    for v in on[(on >= t0) & (on <= t0 + 60)]:
        b.axvline(v, color="#1b6ca8", lw=0.9, alpha=.85)
    b.set_title(f"60 s from {t0:.0f} s · blue = the {len(on)} onsets in prep0",
                fontsize=9, color=INK, loc="left")
    for ax in (a, b):
        ax.set_xlabel("seconds", fontsize=9)
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{pid}_photodiode.png"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"{pid}: {lab}, {100*rail:.1f}% at the rail, {len(on)} onsets -> {png.name}")
    return png


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", nargs="*")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    pats = sorted(PATIENTS) if a.all else (a.patients or [])
    if not pats:
        ap.error("give --patients or --all")
    for pid in pats:
        if pid not in PATIENTS:
            print(f"{pid}: no entry - add its recording and trigger table to PATIENTS")
            continue
        figure_for(pid, PATIENTS[pid])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
