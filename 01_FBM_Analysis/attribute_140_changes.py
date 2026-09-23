#!/usr/bin/env python3
"""
attribute_140_changes.py - WHY each patient's cubes changed between the two 140 trees.

compare_140_trees.py measures HOW MUCH changed. This asks where the change lives and
what produced it, from evidence that exists in BOTH trees:

  the difference SPECTRUM   mean |new - old| per frequency bin, per patient x condition.
                            A comb at 50 Hz and its harmonics is the notch; a broad,
                            smooth rise is the reference; a flat offset across everything
                            is the trials that went into the average.
  the difference over TIME  the same per warped time bin - a change confined to the post
                            half is the response window, one spread over the whole axis
                            is preprocessing.
  the old run DATE          the `_old` tree is not one run: it is 2026-08-21 for most
                            patients with reruns layered on top to 09-17. A patient whose
                            old cubes are from August has every change since August in it.
  trials                    Report/<pid>_IQR.tsv in both trees: n_in and n_kept per
                            condition. Equal counts mean the trials are not the cause.
  the notch                 Report/<pid>_notch_audit.tsv in both trees.
  the reference             wm_reref_report.tsv, where the tree still has the patient's
                            row (the old file was overwritten per batch, so it is partial).

    python attribute_140_changes.py

Outputs -> outputs/compare_140/
    spectra.npz              per patient x condition: |Δ| per frequency and per time bin
    attribution.tsv          one row per patient: old run date, trials, notch, reference
    D2_why.png               the difference spectra, one panel per patient
"""
from __future__ import annotations

import datetime as dt
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
CMP = OUTPUTS / "compare_140"
FIGS = CMP / "figures"
NEW_RAW, OLD_RAW = OUTPUTS / "04_ersp_LM_RAWONLY", OUTPUTS / "04_ersp_LM_RAWONLY_old"
NEW_QC, OLD_QC = OUTPUTS / "04_ersp_LM", OUTPUTS / "04_ersp_LM_old"
CONDS = ("audio", "picture", "reading")
DF_HZ = 1000.0 / 256.0
TOL = 0.01          # dB. Below this a cube is "numerically the same"; see the report.
INK, MUTED, RED, GREEN, BLUE = "#1b232c", "#68727d", "#c1121f", "#1b7837", "#2471a3"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 7.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.edgecolor": "#c9d1d9", "figure.facecolor": "white"})


def one(args):
    pn, po = args
    try:
        A = np.load(pn).astype(float)
        B = np.load(po).astype(float)
    except Exception:
        return None
    nf = min(A.shape[0], B.shape[0])
    if A.shape[1] != B.shape[1]:
        return None
    D = np.abs(A[:nf] - B[:nf])
    return np.nanmean(D, axis=1), np.nanmean(D, axis=0), np.nanmean(np.abs(B[nf:]), axis=1) if B.shape[0] > nf else None


def spectra() -> dict:
    """mean |Δ| per frequency bin and per time bin, averaged over a patient's contacts."""
    out = {}
    for pdir in sorted(p for p in NEW_RAW.iterdir() if p.is_dir()):
        pid = pdir.name
        for cond in CONDS:
            dn, do = pdir / "LM" / "ERSP_matrix" / cond, OLD_RAW / pid / "LM" / "ERSP_matrix" / cond
            if not dn.is_dir() or not do.is_dir():
                continue
            jobs = []
            for f in dn.glob("*.npy"):
                el = re.search(r"_ERSP_(.+?)_TN\.npy$", f.name)
                if not el:
                    continue
                hit = list(do.glob(f"{pid}_{cond}_*_ERSP_{el.group(1)}_TN.npy"))
                if hit:
                    jobs.append((str(f), str(hit[0])))
            if not jobs:
                continue
            F, T, X = [], [], []
            with ThreadPoolExecutor(12) as ex:
                for r in ex.map(one, jobs):
                    if r is None:
                        continue
                    F.append(r[0]); T.append(r[1])
                    if r[2] is not None:
                        X.append(r[2])
            if F:
                out[f"{pid}|{cond}|f"] = np.mean(F, 0)
                out[f"{pid}|{cond}|t"] = np.mean(T, 0)
                if X:
                    out[f"{pid}|{cond}|dropped"] = np.mean(X, 0)
                print(f"  {pid:9s} {cond:8s} {len(F):4d} contacts  peak |Δ| {np.max(np.mean(F,0)):.3f} dB "
                      f"at {np.argmax(np.mean(F,0))*DF_HZ:.0f} Hz", flush=True)
    np.savez_compressed(CMP / "spectra.npz", **out)
    return out


def read_iqr(tree: Path, pid: str) -> dict:
    f = tree / pid / "LM" / "Report" / f"{pid}_IQR.tsv"
    if not f.exists():
        return {}
    d = pd.read_csv(f, sep="\t", dtype=str)
    return {r["condition"]: (r.get("n_in", ""), r.get("n_kept", "")) for _, r in d.iterrows()}


def read_notch(tree: Path, pid: str):
    f = tree / pid / "LM" / "Report" / f"{pid}_notch_audit.tsv"
    if not f.exists():
        return None
    try:
        return pd.read_csv(f, sep="\t")
    except Exception:
        return None


def notch_diff(pid: str) -> dict:
    """THE decisive comparison. The adaptive notch is told fmax, so it only considers
    harmonics up to it: at 500 Hz it notched up to 450-483 Hz, at 400 it stops. The notch
    is applied to the TIME SERIES (method 'interp' rebuilds it through the FFT), so
    dropping a harmonic above 400 Hz changes the signal at every frequency - the cubes
    below 400 are NOT the old ones cropped, they are computed from a differently
    filtered signal. This counts what changed, and whether any decision BELOW the new
    ceiling flipped (which would be a different story)."""
    o, n = read_notch(OLD_QC, pid), read_notch(NEW_QC, pid)
    if o is None or n is None:
        return dict(notch_rows_old=np.nan, notch_rows_new=np.nan, notch_lost_above=np.nan,
                    notch_lost_below=np.nan, notch_flipped=np.nan, notch_fmax_old=np.nan,
                    notch_fmax_new=np.nan, notch_note="audit missing in one tree")
    key = ["condition", "shaft", "freq_hz"]
    if not set(key).issubset(o.columns) or not set(key).issubset(n.columns):
        return dict(notch_note="unexpected audit columns")
    io_, in_ = o.set_index(key), n.set_index(key)
    only_old = io_.index.difference(in_.index)
    only_new = in_.index.difference(io_.index)
    both = io_.index.intersection(in_.index)
    fmax_new = float(n.freq_hz.max()) if len(n) else np.nan
    lost_above = sum(1 for k in only_old if k[2] > fmax_new + 1e-6)
    lost_below = len(only_old) - lost_above
    flipped = 0
    for k in both:
        a, b = io_.loc[k, "notched"], in_.loc[k, "notched"]
        a = bool(a if np.ndim(a) == 0 else np.asarray(a).ravel()[0])
        b = bool(b if np.ndim(b) == 0 else np.asarray(b).ravel()[0])
        flipped += int(a != b)
    return dict(notch_rows_old=len(o), notch_rows_new=len(n),
                notch_fmax_old=float(o.freq_hz.max()), notch_fmax_new=fmax_new,
                notch_lost_above=lost_above, notch_lost_below=lost_below,
                notch_added=len(only_new), notch_flipped=flipped,
                notch_note=("harmonics above the new ceiling dropped" if lost_above and not lost_below and not flipped
                            else "decisions changed below the ceiling too" if (lost_below or flipped)
                            else "no change"))


def read_ref(tree: Path, pid: str) -> str:
    f = tree / "wm_reref_report.tsv"
    if not f.exists():
        return ""
    d = pd.read_csv(f, sep="\t", dtype=str)
    r = d[d.patient_id.astype(str) == pid]
    if not len(r):
        return "(no row in this tree's report)"
    r = r.iloc[-1]
    return f"{r.get('wm_channels_used','')}"


def run_date(tree: Path, pid: str) -> str:
    d = tree / pid / "LM" / "ERSP_matrix"
    ts = []
    for c in CONDS:
        p = d / c
        if p.is_dir():
            ts += [f.stat().st_mtime for f in list(p.glob("*.npy"))[:5]]
    return dt.datetime.fromtimestamp(np.median(ts)).strftime("%Y-%m-%d %H:%M") if ts else ""


def attribution(C: pd.DataFrame) -> pd.DataFrame:
    ok = C[C.status == "ok"].copy()
    rows = []
    for pid, g in ok.groupby("patient"):
        i_old, i_new = read_iqr(OLD_QC, pid), read_iqr(NEW_QC, pid)
        tr = []
        for c in CONDS:
            a, b = i_old.get(c), i_new.get(c)
            if a and b:
                tr.append(f"{c} {a[0]}→{b[0]} in / {a[1]}→{b[1]} kept" if a != b else f"{c} {a[1]} kept (same)")
            elif b:
                tr.append(f"{c} only new ({b[1]} kept)")
            elif a:
                tr.append(f"{c} only old ({a[1]} kept)")
        ref_o, ref_n = read_ref(OLD_RAW, pid), read_ref(NEW_RAW, pid)
        rows.append(dict(
            patient=pid, n=len(g),
            old_run=run_date(OLD_RAW, pid), new_run=run_date(NEW_RAW, pid),
            med_rmse=g.rmse.median(), p95_rmse=g.rmse.quantile(.95), max_rmse=g.rmse.max(),
            min_r=g.r.min(), med_maxabs=g.max_abs_diff.median(),
            n_same=int((g.max_abs_diff <= TOL).sum()),
            med_hg=g.hg_mean_abs_diff.median(),
            trials="; ".join(tr),
            trials_changed=any("→" in t for t in tr),
            **notch_diff(pid),
            ref_old=ref_o[:120], ref_new=ref_n[:120],
            ref_changed=(bool(ref_o) and bool(ref_n) and not ref_o.startswith("(") and ref_o != ref_n),
        ))
    A = pd.DataFrame(rows).sort_values("med_rmse", ascending=False)
    A.to_csv(CMP / "attribution.tsv", sep="\t", index=False)
    return A


def fig_why(S: dict, A: pd.DataFrame):
    pats = list(A.patient)
    ncol = 4
    nrow = int(np.ceil(len(pats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 1.9 * nrow), dpi=150,
                             sharex=True)
    axes = np.atleast_2d(axes)
    f_hz = None
    for k, pid in enumerate(pats):
        ax = axes[k // ncol, k % ncol]
        any_ = False
        for cond, c in zip(CONDS, (RED, GREEN, BLUE)):
            key = f"{pid}|{cond}|f"
            if key not in S:
                continue
            v = S[key]
            f_hz = np.arange(len(v)) * DF_HZ
            ax.plot(f_hz, v, lw=1.0, color=c, label=cond)
            any_ = True
        if not any_:
            ax.set_axis_off(); continue
        for h in range(50, 401, 50):
            ax.axvline(h, color="#d8dee4", lw=.6, zorder=0)
        row = A[A.patient == pid].iloc[0]
        ax.set_title(f"{pid} · old {row.old_run[:10]} · med RMSE {row.med_rmse:.3f}", fontsize=7.5, loc="left")
        ax.set_yscale("log")
        ax.tick_params(labelsize=6)
        if k % ncol == 0:
            ax.set_ylabel("mean |Δ| (dB)", fontsize=7)
        if k // ncol == nrow - 1:
            ax.set_xlabel("Hz", fontsize=7)
        if k == 0:
            ax.legend(fontsize=6, frameon=False, ncol=3)
    for k in range(len(pats), nrow * ncol):
        axes[k // ncol, k % ncol].set_axis_off()
    fig.suptitle("FIG D.2 · WHERE the change is, in frequency — mean |new − old| per frequency bin, averaged over each "
                 "patient's contacts (log scale; grey lines = 50 Hz and harmonics)\n"
                 "A comb on the grey lines is the notch; a broad smooth rise is the reference; a flat lift across every "
                 "frequency is the set of trials that went into the average. Patients ordered by median RMSE.",
                 fontsize=9.5, x=0.01, ha="left", y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "D2_why_frequency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # the same over the warped time axis
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 1.9 * nrow), dpi=150, sharex=True)
    axes = np.atleast_2d(axes)
    for k, pid in enumerate(pats):
        ax = axes[k // ncol, k % ncol]
        any_ = False
        for cond, c in zip(CONDS, (RED, GREEN, BLUE)):
            key = f"{pid}|{cond}|t"
            if key not in S:
                continue
            v = S[key]
            ax.plot(np.arange(len(v)), v, lw=1.0, color=c, label=cond)
            any_ = True
        if not any_:
            ax.set_axis_off(); continue
        ax.axvline(150, color=INK, lw=.7, ls=":")
        row = A[A.patient == pid].iloc[0]
        ax.set_title(f"{pid} · {row.trials.split(';')[0].strip()}", fontsize=7, loc="left")
        ax.set_yscale("log"); ax.tick_params(labelsize=6)
        if k % ncol == 0:
            ax.set_ylabel("mean |Δ| (dB)", fontsize=7)
        if k // ncol == nrow - 1:
            ax.set_xlabel("warped bin (150 = stimulus offset)", fontsize=7)
        if k == 0:
            ax.legend(fontsize=6, frameon=False, ncol=3)
    for k in range(len(pats), nrow * ncol):
        axes[k // ncol, k % ncol].set_axis_off()
    fig.suptitle("FIG D.3 · WHERE the change is, in time — mean |new − old| per warped bin; the dotted line is the "
                 "stimulus offset (bin 150).\nA difference confined to one half is the response window or the trial "
                 "set; one spread evenly is preprocessing.", fontsize=9.5, x=0.01, ha="left", y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(FIGS / "D3_why_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    CMP.mkdir(parents=True, exist_ok=True)
    C = pd.read_csv(CMP / "cubes.tsv", sep="\t")
    print("[spectra]")
    S = dict(np.load(CMP / "spectra.npz")) if (CMP / "spectra.npz").exists() else spectra()
    print("\n[attribution]")
    A = attribution(C)
    cols = ["patient", "n", "old_run", "med_rmse", "min_r", "n_same", "trials_changed",
            "notch_fmax_old", "notch_fmax_new", "notch_lost_above", "notch_lost_below",
            "notch_flipped", "ref_changed"]
    print(A[[c for c in cols if c in A.columns]].round(3).to_string(index=False))
    fig_why(S, A)
    print(f"\nwrote -> {CMP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
