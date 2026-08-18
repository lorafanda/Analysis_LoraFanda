#!/usr/bin/env python3
"""
scan_audio_channels.py - is there a microphone on the Blackrock analog inputs?

WHY THIS MATTERS MORE THAN IT SOUNDS
There is no speech-onset event anywhere in this dataset. Every timing result in
stage 05 is locked to the GO cue - the moment the stimulus ended - not to the
moment the patient actually spoke. The gap between those two is the response
latency itself, and right now it is unmeasured.

Blackrock names its analog inputs `ainp1..ainp16`. The MicroEPI micro-electrode
side is recorded on Blackrock at 30 kHz. If a microphone was patched into one of
those inputs, the speech signal is in `raw_blackrock/*.ns6` - the only stream in
this dataset fast enough to carry audio, and the only route to a real speech
onset.

WHAT WAS ALREADY RULED OUT
The curated `*_export_Labs_*.mat` files contain no ainp channels at all (checked
on all six MicroEPI patients: they carry Micromed-style X1-X9 / MKR1-4 / ECG plus
a separate photodiode array). So the exports cannot answer this and the raw NSx
has to be read. The `04_ersp_LM_RAWONLY` tree does contain ERSP cubes named
`ainp1..ainp3` for PAT_6704 and PAT_6854, which is evidence that an EARLIER
export did include them.

    python scan_audio_channels.py                 # inventory + metrics, all patients
    python scan_audio_channels.py --plot          # also write the diagnostic figures
    python scan_audio_channels.py --patient G-04
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "functions"))
sys.path.insert(0, str(HERE.parent / "01_FBM_Analysis"))
sys.modules.setdefault("mne", types.ModuleType("mne"))  # config imports it transitively
import lf_nsx as N  # noqa: E402
from functions import config as cfg  # noqa: E402

RAW = Path(r"S:\HumanNeuronLab\DATARAW\MICROEPI")
OUT = HERE / "outputs" / "audio_scan"


def nsx_files(pid: str):
    d = RAW / f"MicroEPI-{pid}" / "raw_blackrock"
    if not d.is_dir():
        return []
    return sorted(Path(p) for p in glob.glob(str(d / "**" / "*.ns*"), recursive=True)
                  if not p.lower().endswith(".nev"))


def speech_metrics(x: np.ndarray, fs: float) -> dict:
    """Cheap, interpretable descriptors. None of them alone proves speech; the
    combination separates a live microphone from an unconnected input."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    out = dict(std=float(x.std()), ptp=float(np.ptp(x)))
    if x.std() < 1e-9 or x.size < int(fs):
        out.update(frac_100_4k=np.nan, frac_4k_up=np.nan, env_2_8Hz=np.nan,
                   crest=np.nan, silence_frac=np.nan, dyn_db=np.nan)
        return out
    from numpy.fft import rfft, rfftfreq
    S = np.abs(rfft(x)) ** 2
    fr = rfftfreq(x.size, 1 / fs)
    tot = max(S.sum(), 1e-30)

    def band(lo, hi):
        return float(S[(fr >= lo) & (fr < hi)].sum() / tot)

    # 20 ms envelope; speech modulates strongly at the syllable rate, 2-8 Hz
    w = max(1, int(fs * 0.02))
    env = np.convolve(np.abs(x), np.ones(w) / w, mode="same")
    Se = np.abs(rfft(env - env.mean())) ** 2
    fe = rfftfreq(env.size, 1 / fs)
    denom = max(Se[(fe >= 0.5) & (fe < 20)].sum(), 1e-30)
    out["env_2_8Hz"] = float(Se[(fe >= 2) & (fe < 8)].sum() / denom)
    out["frac_100_4k"] = band(100, 4000)
    out["frac_4k_up"] = band(4000, min(15000, fs / 2 - 1))
    out["crest"] = float(np.abs(x).max() / max(x.std(), 1e-12))
    # speech is intermittent: a live mic in a task has quiet stretches
    thr = np.percentile(env, 20)
    out["silence_frac"] = float((env <= max(thr, 1e-12)).mean())
    lo = max(np.percentile(env, 10), 1e-9)
    out["dyn_db"] = float(20 * np.log10(max(np.percentile(env, 95), 1e-9) / lo))
    return out


def scan_file(p: Path, block_s: float, n_blocks: int) -> list:
    try:
        h = N.read_header(p)
    except Exception as e:
        return [dict(file=p.name, error=f"{type(e).__name__}: {e}")]
    ai = N.find_channels(h, "ainp")
    if not ai:
        return [dict(file=p.name, fs=h.fs, n_channels=h.n_channels,
                     duration_s=h.n_samples / h.fs if h.fs else np.nan,
                     channel=None, note="no ainp channels")]
    dur = h.n_samples / h.fs
    # sample blocks spread across the file rather than one window: a single
    # 20 s slice can easily land in silence and say nothing
    starts = np.linspace(0, max(dur - block_s, 0), n_blocks)
    rows = []
    for k, i in enumerate(ai):
        agg = []
        for t0 in starts:
            X = N.read_window(h, [i], float(t0), float(t0 + block_s),
                              max_samples=int(block_s * h.fs) + 10)
            fs_eff = N.effective_fs(h, float(t0), float(t0 + block_s),
                                    max_samples=int(block_s * h.fs) + 10)
            if X.shape[1] > 10:
                agg.append(speech_metrics(X[0], fs_eff))
        if not agg:
            continue
        m = {k2: float(np.nanmean([a[k2] for a in agg])) for k2 in agg[0]}
        rows.append(dict(file=p.name, fs=h.fs, n_channels=h.n_channels,
                         duration_s=dur, channel=h.chan_labels[i], **m))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", action="append")
    ap.add_argument("--block-s", type=float, default=10.0)
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument("--max-files", type=int, default=3,
                    help="NSx files per patient to sample (each is ~5 min / 920 MB)")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    pats = a.patient or sorted(cfg.MICROEPI_MAT_PRESETS)
    rows = []
    for pid in pats:
        files = nsx_files(pid)
        pat_name = cfg.MICROEPI_MAT_PRESETS.get(pid, {}).get("pat_name", "")
        print(f"\n== {pid} ({pat_name}): {len(files)} NSx files")
        if not files:
            rows.append(dict(patient=pid, pat_name=pat_name, file=None,
                             note="no raw_blackrock"))
            continue
        for p in files[:a.max_files]:
            for r in scan_file(p, a.block_s, a.n_blocks):
                r.update(patient=pid, pat_name=pat_name)
                rows.append(r)
                if r.get("channel"):
                    print(f"   {p.name:34s} {r['channel']:7s} "
                          f"std={r.get('std', float('nan')):8.1f} "
                          f"100-4k={r.get('frac_100_4k', float('nan')):.3f} "
                          f"4k+={r.get('frac_4k_up', float('nan')):.3f} "
                          f"env2-8={r.get('env_2_8Hz', float('nan')):.3f} "
                          f"dyn={r.get('dyn_db', float('nan')):5.1f}dB")
                else:
                    print(f"   {p.name:34s} {r.get('note', r.get('error', ''))}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "audio_channel_scan.csv", index=False)

    print("\n" + "=" * 78)
    print("VERDICT PER PATIENT")
    print("=" * 78)
    verdict = {}
    if "channel" in df.columns:
        d = df[df["channel"].notna()]
        for pid, g in d.groupby("patient"):
            best = g.loc[g["std"].idxmax()] if g["std"].notna().any() else None
            if best is None:
                verdict[pid] = "no usable ainp"
                continue
            # A live microphone: energy concentrated below ~4 kHz, strong syllable-rate
            # modulation, and real dynamic range. Amplifier noise on an unconnected
            # input is flat, hiss-dominated and has almost no dynamic range.
            speechy = (best["frac_100_4k"] > 0.45 and best["env_2_8Hz"] > 0.25
                       and best["dyn_db"] > 12)
            noisy = best["frac_4k_up"] > 0.6 and best["dyn_db"] < 12
            verdict[pid] = ("LIKELY AUDIO" if speechy else
                            "likely amplifier noise / unconnected" if noisy else
                            "inconclusive - inspect the plots")
            print(f"  {pid} ({g['pat_name'].iloc[0]}): {verdict[pid]}")
            print(f"     strongest = {best['channel']} on {best['file']}, "
                  f"std {best['std']:.1f}, 100-4k {best['frac_100_4k']:.3f}, "
                  f"4k+ {best['frac_4k_up']:.3f}, env2-8 {best['env_2_8Hz']:.3f}, "
                  f"dyn {best['dyn_db']:.1f} dB")
    (OUT / "verdict.json").write_text(json.dumps(
        dict(verdict=verdict, block_s=a.block_s, n_blocks=a.n_blocks,
             max_files=a.max_files,
             written=datetime.now().strftime("%Y-%m-%d %H:%M:%S")), indent=2),
        encoding="utf-8")
    print(f"\n  -> {OUT / 'audio_channel_scan.csv'}")

    if a.plot:
        make_plots(pats, a)
    return 0


def make_plots(pats, a) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for pid in pats:
        files = nsx_files(pid)
        if not files:
            continue
        p = files[0]
        try:
            h = N.read_header(p)
        except Exception:
            continue
        ai = N.find_channels(h, "ainp")
        if not ai:
            continue
        dur = h.n_samples / h.fs
        fig, axes = plt.subplots(len(ai) + 1, 1, figsize=(13, 2.4 * (len(ai) + 1)),
                                 sharex=True)
        # envelope across the WHOLE file, decimated: speech shows as bursts
        for k, i in enumerate(ai):
            X = N.read_window(h, [i], 0.0, dur, max_samples=400_000)
            fs_eff = N.effective_fs(h, 0.0, dur, max_samples=400_000)
            x = X[0].astype(float)
            t = np.arange(x.size) / fs_eff
            w = max(1, int(fs_eff * 0.05))
            env = np.convolve(np.abs(x - x.mean()), np.ones(w) / w, mode="same")
            axes[k].plot(t, env, lw=.5, color=f"C{k}")
            axes[k].set_ylabel(f"{h.chan_labels[i]}\n|env|", fontsize=9)
            axes[k].spines[["top", "right"]].set_visible(False)
        # spectrogram of the strongest channel
        i = ai[int(np.argmax([N.read_window(h, [j], dur / 2, dur / 2 + 5).std()
                              for j in ai]))]
        X = N.read_window(h, [i], 0.0, dur, max_samples=1_200_000)
        fs_eff = N.effective_fs(h, 0.0, dur, max_samples=1_200_000)
        axes[-1].specgram(X[0].astype(float), NFFT=1024, Fs=fs_eff, noverlap=512,
                          cmap="magma")
        axes[-1].set_ylabel(f"{h.chan_labels[i]}\nHz", fontsize=9)
        axes[-1].set_xlabel("seconds into this NSx file "
                            "(Blackrock clock, NOT the export/task clock)")
        fig.suptitle(f"{pid} — Blackrock analog inputs, {p.name}  "
                     f"({dur/60:.1f} min @ {h.fs:.0f} Hz)", x=.02, ha="left")
        fig.tight_layout()
        OUT.mkdir(parents=True, exist_ok=True)
        fp = OUT / f"AUDIO_{pid}_{p.stem}.png"
        fig.savefig(fp, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {fp.name}")


if __name__ == "__main__":
    raise SystemExit(main())
