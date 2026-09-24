"""
precompute_contact_ersp.py - the per-contact ERSP matrices the LM visualizer's spectrogram draws.

    "C:/Users/fanda/AppData/Local/Programs/Python/Python311/python.exe" \
        02_FBM_Clustering/scripts/precompute_contact_ersp.py

WHAT. One small binary per contact of the activity_viz bundle, holding that contact's
actual ERSP - the same ERSP_matrix/<cond>/*_TN.npy the exemplar electrode figures are
drawn from, 0-400 Hz, all three conditions - so the page can show the real spectrogram of
a selected electrode rather than the eight band means of the cube. The page fetches one
file when a contact is selected (~46 kB), never the whole set.

SOURCE. The same cache precompute_activity_cube.py reads (2026-09-22: the newest concat cache,
02_FBM_Clustering/outputs/_dataset/concat_source_v<N>, or --dataset <dir>):
    X_3d.npy          (n, 103, 300) float32
    df_meta.parquet   (patient_id, electrode, condition)
and the bundle's own contacts.json, so contact i here is contact i in the cube.

OUTPUT.  02_FBM_Clustering/outputs/250_recon/fsaverage/activity_viz/ersp/
    manifest.json      shape, axes, quantisation
    c<i>.bin           uint8 [n_cond, n_freq, n_time]: 0 = no data, 1..255 = vmin..vmax dB

    Frequency: bins 0..102 of the 129-bin 0-500 Hz axis = 0-398 Hz, kept at full resolution.
    Time: the 300 warped bins averaged in pairs to 150 - the panel is ~130 px wide per
    condition, so nothing is lost, and the whole set is ~134 MB instead of ~270.
    Values: clipped to +-VLIM dB and quantised to a byte (0.08 dB steps); the cube's 99.9th
    percentile is 5 dB, so nothing real is clipped.

Commit the ersp/ folder to the activity-visualizer branch beside the rest of the bundle
(see make_lm_visualizer.py for the worktree recipe).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]   # not a drive letter: S:\ and the UNC path are the same repo


def newest_concat_cache() -> Path:
    d = REPO / "02_FBM_Clustering" / "outputs" / "_dataset"
    cands = [(int(p.name[len("concat_source_v"):]), p) for p in d.glob("concat_source_v*")
             if p.is_dir() and p.name[len("concat_source_v"):].isdigit()]
    if not cands:
        raise FileNotFoundError(f"no concat_source_v<N> under {d}")
    return max(cands)[1]


_ap = argparse.ArgumentParser()
_ap.add_argument("--dataset", help="cache dir holding X_3d.npy + df_meta.parquet (default: the newest concat_source_v<N>)")
_ARGS, _ = _ap.parse_known_args()
DATASET_DIR = Path(_ARGS.dataset) if _ARGS.dataset else newest_concat_cache()
ERSP_NPY = DATASET_DIR / "X_3d.npy"
META_PARQUET = DATASET_DIR / "df_meta.parquet"
BUNDLE = REPO / "02_FBM_Clustering" / "outputs" / "250_recon" / "fsaverage" / "activity_viz"
OUT_DIR = BUNDLE / "ersp"

NF, NT, FMAX_HZ = 103, 300, 398.4375   # 102 x 3.90625 Hz: the last bin of the 0-400 Hz cube (was 500 = bin 128 of the 0-500 one), 2026-09-18
CONDITIONS = ["audio", "picture", "reading"]
F_HI_HZ = 400.0
T_DS = 2                       # 300 -> 150 time bins
VLIM = 10.0                    # dB, symmetric; byte 1 = -VLIM, byte 255 = +VLIM, byte 0 = no data


def norm_el(s) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(s)).upper()


def main() -> None:
    X = np.load(ERSP_NPY, mmap_mode="r")
    if X.ndim != 3:
        X = X.reshape(X.shape[0], NF, NT)
    assert X.shape[1:] == (NF, NT), X.shape
    df = pd.read_parquet(META_PARQUET).reset_index(drop=True)
    assert len(df) == X.shape[0]
    df["key"] = list(zip(df["patient_id"].astype(str).str.strip(),
                         df["electrode"].astype(str).str.strip().map(norm_el),
                         df["condition"].astype(str).str.strip()))
    row_of = {}
    for i, k in enumerate(df["key"]):
        row_of.setdefault(k, i)          # one sample per (patient, electrode, condition) here

    contacts = json.loads((BUNDLE / "contacts.json").read_text(encoding="utf-8"))
    n_f = NF                                                       # the cube already ends at 400 Hz
    n_t = NT // T_DS
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scale = 254.0 / (2 * VLIM)
    n_ok = 0
    for ci, ct in enumerate(contacts):
        out = np.zeros((len(CONDITIONS), n_f, n_t), dtype=np.uint8)
        for k, cond in enumerate(CONDITIONS):
            r = row_of.get((str(ct["patient_id"]), norm_el(ct["electrode"]), cond))
            if r is None:
                continue
            m = np.asarray(X[r, :n_f, :], dtype=np.float32)
            m = m.reshape(n_f, n_t, T_DS)
            with np.errstate(invalid="ignore"):
                m = np.nanmean(m, axis=2)
            q = np.clip((m + VLIM) * scale, 0, 254) + 1.0
            q[~np.isfinite(m)] = 0
            out[k] = q.astype(np.uint8)
            n_ok += 1
        out.tofile(OUT_DIR / f"c{ci}.bin")
        if ci % 500 == 0:
            print(f"  {ci}/{len(contacts)}")
    manifest = {
        "source": str(ERSP_NPY.relative_to(REPO)).replace("\\", "/"),
        "conditions": CONDITIONS, "n_cond": len(CONDITIONS),
        "n_freq": n_f, "f_hz": [0.0, (n_f - 1) * FMAX_HZ / (NF - 1)], "fmax_source_hz": FMAX_HZ,
        "n_time": n_t, "time_downsample": T_DS, "n_time_source": NT,
        "dtype": "uint8", "order": ["cond", "freq", "time"],
        "vmin": -VLIM, "vmax": VLIM, "nan_byte": 0,
        "file": "c{contact_index}.bin", "n_contact": len(contacts),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total = sum(p.stat().st_size for p in OUT_DIR.glob("*.bin")) / 1e6
    print(f"[done] {len(contacts)} contacts, {n_ok} contact-conditions with data, {total:.0f} MB -> {OUT_DIR}")


if __name__ == "__main__":
    main()
