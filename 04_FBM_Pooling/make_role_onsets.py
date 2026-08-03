#!/usr/bin/env python3
"""
make_role_onsets.py — persist the per-role onset that 460 computes and throws away.

`lf_pool.plot_role_hga_timeseries` returns an onset_dict and 460 uses it to draw
role_onset_ordering.png, but never writes it down. Its own docstring says to patch it
into role_info.json; that was never done, and the consequence is that the a-priori
ROLE ordering cannot be compared against the data-driven CLUSTER ordering without
re-running the whole pooling notebook.

This recomputes the same quantity from the cached pooling cube and writes
`role_onsets.csv` + patches `role_info.json`, using a definition byte-identical to
lf_cluster_timing so the two ladders share an axis:

    onset = first bin whose |mean HGA| crosses `onset_thr_db`, as % of the warped
            trial (50% = GO). No crossing -> NaN, kept out of the ladder rather than
            drawn at 100.

Multi-label on purpose: a contact contributes to EVERY role it matched
(`roles_matched`), not just its primary. Forcing one role per contact would throw away
the 30% multi-load cases that the attribution matrix exists to document.

    python make_role_onsets.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "04_FBM_Pooling" / "outputs" / "_dataset" / "pooling" / "_raw_ungated"
POOL = ROOT / "04_FBM_Pooling" / "outputs" / "pooling" / "pool_web"
CONDITIONS = ("audio", "picture", "reading")
HGA_HZ = (70.0, 150.0)
FMAX_HZ = 500.0
ONSET_THR_DB = 0.5


def norm(s) -> str:
    return "" if s is None else str(s).replace("_", "").replace("-", "").upper()


def main() -> int:
    X3 = np.load(CACHE / "X_3d.npy", mmap_mode="r")
    meta = pd.read_parquet(CACHE / "df_meta.parquet").reset_index(drop=True)
    meta["contact_norm"] = meta["electrode"].map(norm)
    meta["_row"] = np.arange(len(meta))
    n_f, n_t = X3.shape[1], X3.shape[2]

    pool = pd.read_csv(POOL / "contacts_pool.csv")
    pool["patient_id"] = pool["patient"].astype(str)
    pool["contact_norm"] = pool["contact"].map(norm)

    # HGA rows on the full-resolution frequency axis
    hz = FMAX_HZ / n_f
    rows_hga = np.array([i for i in range(n_f)
                         if i * hz < HGA_HZ[1] and (i + 1) * hz > HGA_HZ[0]])
    print(f"  cube {X3.shape}; HGA rows {rows_hga.min()}..{rows_hga.max()} "
          f"({HGA_HZ[0]:.0f}-{HGA_HZ[1]:.0f} Hz)")

    # one concatenated HGA trace per electrode
    key_rows = {}
    for (pid, cn), g in meta.groupby(["patient_id", "contact_norm"], sort=False):
        by = {c: g[g.condition == c] for c in CONDITIONS}
        if any(len(by[c]) == 0 for c in CONDITIONS):
            continue
        key_rows[(pid, cn)] = [int(by[c].iloc[0]["_row"]) for c in CONDITIONS]
    print(f"  {len(key_rows)} electrodes with all three conditions")

    hga = {}
    for k, rr in key_rows.items():
        hga[k] = np.stack([np.asarray(X3[r])[rows_hga].mean(axis=0) for r in rr])  # (3, n_t)

    t_pct = np.linspace(0, 100, n_t, endpoint=False) + 50.0 / n_t
    roles: dict[str, list] = {}
    for _, r in pool.iterrows():
        k = (str(r["patient_id"]), str(r["contact_norm"]))
        if k not in hga:
            continue
        for role in str(r.get("roles_matched", "") or "").split(";"):
            if role:
                roles.setdefault(role, []).append(k)

    out = []
    for role, keys in sorted(roles.items()):
        M = np.stack([hga[k] for k in keys])                    # (n, 3, n_t)
        for bi, cond in enumerate(CONDITIONS):
            gm = M[:, bi, :].mean(axis=0)
            hit = np.where(np.abs(gm) > ONSET_THR_DB)[0]
            stim = np.where((np.abs(gm) > ONSET_THR_DB) & (t_pct < 50))[0]
            resp = np.where((np.abs(gm) > ONSET_THR_DB) & (t_pct >= 50))[0]
            out.append(dict(
                role=role, condition=cond, n=len(keys),
                onset_pct=float(t_pct[hit[0]]) if len(hit) else np.nan,
                onset_stim_pct=float(t_pct[stim[0]]) if len(stim) else np.nan,
                onset_resp_pct=float(t_pct[resp[0]]) if len(resp) else np.nan,
                peak_pct=float(t_pct[int(np.argmax(np.abs(gm)))]),
                peak_db=float(gm[int(np.argmax(np.abs(gm)))]),
            ))
    df = pd.DataFrame(out)
    df.to_csv(POOL / "role_onsets.csv", index=False)
    print(f"  {df.role.nunique()} roles x {len(CONDITIONS)} conditions -> role_onsets.csv")

    # patch role_info.json so the website sidebar can sort by onset too
    rip = POOL / "role_info.json"
    ri = json.loads(rip.read_text(encoding="utf-8"))
    n_patched = 0
    for role, g in df.groupby("role"):
        if role in ri and isinstance(ri[role], dict):
            ri[role]["onset"] = {r.condition: (None if pd.isna(r.onset_pct) else round(r.onset_pct, 2))
                                 for r in g.itertuples()}
            n_patched += 1
    rip.write_text(json.dumps(ri, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  patched onset into {n_patched}/{len(ri)} entries of role_info.json")

    show = (df[df.condition == "audio"].dropna(subset=["onset_pct"])
            .sort_values("onset_pct")[["role", "n", "onset_pct"]].head(8))
    print("\n  earliest roles in AUDIO:")
    print(show.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
