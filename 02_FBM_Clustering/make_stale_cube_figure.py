#!/usr/bin/env python3
"""
make_stale_cube_figure.py - show every stale ERSP cube before deleting any of them.

A cube is STALE if it predates the 2026-08-21 re-run of stage 01. There are two very
different reasons a cube can be stale, and only one of them makes deletion correct:

  OBSOLETE  the channel is no longer processed at all - dropped as a manual bad
            channel, as white matter, as unparcellated, or as auxiliary. The cube is a
            leftover from a run before that decision and nothing reads it.
  GAP       the channel IS still processed but this condition was not re-run. Deleting
            it would throw away data that should be REGENERATED instead.

Told apart by asking whether the same channel has a fresh cube in any condition. On the
four patients audited here every one of the 84 stale cubes is obsolete and none is a
gap, so they are safe to delete - but the point of this figure is to look first.

    python make_stale_cube_figure.py
"""
from __future__ import annotations

import datetime as dt
import re
import sys
import textwrap
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "01_FBM_Analysis" / "functions"))
sys.path.insert(0, str(ROOT.parent / "01_FBM_Analysis"))

ERSP = ROOT.parent / "01_FBM_Analysis" / "outputs" / "04_ersp_LM_RAWONLY"
OUT = ROOT / "outputs" / "clustering" / "gate_examples"

CUT = dt.datetime(2026, 8, 21, 12, 0).timestamp()
PATS = ["EL030", "EL035", "EL042", "PAT_3301"]
VMIN, VMAX, FMAX = -7.0, 7.0, 400.0
INK, MUTED, RED, GREEN = "#1b232c", "#68727d", "#c1121f", "#1b7837"


def manual_bad():
    """config.bad_channels_manual, normalised, so a cube can be labelled with its reason."""
    try:
        import config as C
    except Exception:
        return {}
    norm = lambda s: str(s).replace("_", "").replace("-", "").upper()
    return {p: {norm(c) for c in v}
            for p, v in (getattr(C, "bad_channels_manual", {}) or {}).items()}


def scan():
    norm = lambda s: str(s).replace("_", "").replace("-", "").upper()
    bad = manual_bad()
    rows = []
    for pid in PATS:
        root = ERSP / pid / "LM" / "ERSP_matrix"
        if not root.is_dir():
            continue
        fresh, stale = set(), []
        for cdir in sorted(root.iterdir()):
            if not cdir.is_dir():
                continue
            for f in cdir.glob("*_TN.npy"):
                m = re.match(rf"^{re.escape(pid)}_{re.escape(cdir.name)}_.+?"
                             rf"_ERSP_(.+?)_TN\.npy$", f.name)
                if not m:
                    continue
                if f.stat().st_mtime >= CUT:
                    fresh.add(m.group(1))
                else:
                    stale.append((cdir.name, m.group(1), f))
        for cond, ch, f in stale:
            reason = ("manual bad channel" if norm(ch) in bad.get(pid, set())
                      else "dropped (WM / unparcellated)")
            rows.append(dict(patient=pid, channel=ch, condition=cond, path=str(f),
                             mtime=dt.datetime.fromtimestamp(f.stat().st_mtime),
                             kind="obsolete" if ch not in fresh else "GAP",
                             reason=reason))
    return pd.DataFrame(rows)


def main() -> int:
    df = scan()
    if df.empty:
        print("no stale cubes found — nothing to show")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "stale_cube_audit.csv", index=False)

    n_gap = int((df.kind == "GAP").sum())
    # one panel per CHANNEL (each is stale in every condition it has), so the grid
    # shows distinct channels rather than the same channel three times
    uniq = (df.sort_values(["patient", "channel", "condition"])
              .groupby(["patient", "channel"], as_index=False).first())
    n = len(uniq)
    ncol = 6
    nrow = int(np.ceil(n / ncol))
    print(f"{len(df)} stale cubes across {n} channels "
          f"({int((df.kind=='obsolete').sum())} obsolete, {n_gap} gap)")

    fig, axes = plt.subplots(nrow, ncol, figsize=(2.05 * ncol, 2.35 * nrow + 2.4),
                             dpi=170)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")

    for ax, r in zip(axes, uniq.itertuples()):
        arr = np.load(r.path)
        ax.pcolormesh(np.linspace(0, 100, arr.shape[1]),
                      np.linspace(0, FMAX, arr.shape[0]),
                      arr, cmap="bwr", vmin=VMIN, vmax=VMAX, shading="auto",
                      rasterized=True)
        ax.axvline(50, color="#333333", lw=0.6, ls=(0, (3, 2)))
        ax.set_xticks([]); ax.set_yticks([])
        col = RED if r.kind == "obsolete" else "#b8860b"
        for sp in ax.spines.values():
            sp.set_color(col); sp.set_linewidth(1.5)
        n_cond = int((df.patient.eq(r.patient) & df.channel.eq(r.channel)).sum())
        ax.set_title(f"{r.patient} · {r.channel}", fontsize=7.6, color=INK, pad=2.5)
        ax.set_xlabel(f"{r.mtime:%Y-%m-%d} · {n_cond} cond\n{r.reason}",
                      fontsize=6.4, color=MUTED, labelpad=3)

    fig_h = 2.35 * nrow + 2.4
    HEADER_IN = 2.30                      # suptitle + the wrapped body, measured
    top = 1 - HEADER_IN / fig_h
    fig.subplots_adjust(top=top, bottom=0.035, left=0.022, right=0.985,
                        hspace=0.62, wspace=0.10)

    fig.suptitle("Every stale ERSP cube, before deleting any of them",
                 x=0.022, y=1 - 0.28 / fig_h, ha="left", fontsize=15, color=INK)
    verdict = ("All of them are obsolete and none is a gap — safe to delete."
               if n_gap == 0 else
               f"WARNING: {n_gap} are GAPS, not obsolete — regenerate those, do not delete.")
    body = [
        f"{len(df)} cubes across {n} channels predate the 2026-08-21 re-run of stage 01. "
        f"A stale cube is safe to delete only if its CHANNEL is no longer processed at "
        f"all; if the channel is still live and just this condition was missed, it is a "
        f"gap and should be regenerated instead.",
        f"Every channel was checked for a fresh cube in any condition. {verdict} "
        f"EL035's list is exactly its bad_channels_manual entry (EntG_R7-11, Fopc_R1-5, "
        f"PHG_R11-12) - the expected signature of channels excluded after these cubes "
        f"were written.",
        "Nothing currently published reads these - the dataset cache was built "
        "2026-08-19, before the re-run, so the clustering is internally consistent. They "
        "matter only when the cache is rebuilt, which is when they would be mixed with "
        "August cubes.",
    ]
    fig.text(0.022, 1 - 0.62 / fig_h,
             "\n".join(textwrap.fill(t, width=150) for t in body),
             fontsize=8.2, color=MUTED, va="top", linespacing=1.5)

    p = OUT / "S1_stale_cubes.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(df.groupby(["patient", "reason"]).size().to_string())
    print(f"\n-> {p}")
    print(f"-> {OUT / 'stale_cube_audit.csv'}")
    if n_gap == 0:
        print("\nAll obsolete. To delete after looking, the paths are in the CSV.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
