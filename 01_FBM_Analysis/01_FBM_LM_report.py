#!/usr/bin/env python3
"""
01_FBM_LM_report.py - one overview PDF per patient from what is in outputs/03_ERSP.

    python 01_FBM_LM_report.py --patients PAT_2868          one patient
    python 01_FBM_LM_report.py                              every patient in the tree

Writes outputs/03_ERSP/<pid>/<pid>_overview.pdf, landscape:
  page 1        the clean PSD per shaft, one panel per condition (PSD_clean/<cond>/PSD/psd_by_shaft.png)
  then          per shaft and condition, at most 6 contacts a page in the shaft's order, always in
                6 columns of the same width (a shorter page leaves the rest empty): row 1 the
                clean ERSP image, row 2 the HFA trials, row 3 the Signal trials

It only collects figures 140 already wrote; it computes nothing. Each page is rasterised and
stored as a JPEG (150 dpi) - a vector PDF of 30 pages of images came to 83 MB.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from functions import config as cfg                      # noqa: E402

TREE = HERE / "outputs" / cfg.ERSP_TREE
CONDS = ("audio", "picture", "reading")
PAGE = (16.5, 11.7)                                      # A3 landscape, inches
PER_PAGE = 6                                             # contacts per page, and always this many columns
CH_RE = re.compile(r"_(?:WM|CAR|NONE)_ERSP_(.+)_TN_CLEAN\.png$")
ROWS = (("ERSP clean", 300), ("HFA trials", 520), ("Signal trials", 520))   # label, thumbnail width px


def shaft(ch: str) -> str:
    return re.sub(r"\d+$", "", ch)


def num(ch: str) -> int:
    m = re.search(r"(\d+)$", ch)
    return int(m.group(1)) if m else 0


def thumb(path: Path, w: int):
    if not path or not path.exists():
        return None
    im = Image.open(path).convert("RGB")
    im.thumbnail((w, w * 3))
    return im


def one(d: Path, pattern: str):
    hits = sorted(d.glob(pattern)) if d.is_dir() else []
    return hits[0] if hits else None


DPI = 150


def page(fig):
    """The figure as one RGB image, and the figure closed."""
    fig.set_dpi(DPI)
    fig.canvas.draw()
    im = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
    plt.close(fig)
    return im


def report(pid: str) -> Path | None:
    lm = TREE / pid / cfg.block_name
    if not lm.is_dir():
        print(f"{pid}: not in the tree"); return None
    hfa, sig = Path(cfg.product_dir(lm, "HFA")), Path(cfg.product_dir(lm, "Signal"))
    out = TREE / pid / f"{pid}_overview.pdf"

    def pages():
        # ---- page 1: clean PSD per shaft, per condition
        fig, axes = plt.subplots(len(CONDS), 1, figsize=PAGE)
        for ax, c in zip(axes, CONDS):
            im = thumb(lm / "PSD_clean" / c / "PSD" / "psd_by_shaft.png", 2600)
            ax.axis("off")
            if im is not None:
                ax.imshow(im)
            ax.set_title(f"{c} - PSD after the notch, per shaft", fontsize=9, loc="left")
        fig.suptitle(f"{pid}  ·  03_ERSP overview", fontsize=13, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.97)); yield page(fig)

        # ---- contacts, by shaft, from the clean ERSP images
        chans = {}
        for c in CONDS:
            for f in (lm / "ERSP_clean" / c).glob("*_TN_CLEAN.png"):
                m = CH_RE.search(f.name)
                if m:
                    chans.setdefault(m.group(1), {})[c] = f
        shafts = {}
        for ch in chans:
            shafts.setdefault(shaft(ch), []).append(ch)
        # at most PER_PAGE contacts on a page, and ALWAYS PER_PAGE columns: a page with two
        # contacts leaves four columns empty, so a contact is the same width on every page
        for sh in sorted(shafts):
            cs = sorted(shafts[sh], key=num)
            parts = [cs[k:k + PER_PAGE] for k in range(0, len(cs), PER_PAGE)]
            for c in CONDS:
                for pi, part in enumerate(parts, 1):
                    fig, axes = plt.subplots(3, PER_PAGE, figsize=PAGE, squeeze=False,
                                             gridspec_kw=dict(height_ratios=[1, 1.5, 2.4], wspace=0.03, hspace=0.04))
                    for ax in axes.ravel():
                        ax.axis("off")
                    for j, ch in enumerate(part):
                        imgs = (thumb(chans[ch].get(c), ROWS[0][1]),
                                thumb(one(hfa / c, f"*_HFAtrials_{ch}.png"), ROWS[1][1]),
                                thumb(one(sig / c, f"*_SIGtrials_{ch}.png"), ROWS[2][1]))
                        for i, im in enumerate(imgs):
                            if im is not None:
                                axes[i, j].imshow(im, aspect="auto")
                            if i == 0:
                                axes[i, j].set_title(ch, fontsize=8, pad=2)
                    for i, (lab, _) in enumerate(ROWS):
                        axes[i, 0].text(-0.06, 0.5, lab, transform=axes[i, 0].transAxes, rotation=90,
                                        va="center", ha="right", fontsize=8)
                    fig.suptitle(f"{pid}  ·  shaft {sh}  ·  {c}  ·  contacts {part[0]}–{part[-1]} of {len(cs)}"
                                 + (f"  ·  page {pi}/{len(parts)}" if len(parts) > 1 else ""),
                                 fontsize=11, x=0.01, ha="left")
                    fig.subplots_adjust(left=0.03, right=0.995, top=0.94, bottom=0.01)
                    yield page(fig)

    n = [0]

    def counted(g):
        for im in g:
            n[0] += 1
            yield im
    g = counted(pages())
    first = next(g)
    first.save(out, "PDF", save_all=True, append_images=g, resolution=DPI, quality=80)
    print(f"{pid}: {n[0]} pages, {out.stat().st_size / 1e6:.1f} MB -> {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", nargs="*")
    a = ap.parse_args()
    pids = a.patients or sorted(p.name for p in TREE.iterdir() if (p / cfg.block_name).is_dir())
    for pid in pids:
        report(pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
