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
import csv
import glob
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
PER_PAGE = 4                                             # contacts per page, and always this many columns
CH_RE = re.compile(r"_(?:WM|CAR|NONE)_ERSP_(.+)_TN_CLEAN\.png$")
# label, thumbnail width px. Four columns on an A3 page give each contact about 4 inches,
# which is 600 px at 150 dpi, so the thumbnails are sized for that rather than for six.
ROWS = (("ERSP clean", 460), ("HFA trials", 780), ("Signal trials", 780))
# row heights: the HFA raster loses a little to the signal traces, which were cramped
HEIGHTS = (1.0, 1.35, 2.85)


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
# the contact's name: kept, bad-listed, and in the white-matter reference. The WM colour
# is a dark amber rather than a true yellow, which is unreadable on white.
GOOD, BAD, WMC = "#2e7d32", "#c1121f", "#b8860b"
BIDS = Path(r"\\nasac-m2.unige.ch\m-HumanNeuronLab\DATARAW\BIDS_elec")


def norm(s: str) -> str:
    """lf_io_utils.normalize_label, without importing it (that module wants mne)."""
    return str(s).replace("_", "").replace("-", "").upper()


def wm_contacts(pid: str) -> set[str]:
    """The contacts in the white-matter reference, by the pipeline's own rule:
    lf_io_utils.derive_wm_channels_from_electrodes_tsv wants tissueWeights_1 > 0.97 AND
    the first tissueLabel token to start with 'wm-'. Reproduced rather than imported,
    because that module pulls in mne. Returns normalised names."""
    num = pid.replace("PAT_", "")
    out = set()
    for sub in (f"MICROEPI/sub-{num}", f"SEEG-HUG/sub-{num}", f"SEEG-BERN/sub-{pid}"):
        for p in glob.glob(str(BIDS / sub / "ieeg" / "*_electrodes.tsv")):
            with open(p, newline="", encoding="utf-8", errors="replace") as fh:
                for r in csv.DictReader(fh, delimiter="\t"):
                    try:
                        w1 = float(r.get("tissueWeights_1", "nan"))
                    except (TypeError, ValueError):
                        continue
                    tok = str(r.get("tissueLabel", "")).strip().split()
                    if w1 > 0.97 and tok and tok[0].startswith("wm-"):
                        out.add(norm(r["name"]))
            return out
    return out


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

        # ---- page 2: the trigger channel the trials were cut from, if it has been drawn
        # Collected, not computed: make_photodiode_figures.py reads the raw recording and
        # writes one PNG per patient, and this picks it up if it is there. A patient whose
        # figure has not been made simply has no such page.
        pdp = TREE.parent / "photodiode_check" / f"{pid}_photodiode.png"
        im = thumb(pdp, 3000)
        if im is not None:
            fig, ax = plt.subplots(figsize=PAGE)
            ax.axis("off")
            ax.imshow(im)
            fig.suptitle(f"{pid}  ·  the trigger channel, and the onsets taken from it",
                         fontsize=13, x=0.01, ha="left")
            fig.tight_layout(rect=(0, 0, 1, 0.97)); yield page(fig)

        # ---- contacts, by shaft, from the clean ERSP images
        chans = {}
        for c in CONDS:
            for f in (lm / "ERSP_clean" / c).glob("*_TN_CLEAN.png"):
                m = CH_RE.search(f.name)
                if m:
                    chans.setdefault(m.group(1), {})[c] = f
        if not chans:
            # a --fast run writes no ERSP_clean, so the contact list comes from the
            # per-trial plots instead; those pages then have an empty first row
            for c in CONDS:
                for f in (sig / c).glob("*_SIGtrials_*.png"):
                    m = re.search(r"_SIGtrials_(.+)\.png$", f.name)
                    if m:
                        chans.setdefault(m.group(1), {})
        # the bad-listed contacts have no figures at all - 140 writes nothing for them -
        # so they are put back here as empty cells with a red name. Without this a removed
        # contact simply vanishes and a shaft reads as if it never had one.
        wm = wm_contacts(pid)
        listed = getattr(cfg, "bad_channels_manual", {}).get(pid, []) or []
        have = {norm(ch) for ch in chans}
        bad = set()
        for b in listed:
            if norm(b) in have:
                bad.add(next(ch for ch in chans if norm(ch) == norm(b)))
            else:
                chans.setdefault(b, {})
                bad.add(b)
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
                                             gridspec_kw=dict(height_ratios=list(HEIGHTS),
                                                              wspace=0.03, hspace=0.06))
                    for ax in axes.ravel():
                        ax.axis("off")
                    for j, ch in enumerate(part):
                        imgs = (thumb(chans[ch].get(c), ROWS[0][1]),
                                thumb(one(hfa / c, f"*_HFAtrials_{ch}.png"), ROWS[1][1]),
                                thumb(one(sig / c, f"*_SIGtrials_{ch}.png"), ROWS[2][1]))
                        for i, im in enumerate(imgs):
                            if im is not None:
                                # no aspect="auto": that stretched every panel to fill its
                                # cell, so the same raster looked a different shape on a
                                # full page than on a half-empty one
                                axes[i, j].imshow(im)
                            if i == 0:
                                # bad wins over WM in the colour, but both are named: a
                                # bad contact that is IN the reference is the case that
                                # matters (it puts its noise on every other channel)
                                is_bad, is_wm = ch in bad, norm(ch) in wm
                                tag = ("  · bad" if is_bad else "") + ("  · WM ref" if is_wm else "")
                                axes[i, j].set_title(ch + tag, fontsize=9, pad=2,
                                                     color=BAD if is_bad else (WMC if is_wm else GOOD),
                                                     fontweight="bold")
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
    ap.add_argument("--tree", default=None,
                    help="read another tree under outputs/ "
                         "(03_ERSP-allchan, 03_ERSP-allchan-ref)")
    a = ap.parse_args()
    if a.tree:
        global TREE
        TREE = HERE / 'outputs' / a.tree
    pids = a.patients or sorted(p.name for p in TREE.iterdir() if (p / cfg.block_name).is_dir())
    for pid in pids:
        report(pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
