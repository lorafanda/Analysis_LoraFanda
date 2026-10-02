#!/usr/bin/env python3
"""
143_delete_non_neural.py - remove the non-neural channels (EKG-, EMG-, photo, E1-E4 ...) from
the patients already in the 03_ERSP tree.

    python 143_delete_non_neural.py              list, per patient - deletes nothing
    python 143_delete_non_neural.py --delete     delete, and print what went per patient
    python 143_delete_non_neural.py --patients EL035 PAT_3455 [--delete]
    python 143_delete_non_neural.py --bad-listed --patients EL038 [--delete]
                                                 the same sweep for the contacts in cfg.bad_channels_manual:
                                                 a contact bad-listed AFTER its patient was run, removed from
                                                 the tree without a rerun (its cubes, halves, images, figures
                                                 and score rows). Use it only when nothing else in the patient
                                                 depends on the contact - not a reference contact; a rerun is
                                                 still what makes the notch decide without it.

WHY. For EL and HUG patients 140's aux drop used a short prefix list (MRK, MKR, X, ECG, EX,
AUDIO), so EKG-, EMG-, the HUG photodiode ("photo") and PAT_3975's E1-E4 inputs survived it:
they went through the notch and got ERSP, HFA and Signal figures like any contact. The
export always skipped them (lf_io_utils._is_non_neural), so no cube exists for them - what
is on disk is figures, and their rows in the per-trial score tables.

WHAT COUNTS AS NON-NEURAL is not decided here: it is lf_io_utils._is_non_neural, the same test
the export and the MicroEPI branch use, applied to the channel name in each file name.

THE PHOTODIODE IS THE EXCEPTION (Lora, 2026-10-02). Its ERSP and HFA figures are kept: they
are the check that the triggers sit where the screen changed, in the four HUG recordings
that carry a "photo" channel (PAT_2868, 3390, 3415, 3455). It is still not data, so its rows
leave the score tables and a Signal figure of it is removed - only ERSP/ and HFA/ stay.

WHAT IT REMOVES, under outputs/<cfg.ERSP_TREE>/<pid>/LM/:
  - every per-channel file of such a channel, in whatever folder it sits (ERSP, HFA, Signal,
    and ERSP_matrix / ERSP_halves / ERSP_clean should one ever be there);
  - its rows in TrialScores/<cond>/*_trial_scores.tsv (the table is rewritten without them),
    so 146's share-of-channels no longer counts a heartbeat as a channel.
What it cannot remove: the channel's trace inside a multi-channel figure (PSD montages, the
montage overview). Those are redrawn without it at the patient's next 140 run.

The list of what was deleted goes to <tree>/logs/non_neural_deleted_<stamp>.tsv as well as
to the screen. A file open in a viewer is skipped and named - close it and run again.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import types
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from functions import config as cfg                      # noqa: E402

try:
    from functions import lf_io_utils as io              # noqa: E402
except ImportError as _e:                                 # a machine without mne: the name test needs none of it
    sys.modules.setdefault("mne", types.ModuleType("mne"))
    from functions import lf_io_utils as io              # noqa: E402

TREE = Path(cfg.outputs_root) / cfg.ERSP_TREE
if not TREE.is_dir():                                     # reached through another spelling of the share
    TREE = HERE / "outputs" / cfg.ERSP_TREE
# <pid>_<cond>_<ref>_<kind>_<channel>[_TN][_half1|2][_CLEAN][_GO].<ext>
FILE_RE = re.compile(r"_(?:WM|CAR|NONE)_(?:ERSP|HFAtrials|HGtrials|SIGtrials)_(?P<ch>.+?)"
                     r"(?:_TN)?(?:_half[12])?(?:_CLEAN)?(?:_GO)?\.(?:png|npy|tif|tiff|json)$")


KEEP_PHOTODIODE_IN = ("ERSP", cfg.HFA_DIR, "HFA")   # the folders whose photodiode figures stay (either layout)


def is_photodiode(ch: str) -> bool:
    return str(ch).strip().upper().startswith("PHOTO")


def channel_of(fname: str):
    m = FILE_RE.search(fname)
    return m.group("ch") if m else None


def scan_patient(pdir: Path, is_target=None, keep_photodiode=True):
    """-> (files {channel: [paths]}, tables [(path, n_rows, channels)], kept {channel: n photodiode figures left})

    `is_target(channel_name)` says which channels go; the default is the non-neural test."""
    if is_target is None:
        is_target = io._is_non_neural
    files, tables, kept = {}, [], {}
    lm = pdir / cfg.block_name
    if not lm.is_dir():
        return files, tables, kept
    for root, _dirs, names in os.walk(lm):
        for n in names:
            if n.endswith("_trial_scores.tsv"):
                p = Path(root) / n
                try:
                    ch = pd.read_csv(p, sep="\t", usecols=["channel"])["channel"].astype(str)
                except Exception as e:
                    print(f"   [warn] {p.name}: {type(e).__name__}: {e}")
                    continue
                hit = ch[ch.map(is_target)]
                if len(hit):
                    tables.append((p, int(len(hit)), sorted(hit.unique())))
                continue
            c = channel_of(n)
            if c and is_target(c):
                rel = Path(root).relative_to(lm).as_posix() if Path(root) != lm else ""
                if keep_photodiode and is_photodiode(c) and any(rel == k or rel.startswith(k + "/") for k in KEEP_PHOTODIODE_IN):
                    kept[c] = kept.get(c, 0) + 1          # the trigger check: stays
                    continue
                files.setdefault(c, []).append(Path(root) / n)
    return files, tables, kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete", action="store_true", help="delete (default: list only)")
    ap.add_argument("--patients", nargs="*", help="folder names in the tree (PAT_/EL form); default: every patient")
    ap.add_argument("--bad-listed", action="store_true",
                    help="instead of the non-neural channels: the contacts in cfg.bad_channels_manual of each patient "
                         "(for a contact bad-listed after its patient was run, removed without a rerun)")
    a = ap.parse_args()
    what = "bad-listed contacts" if a.bad_listed else "non-neural channels"

    pdirs = sorted(p for p in TREE.iterdir() if (p / cfg.block_name).is_dir())
    if a.patients:
        pdirs = [p for p in pdirs if p.name in set(a.patients)]
    print(f"tree: {TREE}   ({len(pdirs)} patients)   {'DELETING' if a.delete else 'listing only - add --delete to remove'}\n")

    log, locked, tot_f, tot_r = [], [], 0, 0
    for pdir in pdirs:
        if a.bad_listed:
            bad = {io.normalize_label(b) for b in getattr(cfg, "bad_channels_manual", {}).get(pdir.name, [])}
            target = (lambda ch, _bad=bad: io.normalize_label(ch) in _bad)
            files, tables, kept = scan_patient(pdir, target, keep_photodiode=False) if bad else ({}, [], {})
        else:
            target = io._is_non_neural
            files, tables, kept = scan_patient(pdir)
        keep_note = ("   [kept: " + ", ".join(f"{k} {v} ERSP/HFA figures" for k, v in sorted(kept.items())) + "]") if kept else ""
        if not files and not tables:
            print(f"{pdir.name:<10} nothing to remove{keep_note}")
            continue
        chans = sorted(set(files) | {c for _, _, cs in tables for c in cs})
        n_f = sum(len(v) for v in files.values())
        n_r = sum(n for _, n, _ in tables)
        by_dir = {}
        for v in files.values():
            for f in v:
                k = f.relative_to(pdir / cfg.block_name).parts[0]
                by_dir[k] = by_dir.get(k, 0) + 1
        print(f"{pdir.name:<10} {' '.join(chans):<22} {n_f:>3} files "
              f"({', '.join(f'{k} {v}' for k, v in sorted(by_dir.items()))})"
              + (f" · {n_r} score rows in {len(tables)} table(s)" if tables else "") + keep_note)
        tot_f += n_f; tot_r += n_r
        for c, v in sorted(files.items()):
            for f in v:
                done = "listed"
                if a.delete:
                    try:
                        f.unlink(); done = "deleted"
                    except PermissionError:
                        done = "LOCKED"; locked.append(f)
                log.append(dict(patient=pdir.name, channel=c, kind="file", what=str(f.relative_to(TREE)), rows="", result=done))
        for p, n, cs in tables:
            done = "listed"
            if a.delete:
                try:
                    d = pd.read_csv(p, sep="\t")
                    keep = ~d["channel"].astype(str).map(target)
                    tmp = p.with_suffix(".tsv.tmp")
                    d[keep].to_csv(tmp, sep="\t", index=False)
                    os.replace(tmp, p); done = "rows removed"
                except PermissionError:
                    done = "LOCKED"; locked.append(p)
            log.append(dict(patient=pdir.name, channel=" ".join(cs), kind="score rows", what=str(p.relative_to(TREE)), rows=n, result=done))

    print(f"\n{tot_f} files and {tot_r} score rows of {what}"
          + (" deleted" if a.delete else " found (nothing deleted)")
          + (f"; {len(locked)} could not be touched (open in another program):" if locked else ""))
    for f in locked:
        print("   " + str(f))
    if log and a.delete:                                  # a listing run leaves no trace in the tree
        out = TREE / "logs"
        out.mkdir(exist_ok=True)
        path = out / f"{'bad_listed' if a.bad_listed else 'non_neural'}_deleted_{datetime.now():%Y%m%d_%H%M%S}.tsv"
        pd.DataFrame(log).to_csv(path, sep="\t", index=False)
        print(f"list -> {path}")
    return 1 if locked else 0


if __name__ == "__main__":
    raise SystemExit(main())
