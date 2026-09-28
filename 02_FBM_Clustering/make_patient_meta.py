#!/usr/bin/env python3
"""
make_patient_meta.py - one row per patient: recording centre and task language.

    python make_patient_meta.py            show what it would write
    python make_patient_meta.py --write    write/refresh patient_meta.csv

WHY A FILE AND NOT A LOOKUP. The centre is derivable from the id, but the task language
is only in the behavioural events FILENAME:

    .../task_FBM/data_LM/raw/sub_EL035_task_LanguageMapping_..._lang_GER_events.tsv

and that file is missing for seven patients of cohort v11. So the derived value is
written once into patient_meta.csv, the blanks are filled in by hand, and every later
run PRESERVES what is already in the file: a hand-entered language is never overwritten
by a re-derivation, and a patient that later gains its events file is filled in
automatically. Nothing here guesses a language from the centre - Geneva ran at least one
patient in German (PAT_3975), so the centre does not imply the language.

CENTRE. "system" in audit_140.tsv: Bern, HUG and MicroEPI. HUG and MicroEPI are both
Geneva, and the file carries both the three-way `system` and the two-way `centre`, so a
plot can use either without re-deriving it.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "01_FBM_Analysis" / "functions"))
import config as cfg  # noqa: E402

RAW = "//nasac-m2.unige.ch/m-HumanNeuronLab/DATARAW"
AUDIT = REPO / "01_FBM_Analysis" / "outputs" / "04_ersp_LM" / "audit_140.tsv"
OUT = ROOT / "patient_meta.csv"
LANG_RX = re.compile(r"lang(?:uage)?[-_]([A-Za-z]{2,4})", re.I)
CENTRE = {"Bern": "Bern", "HUG": "Geneva", "MicroEPI": "Geneva"}


def base_root(patient: str, raw_id: str) -> str:
    """Where the patient's task folder lives - the layout build_paths_for_patient uses."""
    rid = str(raw_id)
    if rid.startswith(("MicroEPI", "G-", "B-")):
        pid, root = f"MicroEPI-{rid}", f"{RAW}/MICROEPI/MicroEPI-{rid}/task/FBM"
    elif str(patient).startswith("EL"):
        pid, root = patient, f"{RAW}/SEEG_EXPERIMENTS_BERN/{patient}/task_FBM"
    else:
        pid = patient if str(patient).startswith("PAT_") else f"PAT_{patient}"
        root = f"{RAW}/SEEG_EXPERIMENTS_HUG/{pid}/task_FBM"
    ov = (getattr(cfg, "PATH_OVERRIDES", {}) or {}).get(pid) or {}
    return (ov.get("base_root") or root).replace("\\", "/")


def lang_from_disk(root: str) -> str:
    """Language codes in the behavioural filenames under a patient's task folder."""
    found = set()
    for pat in ("*/raw/*", "*/*", "*", "*/*/*", "*/*/*/*"):
        for f in glob.glob(f"{root}/{pat}"):
            if f.lower().endswith((".tsv", ".txt")):
                m = LANG_RX.search(os.path.basename(f))
                if m:
                    found.add(m.group(1).upper())
        if found:
            break
    return "/".join(sorted(found))


def lang_from_config(patient: str, raw_id: str) -> str:
    """The MicroEPI patients keep the same filename in config (tsv_file / manual_trig),
    which is the only copy reachable for the ones whose task folder is laid out
    differently."""
    keys = {str(patient).replace("-", "").upper(), str(raw_id).replace("-", "").upper()}
    found = set()
    for name in dir(cfg):
        if name.startswith("__"):
            continue
        v = getattr(cfg, name)
        if not isinstance(v, dict):
            continue
        for k, val in v.items():
            if str(k).replace("-", "").upper() in keys:
                found.update(c.upper() for c in LANG_RX.findall(str(val)))
    return "/".join(sorted(found))


def main(write: bool) -> int:
    aud = pd.read_csv(AUDIT, sep="\t")[["patient", "raw_id", "system"]]
    old = pd.read_csv(OUT).set_index("patient") if OUT.exists() else None

    rows = []
    for r in aud.itertuples():
        kept = ""
        if old is not None and r.patient in old.index:
            kept = str(old.loc[r.patient, "language"] or "").strip()
            if kept.lower() == "nan":
                kept = ""
        lang = kept or lang_from_disk(base_root(r.patient, r.raw_id)) \
            or lang_from_config(r.patient, r.raw_id)
        rows.append(dict(patient=r.patient, system=r.system,
                         centre=CENTRE.get(str(r.system), str(r.system)),
                         language=lang,
                         source=("kept" if kept else ("derived" if lang else ""))))
    d = pd.DataFrame(rows).sort_values(["centre", "patient"])

    print(d.to_string(index=False))
    known = d[d.language != ""]
    print(f"\ncentres: {d.centre.value_counts().to_dict()}")
    print(f"languages: {known.language.value_counts().to_dict()}")
    miss = list(d.loc[d.language == "", "patient"])
    if miss:
        print(f"\nNO LANGUAGE for {len(miss)}: {', '.join(miss)}"
              f"\n  -> fill the 'language' column in {OUT.name} by hand; a value there is"
              f"\n     kept on every later run.")
    if write:
        d.to_csv(OUT, index=False)
        print(f"\nwrote {OUT}")
    else:
        print("\n(--write to save)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    raise SystemExit(main(ap.parse_args().write))
