"""
142_stale_cubes.py - what a 140 rerun left behind: files older than their patient's run.

    python 142_stale_cubes.py              # list, per patient
    python 142_stale_cubes.py --delete     # delete them

140 never clears a patient folder. A rerun overwrites the files of the channels it still
produces; a channel that the rerun no longer produces (newly a WM reference, newly dropped
as Unknown, newly bad-listed) keeps its cubes, halves, CLEAN and QC figures from the earlier
run, and those enter the next cohort build as if they were current (v8 carried 48 such
cubes). The run start of every patient is the audit's (outputs/04_ersp_LM/audit_140.tsv,
141_audit_140.py - run it first); every file under the patient's folders in both trees
with an mtime before that start is stale. Patients without a run are skipped.
"""
import argparse
import glob
import os
from datetime import datetime

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.join(HERE, "outputs", "04_ersp_LM", "audit_140.tsv")
TREES = {"cubes": os.path.join(HERE, "outputs", "04_ersp_LM_RAWONLY"),
         "qc": os.path.join(HERE, "outputs", "04_ersp_LM")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete", action="store_true")
    ap.add_argument("--patients", nargs="*")
    a = ap.parse_args()
    d = pd.read_csv(AUDIT, sep="\t", dtype=str).fillna("").set_index("patient")
    total, per = [], {}
    for pid, r in d.iterrows():
        if a.patients and pid not in a.patients:
            continue
        if not r["run_start"] or r["log"] == "NOT RUN":
            continue
        t0 = datetime.strptime(r["run_start"], "%Y-%m-%d %H:%M").timestamp()
        stale = []
        for tree in TREES.values():
            for f in glob.glob(os.path.join(tree, pid, "LM", "**", "*"), recursive=True):
                # Thumbs.db and the trigger-check figures (el043_build_picture_triggers.py) are not 140 products
                if os.path.isfile(f) and not f.endswith(("Thumbs.db", "_PDcheck.png")) and os.path.getmtime(f) < t0:
                    stale.append(f)
        if stale:
            per[pid] = stale
            total += stale
    if not total:
        print("no stale file: every file is younger than its patient's run")
        return 0
    for pid, files in per.items():
        kinds = {}
        for f in files:
            k = os.path.relpath(f, TREES["cubes"] if f.startswith(TREES["cubes"]) else TREES["qc"]).split(os.sep)[2]
            kinds[k] = kinds.get(k, 0) + 1
        names = sorted({os.path.basename(f).split("_ERSP_")[-1].split("_HGtrials_")[-1].rsplit("_TN", 1)[0].rsplit(".", 1)[0] for f in files})
        print(f"{pid}: {len(files)} stale files {kinds} - channels: {' '.join(names[:20])}{' ...' if len(names) > 20 else ''}")
    print(f"\n{len(total)} stale files in {len(per)} patients")
    if a.delete:
        for f in total:
            os.remove(f)
        print(f"deleted {len(total)} files")
    else:
        print("(--delete removes them)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
