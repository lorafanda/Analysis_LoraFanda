#!/usr/bin/env python
"""152_tree_audit.py - every file of the 03_ERSP tree checked against what 140 writes NOW, so the
next concat build reads nothing it should not.

    python .\152_tree_audit.py                       audit everything, list only
    python .\152_tree_audit.py --patients PAT_1327    one or more patients
    python .\152_tree_audit.py --delete               remove what the audit marks "delete"

WHAT IS CHECKED, per patient folder <pid>/ of outputs/<cfg.ERSP_TREE>:
  layout    <pid>/LM/{ERSP, ERSP_clean, ERSP_halves, ERSP_matrix, PSD_raw, PSD_clean, PerTrial}, and
            PerTrial/{HFA, Signal, Report, TrialScores, Photodiode}; condition folders only; the
            optional <pid>_overview.pdf. Any other folder or file is an orphan (old layouts, *_noref,
            copies, notebooks' leftovers).
  names     each folder has one legal file name: <pid>_<cond>_WM_ERSP_<ch>_TN.npy (cubes),
            ..._TN_half1/2.npy, ..._TN_CLEAN.png, ..._TN.png, <pid>_<cond>_WM_HFAtrials_<ch>.png,
            <pid>_<cond>_WM_SIGtrials_<ch>.png, the Report and TrialScores names. The pid and the
            condition in the name must be the folder's. "Copy", "(1)", .tif, .json sidecars, _GO,
            CAR/NONE references: orphans.
  run       the patient's last run start = the newest logs/<pid>_<stamp>.log. A cube older than
            that is stale (a rerun no longer produced it); every other file older than that is
            stale too, except PSD figures and the overview, which --no-psd runs legitimately keep.
  channels  the CURRENT channel set of a condition = the cubes written by the last run. A file of
            any other channel is an orphan (no current cube). A channel that is bad-listed
            (cfg.bad_channels_manual), a WM reference contact (wm_reref_report) or non-neural
            (lf_io_utils._is_non_neural) must have no file at all - the photodiode's two figures in
            PerTrial/Photodiode are the one exception.
  complete  every current channel must have its 2 halves, CLEAN, ERSP figure, HFA, Signal and its
            rows in the condition's trial scores (flagged, never deleted: a rerun fixes it).
  tables    trial-score rows of channels that are not current (flagged; 143 rewrites them).
  patients  tree folders without a config patient, config patients without a folder, folders
            without a log or without an OK row in wm_reref_report.

Marks "delete" = orphan / stale / forbidden channel; "flag" = incomplete, old PSD, unknown report
file, stale score rows. Writes outputs/<cfg.ERSP_TREE>/tree_audit.tsv (one row per finding) and
tree_audit_summary.tsv (per patient: current cubes per condition and the counts), prints the
summary. --delete removes the "delete" rows (files, and orphan folders whole), logging every path
to logs/tree_audit_deleted_<stamp>.tsv. It never touches the tree root, logs/ or another tree.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import types
from datetime import datetime

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from functions import config as cfg                      # noqa: E402
if "mne" not in sys.modules:                              # lf_io_utils imports mne; only its name test is needed here
    try:
        import mne                                        # noqa: F401
    except ImportError:
        sys.modules["mne"] = types.ModuleType("mne")
from functions import lf_io_utils as io                  # noqa: E402

TREE = os.path.join(HERE, "outputs", cfg.ERSP_TREE)
TASK = getattr(cfg, "block_name", "LM")
CONDS = tuple(cfg.conditions_expected)
LM_DIRS = ("ERSP", "ERSP_clean", "ERSP_halves", "ERSP_matrix", "PSD_raw", "PSD_clean", "PerTrial")
PT_DIRS = ("HFA", "Signal", "Report", "TrialScores", "Photodiode")
BAD_NAME = re.compile(r"(copy|\(\d+\)|~|\.bak$|\.tif+$|thumbs\.db|\.ipynb_checkpoints)", re.I)


def norm(ch):
    return io.normalize_label(str(ch))


def is_photodiode(ch):
    return str(ch).strip().upper().startswith("PHOTO")


REF = r"(?P<ref>WM|CAR|NONE)"          # the reference route 140 writes into the name (WM, whole/grid CAR, or none)


def patterns(pid, cond):
    p, c = re.escape(pid), re.escape(cond)
    return {
        "ERSP_matrix": re.compile(rf"^{p}_{c}_{REF}_ERSP_(?P<ch>.+?)_TN\.npy$"),
        "ERSP_halves": re.compile(rf"^{p}_{c}_{REF}_ERSP_(?P<ch>.+?)_TN_half[12]\.npy$"),
        "ERSP_clean": re.compile(rf"^{p}_{c}_{REF}_ERSP_(?P<ch>.+?)_TN_CLEAN\.png$"),
        "ERSP": re.compile(rf"^{p}_{c}_{REF}_ERSP_(?P<ch>.+?)_TN\.png$"),
        "HFA": re.compile(rf"^{p}_{c}_{REF}_HFAtrials_(?P<ch>.+?)\.png$"),
        "Signal": re.compile(rf"^{p}_{c}_{REF}_SIGtrials_(?P<ch>.+?)\.png$"),
        "Photodiode": re.compile(rf"^{p}_{c}_{REF}_(?:ERSP_(?P<ch1>.+?)_TN|HFAtrials_(?P<ch2>.+?))\.png$"),
    }


def report_ok(pid, n):
    fixed = {f"{pid}_IQR.tsv", f"{pid}_montage_overview.png", f"{pid}_notch_audit.tsv", f"{pid}_unexplained_peaks.tsv"}
    return n in fixed or bool(re.match(rf"^{re.escape(pid)}_(?:{'|'.join(CONDS)})_iqr_postDur_QC\.png$", n))


def run_start(pid):
    """The start of the newest run that ENDED OK (its status row is in the log). Newer logs without a
    status row (in progress, or died) or with ERR are reported as notes, not taken as the run."""
    logdir = os.path.join(TREE, "logs")
    logs = sorted(f for f in os.listdir(logdir) if re.match(rf"^{re.escape(pid)}_\d{{8}}_\d{{6}}\.log$", f)) if os.path.isdir(logdir) else []
    notes = []
    for name in reversed(logs):
        stamp = re.search(r"_(\d{8}_\d{6})\.log$", name).group(1)
        t = datetime.strptime(stamp, "%Y%m%d_%H%M%S")
        try:
            txt = open(os.path.join(logdir, name), encoding="utf-8", errors="replace").read()
        except OSError:
            txt = ""
        st = re.findall(r"^\s+(OK|ERR)\S*\s+" + re.escape(pid) + r"\b", txt, flags=re.M)
        if st and st[-1] == "OK":
            return t, name, notes
        notes.append(f"run {t:%Y-%m-%d %H:%M} " + ("ended with an error" if st else "has no end: in progress or died") + f" ({name})")
    return None, None, notes


def walk_files(root):
    for dp, dn, fn in os.walk(root):
        for f in fn:
            p = os.path.join(dp, f)
            try:
                yield p, os.path.getmtime(p)
            except OSError:
                yield p, None


def older(mt, t0):
    return t0 is not None and mt is not None and datetime.fromtimestamp(mt) < t0


def audit_patient(pid, report_row, bad_norm, wm_kept=False):
    """Returns (findings, summary) for one patient folder. wm_kept: the route keeps the WM
    contacts in the montage (the MicroEPI macro reference), so they legitimately have outputs."""
    F = []
    root = os.path.join(TREE, pid)

    def rel(p):
        return os.path.relpath(p, root).replace(os.sep, "/")

    def add(path, kind, reason, action, ch="", cond="", mtime=None):
        F.append(dict(patient=pid, path=rel(path), kind=kind, cond=cond, channel=ch, reason=reason, action=action,
                      mtime=(datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M") if mtime else "")))

    t0, logname, notes = run_start(pid)
    wm_used = set()
    if not wm_kept and report_row is not None and isinstance(report_row.get("wm_channels_used"), str) \
            and not report_row["wm_channels_used"].startswith("CAR:"):
        wm_used = {norm(c) for c in report_row["wm_channels_used"].split("|") if c}
    summary = dict(patient=pid, log=logname or "", run_start=(t0.strftime("%Y-%m-%d %H:%M") if t0 else ""),
                   report_status=(str(report_row["status"]) if report_row is not None else "no row"),
                   wm_contacts_kept=wm_kept, later_runs="; ".join(notes))
    for n in notes:
        add(root, "patient", n + ": its files are newer than the last OK run and are taken as current", "flag")
    if t0 is None:
        add(root, "patient", "no 140 run that ended OK: run time unknown, nothing judged stale", "flag")
    if report_row is None or str(report_row.get("status", "")).lower() != "ok":
        add(root, "patient", f"wm_reref_report status: {summary['report_status']}", "flag")

    # ---- layout
    for n in os.listdir(root):
        p = os.path.join(root, n)
        if n == TASK and os.path.isdir(p):
            continue
        if n == f"{pid}_overview.pdf" and os.path.isfile(p):
            continue
        add(p, "layout", "not part of a 140 run (patient level)", "delete")
    lm = os.path.join(root, TASK)
    if not os.path.isdir(lm):
        add(root, "layout", f"no {TASK}/ folder", "flag")
        return F, summary
    for n in os.listdir(lm):
        p = os.path.join(lm, n)
        if n in LM_DIRS and os.path.isdir(p):
            continue
        add(p, "layout", f"not part of a 140 run ({TASK} level)", "delete")
    pt = os.path.join(lm, "PerTrial")
    if os.path.isdir(pt):
        for n in os.listdir(pt):
            p = os.path.join(pt, n)
            if n in PT_DIRS and os.path.isdir(p):
                continue
            add(p, "layout", "not part of a 140 run (PerTrial level)", "delete")

    # ---- the channel folders
    folders = {k: os.path.join(lm, k) for k in ("ERSP", "ERSP_clean", "ERSP_halves", "ERSP_matrix")}
    folders.update({k: os.path.join(pt, k) for k in ("HFA", "Signal", "Photodiode")})
    files = {}            # (kind, cond) -> {normalised channel: [(path, mtime, channel), ...]}
    for kind, fdir in folders.items():
        if not os.path.isdir(fdir):
            continue
        for n in os.listdir(fdir):
            p = os.path.join(fdir, n)
            if not os.path.isdir(p):
                add(p, kind, "a file where only condition folders belong", "delete", mtime=os.path.getmtime(p))
                continue
            if n not in CONDS:
                add(p, kind, f"not a condition folder ({n})", "delete")
                continue
            pats = patterns(pid, n)
            for fp, mt in walk_files(p):
                name = os.path.basename(fp)
                if os.path.dirname(fp) != p:
                    add(fp, kind, "nested below the condition folder", "delete", cond=n, mtime=mt)
                    continue
                if BAD_NAME.search(name):
                    add(fp, kind, "copy / backup / old format in the name", "delete", cond=n, mtime=mt)
                    continue
                m = pats[kind].match(name)
                if not m:
                    add(fp, kind, "name is not what 140 writes here (other pid, condition, reference, tag or sidecar)", "delete", cond=n, mtime=mt)
                    continue
                g = m.groupdict()
                ch = g.get("ch") or g.get("ch1") or g.get("ch2")
                files.setdefault((kind, n), {}).setdefault(norm(ch), []).append((fp, mt, ch, g.get("ref")))

    # the current channels of each condition: the cubes of the last run; the route = their tag
    current, refs = {}, {}
    for cond in CONDS:
        cur = set()
        for chn, lst in files.get(("ERSP_matrix", cond), {}).items():
            for fp, mt, ch, ref in lst:
                if older(mt, t0):
                    add(fp, "ERSP_matrix", f"cube older than the last run ({summary['run_start']}): the rerun no longer produced this channel", "delete", ch, cond, mt)
                else:
                    cur.add(chn)
                    refs[ref] = refs.get(ref, 0) + 1
        current[cond] = cur
        summary[f"cubes_{cond}"] = len(cur)
    cur_ref = max(refs, key=refs.get) if refs else None
    summary["reference"] = cur_ref or ""
    if len(refs) > 1:
        add(lm, "patient", "current cubes carry more than one reference tag: " + ", ".join(f"{k} {v}" for k, v in refs.items()), "flag")

    # every file: forbidden channel, no current cube, stale
    for (kind, cond), bych in files.items():
        for chn, lst in bych.items():
            for fp, mt, ch, ref in lst:
                if cur_ref and ref != cur_ref:
                    add(fp, kind, f"reference tag {ref}: not the patient's current route ({cur_ref})", "delete", ch, cond, mt)
                    continue
                photo = is_photodiode(ch)
                if kind == "Photodiode":
                    if not photo:
                        add(fp, kind, "not the photodiode channel", "delete", ch, cond, mt)
                        continue
                elif photo:
                    add(fp, kind, "photodiode figure outside PerTrial/Photodiode", "delete", ch, cond, mt)
                    continue
                elif chn in bad_norm:
                    add(fp, kind, "bad-listed contact (cfg.bad_channels_manual)", "delete", ch, cond, mt)
                    continue
                elif chn in wm_used:
                    add(fp, kind, "WM reference contact: has no outputs", "delete", ch, cond, mt)
                    continue
                elif io._is_non_neural(ch):
                    add(fp, kind, "non-neural channel", "delete", ch, cond, mt)
                    continue
                elif kind != "ERSP_matrix" and chn not in current.get(cond, set()):
                    add(fp, kind, "no current cube for this channel (left by an earlier run)", "delete", ch, cond, mt)
                    continue
                if kind != "ERSP_matrix" and older(mt, t0):
                    add(fp, kind, f"older than the last run ({summary['run_start']})", "delete", ch, cond, mt)

    # completeness of the current channels
    for cond in CONDS:
        for chn in current.get(cond, set()):
            need = {"ERSP_halves": 2, "ERSP_clean": 1, "ERSP": 1, "HFA": 1, "Signal": 1}
            missing = [k for k, n in need.items() if len(files.get((k, cond), {}).get(chn, [])) < n]
            if missing:
                cube = files[("ERSP_matrix", cond)][chn][0]
                add(cube[0], "complete", "current cube without: " + ", ".join(missing), "flag", cube[2], cond, cube[1])
    if wm_kept:
        summary["note"] = "MicroEPI macro reference: WM contacts stay in the montage and keep their outputs"

    # Report, TrialScores, PSD, overview
    rep = os.path.join(pt, "Report")
    if os.path.isdir(rep):
        for fp, mt in walk_files(rep):
            n = os.path.basename(fp)
            if BAD_NAME.search(n):
                add(fp, "Report", "copy / backup / old format in the name", "delete", mtime=mt)
            elif not report_ok(pid, n):
                add(fp, "Report", "not a report file 140 writes", "flag", mtime=mt)
            elif older(mt, t0):
                add(fp, "Report", f"older than the last run ({summary['run_start']})", "delete", mtime=mt)
    ts = os.path.join(pt, "TrialScores")
    if os.path.isdir(ts):
        for fp, mt in walk_files(ts):
            n = os.path.basename(fp)
            m = re.match(rf"^{re.escape(pid)}_({'|'.join(CONDS)})_trial_scores\.tsv$", n)
            if not m or BAD_NAME.search(n):
                add(fp, "TrialScores", "name is not what 140 writes here", "delete", mtime=mt)
                continue
            cond = m.group(1)
            if older(mt, t0):
                add(fp, "TrialScores", f"older than the last run ({summary['run_start']})", "delete", cond=cond, mtime=mt)
                continue
            try:
                chs = pd.read_csv(fp, sep="\t", usecols=["channel"])["channel"].astype(str)
            except Exception as e:                                    # noqa: BLE001
                add(fp, "TrialScores", f"unreadable: {e}", "flag", cond=cond, mtime=mt)
                continue
            present = set(chs.map(norm))
            extra = sorted(c for c in present - current.get(cond, set()) if not is_photodiode(c))
            if extra:
                add(fp, "TrialScores", f"{len(extra)} channels in the table without a current cube: {' '.join(extra[:8])}{' ...' if len(extra) > 8 else ''}", "flag", cond=cond, mtime=mt)
            missing = sorted(current.get(cond, set()) - present)
            if missing:
                add(fp, "TrialScores", f"{len(missing)} current channels without rows: {' '.join(missing[:8])}", "flag", cond=cond, mtime=mt)
    for psd in ("PSD_raw", "PSD_clean"):
        d = os.path.join(lm, psd)
        if os.path.isdir(d):
            for fp, mt in walk_files(d):
                n = os.path.basename(fp)
                if BAD_NAME.search(n):
                    add(fp, psd, "copy / backup / old format in the name", "delete", mtime=mt)
                elif older(mt, t0):
                    add(fp, psd, "older than the last run (a --no-psd run keeps them; check)", "flag", mtime=mt)
    ov = os.path.join(root, f"{pid}_overview.pdf")
    if os.path.isfile(ov) and older(os.path.getmtime(ov), t0):
        add(ov, "overview", "older than the last run", "flag", mtime=os.path.getmtime(ov))

    summary["delete"] = sum(1 for f in F if f["action"] == "delete")
    summary["flag"] = sum(1 for f in F if f["action"] == "flag")
    return F, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", nargs="*", default=None)
    ap.add_argument("--delete", action="store_true")
    a = ap.parse_args()

    rp = os.path.join(TREE, "wm_reref_report.tsv")
    rep = pd.read_csv(rp, sep="\t") if os.path.exists(rp) else pd.DataFrame()
    rows = {str(r["patient_id"]): r for r in rep.to_dict("records")} if len(rep) else {}
    raw_of = {str(r["patient_id"]): str(r["pid_raw"]) for r in rep.to_dict("records")} if len(rep) else {}
    tree_pids = sorted(d for d in os.listdir(TREE) if os.path.isdir(os.path.join(TREE, d)) and d != "logs")
    if a.patients:
        tree_pids = [p for p in tree_pids if p in a.patients or raw_of.get(p) in a.patients]
    cfg_ids = set(str(p) for p in cfg.patient_ids)
    bad = getattr(cfg, "bad_channels_manual", {})
    all_F, summaries = [], []
    for pid in tree_pids:
        bad_norm = {norm(c) for c in (list(bad.get(pid, [])) + list(bad.get(raw_of.get(pid, ""), [])))}
        F, S = audit_patient(pid, rows.get(pid), bad_norm, wm_kept=str(raw_of.get(pid, "")).startswith("G-"))
        if pid not in cfg_ids and raw_of.get(pid) not in cfg_ids:
            F.append(dict(patient=pid, path=".", kind="patient", cond="", channel="", reason="patient folder with no entry in cfg.patient_ids", action="flag", mtime=""))
            S["flag"] = S.get("flag", 0) + 1
        all_F += F
        summaries.append(S)
        print(f"{pid:<10} run {S.get('run_start') or '-':<16} cubes " + " ".join(f"{c[:3]} {S.get('cubes_' + c, 0):>3}" for c in CONDS)
              + f"   delete {S['delete']:>4}   flag {S['flag']:>3}")
    missing = sorted(p for p in cfg_ids if p not in tree_pids and p not in raw_of.values())
    if missing and not a.patients:
        print("config patients without a folder:", " ".join(missing))

    D = pd.DataFrame(all_F, columns=["patient", "path", "kind", "cond", "channel", "reason", "action", "mtime"])
    out = os.path.join(TREE, "tree_audit.tsv")
    D.to_csv(out, sep="\t", index=False)
    pd.DataFrame(summaries).to_csv(os.path.join(TREE, "tree_audit_summary.tsv"), sep="\t", index=False)
    if len(D):
        print("\nby reason:")
        print(D.groupby(["action", "kind", "reason"]).size().sort_values(ascending=False).head(40).to_string())
    n_del = int((D.action == "delete").sum()) if len(D) else 0
    n_flag = int((D.action == "flag").sum()) if len(D) else 0
    print(f"\n{n_del} to delete, {n_flag} flagged -> {out}")

    if a.delete and n_del:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        done = []
        for r in D[D.action == "delete"].itertuples():
            p = os.path.join(TREE, r.patient, r.path)
            if not os.path.abspath(p).startswith(os.path.abspath(os.path.join(TREE, r.patient))):
                continue
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p)
                elif os.path.isfile(p):
                    os.remove(p)
                else:
                    continue
                done.append(dict(patient=r.patient, path=r.path, reason=r.reason))
            except OSError as e:
                print(f"   could not remove {r.patient}/{r.path}: {e}")
        os.makedirs(os.path.join(TREE, "logs"), exist_ok=True)
        pd.DataFrame(done).to_csv(os.path.join(TREE, "logs", f"tree_audit_deleted_{stamp}.tsv"), sep="\t", index=False)
        print(f"deleted {len(done)} -> logs/tree_audit_deleted_{stamp}.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
