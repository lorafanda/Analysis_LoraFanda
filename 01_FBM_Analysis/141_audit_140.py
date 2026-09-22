"""
141_audit_140.py - one table per 140 run: what each patient was processed with and what came out.

    "C:/Users/fanda/AppData/Local/Programs/Python/Python311/python.exe" 01_FBM_Analysis/141_audit_140.py

Reads, per patient of cfg.patient_ids: the newest 140 log (outputs/04_ersp_LM/logs/<pid>_<stamp>.log),
the WM re-referencing report (outputs/04_ersp_LM_RAWONLY/wm_reref_report.tsv), the IQR trial report
(outputs/04_ersp_LM/<pid>/LM/Report/<pid>_IQR.tsv), the cube tree (outputs/04_ersp_LM_RAWONLY) and,
for the comparison, the previous tree (outputs/04_ersp_LM_RAWONLY_old). Config is read live, so the
bad list / reference route columns describe what the code would do NOW - if a list changed after the
run, the cube-count check at the end of the row is where it shows.

Writes outputs/04_ersp_LM/audit_140.tsv (one row per patient, every column) and audit_140.md (the same
in three readable tables), and prints the .md.
"""
import glob
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
from functions import config as cfg   # noqa: E402

QC_ROOT = os.path.join("outputs", "04_ersp_LM")
LOG_DIR = os.path.join(QC_ROOT, "logs")
NEW_ROOT = os.path.join("outputs", "04_ersp_LM_RAWONLY")
OLD_ROOT = os.path.join("outputs", "04_ersp_LM_RAWONLY_old")
CONDS = ("audio", "picture", "reading")
MICRO_RE = re.compile(r"^[A-Za-z]+m\d+$")           # ADm3, FODm12 ... the microwire bundles
CUBE_RE = re.compile(r"_(?:WM|CAR)_ERSP_(.+)_TN\.npy$")


def pid_of(raw):
    raw = str(raw)
    if raw.startswith("G-"):
        return cfg.MICROEPI_MAT_PRESETS[raw]["pat_name"]
    if raw.isdigit():
        return f"PAT_{raw}"
    return raw


def system_of(raw):
    raw = str(raw)
    if raw.startswith("G-"):
        return "MicroEPI"
    if raw.startswith("EL"):
        return "Bern"
    return "HUG"


def newest_log(pid):
    logs = sorted(glob.glob(os.path.join(LOG_DIR, f"{pid}_*.log")))
    return logs[-1] if logs else None


def parse_log(path):
    """Everything the log says about the run, per patient and per condition."""
    d = {"per_cond": {c: {} for c in CONDS}}
    if path is None:
        return d
    txt = open(path, encoding="utf-8", errors="replace").read()
    lines = txt.splitlines()
    m = re.search(r"^\[140\] (\S+)\s+\((\S+)\)\s+(\d{8}_\d{6})\s+fmax (\d+) Hz\s+qc=(\w+) export=(\w+) psd=(\w+)", txt, re.M)
    if m:
        d["stamp"] = m.group(3)
        d["fmax_hz"] = int(m.group(4))
        d["flags"] = f"qc={m.group(5)} export={m.group(6)} psd={m.group(7)}"
        t0 = datetime.strptime(m.group(3), "%Y%m%d_%H%M%S")
        t1 = datetime.fromtimestamp(os.path.getmtime(path))
        d["run_start"] = t0.strftime("%Y-%m-%d %H:%M")
        d["run_min"] = round((t1 - t0).total_seconds() / 60, 1)
    for ln in lines:
        s = ln.strip()
        if (m := re.search(r"\[microepi\] (\d+) channels = (\d+) macros \+ (\d+) micros", s)):
            d["export_macros"], d["export_micros"] = int(m.group(2)), int(m.group(3))
        elif (m := re.search(r"aux drop: (\d+) \S+ (\d+) channels", s)):
            d["aux_dropped"] = int(m.group(1)) - int(m.group(2))
        elif (m := re.search(r"WM channels from TSV: (\d+)", s)):
            d["wm_source"], d["wm_found"] = "BIDS TSV (MicroEPI)", int(m.group(1))
        elif (m := re.search(r"\[WM\] \S+: no BIDS TSV; using anatomy Lookup -> (\d+) WM channels", s)):
            d["wm_source"], d["wm_found"] = "Lookup workbook", int(m.group(1))
        elif (m := re.search(r"\[note\] \S+: per-grid CAR applied \S+ groups: (.*)", s)):
            d["car"] = "per-grid CAR: " + m.group(1)
        elif (m := re.search(r"\[note\] \S+: whole-recording CAR - mean of (\d+) channels, (\d+) bad channels left out", s)):
            d["car"] = f"whole-recording CAR: mean of {m.group(1)} ch, {m.group(2)} bad left out"
        elif (m := re.search(r"cropped to (\d+)s\.\.(\d+)s \(([\d.]+)s of ([\d.]+)s\)", s)):
            d["crop_applied"] = f"{m.group(1)}-{m.group(2)} s ({float(m.group(3)):.0f} of {float(m.group(4)):.0f} s)"
        elif (m := re.search(r"dropped (\d+) 'Unknown' channels", s)):
            d["unknown_dropped_log"] = int(m.group(1))
        elif "stripped _L#/_R#" in s:
            d["hemi_stripped"] = True
        elif (m := re.search(r"notch scope: (.*)", s)):
            d["notch_scope"] = m.group(1)
            d["notch_per_shaft"] = "PER SHAFT" in m.group(1)
        elif re.search(r"\[notch\] \S+\s+method: spectrum interpolation", s):
            d["notch_method_log"] = "interp"
        elif (m := re.search(r"Loaded TRC: shape=\((\d+), (\d+)\), fs=(\d+) Hz", s)):
            d["fs_hz"] = int(m.group(3)); d["rec_s"] = round(int(m.group(1)) / int(m.group(3)))
        elif (m := re.search(r"\[raw\] \S+: (\d+) files joined -> (\d+) samples, (\d+) s", s)):
            d["rec_s"] = int(m.group(3)); d["raw_files_joined"] = int(m.group(1))
        elif (m := re.search(r"\[raw\] \S+: \S+ -> \d+ samples @ (\d+) Hz", s)):
            d["fs_hz"] = int(m.group(1))
        elif (m := re.search(r"notch audit -> .*\((\d+) peaks\)", s)):
            d["unexplained_peaks"] = int(m.group(1))
        elif (m := re.search(r"^\[error\] \S+: (.*)", s)):
            d["error"] = m.group(1)
        elif "byte-identical to" in s:
            d["dup_trigger_tables"] = d.get("dup_trigger_tables", 0) + 1
        elif (m := re.search(r"\[\S+ \| (\w+)\] dropped (\d+) trials outside crop window", s)):
            d["per_cond"][m.group(1)]["dropped_outside_crop"] = int(m.group(2))
        elif (m := re.search(r"\[\S+ \| (\w+)\] block (\d+)s\.\.(\d+)s \((\d+) s, (\d+) trials\)", s)):
            pc = d["per_cond"][m.group(1)]
            pc["block_s"], pc["trials_used"] = int(m.group(4)), int(m.group(5))
        elif (m := re.search(r"\[\S+ \| (\w+)\] notched (\d+) of (\d+) candidates(?: over (\d+) shafts)?; (\d+) comb peaks remain, (\d+) off-comb peaks(?: - best comb fit ([\d.]+) Hz \((\d+)/(\d+)\))?", s)):
            pc = d["per_cond"][m.group(1)]
            pc["notched"] = f"{m.group(2)}/{m.group(3)}" + (f" over {m.group(4)} shafts" if m.group(4) else "")
            pc["comb_left"], pc["offcomb"] = int(m.group(5)), int(m.group(6))
            pc["comb_fit"] = f"{m.group(7)} Hz ({m.group(8)}/{m.group(9)})" if m.group(7) else ""
        elif (m := re.search(r"\[\S+ \| (\w+)\] (\d+) removed trials fall outside the block", s)):
            d["per_cond"][m.group(1)]["removed_not_drawn"] = int(m.group(2))
    d["n_shaft_lines"] = txt.count("[notch] shaft ")
    return d


def cube_names(root, pid, cond):
    out = {}
    for f in glob.glob(os.path.join(root, pid, "LM", "ERSP_matrix", cond, "*_TN.npy")):
        m = CUBE_RE.search(os.path.basename(f))
        if m:
            out[m.group(1)] = f
    return out


def alias_key(name, pid):
    """cfg_norm plus cfg.CHANNEL_SHAFT_ALIAS: the key a recording name has in the anatomy table
    and in the WM report (aI_R10 of EL042 -> ALR10), as lf_io_utils.alias_label does it."""
    s = cfg_norm(name)
    table = (getattr(cfg, "CHANNEL_SHAFT_ALIAS", {}) or {}).get(pid, {})
    m = re.match(r"^(.*?)(\d+)$", s)
    if table and m and m.group(1) in table:
        return table[m.group(1)] + m.group(2)
    return s


def classify(names, bad, wm, pid):
    """Why a cube of the old tree is not in the new one."""
    bad_n = {cfg_norm(b) for b in bad}
    wm_n = {cfg_norm(w) for w in wm}
    groups = {"bad": [], "wm_ref": [], "microwire": [], "not in config lists (Unknown drop / aux / condition not run)": []}
    for n in sorted(names):
        k = alias_key(n, pid)
        if k in bad_n:
            groups["bad"].append(n)
        elif MICRO_RE.match(n):
            groups["microwire"].append(n)
        elif k in wm_n:
            groups["wm_ref"].append(n)
        else:
            groups["not in config lists (Unknown drop / aux / condition not run)"].append(n)
    return groups


def cfg_norm(s):
    return re.sub(r"[^A-Za-z0-9]", "", str(s)).upper()


def fmt_groups(g):
    parts = []
    for k, v in g.items():
        if v:
            parts.append(f"{k} {len(v)}: {' '.join(v)}")
    return "; ".join(parts)


def main():
    rep_path = os.path.join(NEW_ROOT, "wm_reref_report.tsv")
    rep = pd.read_csv(rep_path, sep="\t", dtype=str).fillna("") if os.path.exists(rep_path) else pd.DataFrame()
    rep = rep.set_index("patient_id") if len(rep) else rep
    rows = []
    for raw in cfg.patient_ids:
        raw = str(raw)
        pid = pid_of(raw)
        sysname = system_of(raw)
        log = newest_log(pid)
        L = parse_log(log)
        r = {"patient": pid, "raw_id": raw, "system": sysname,
             "run_start": L.get("run_start", ""), "run_min": L.get("run_min", ""),
             "log": os.path.basename(log) if log else "NOT RUN",
             "fmax_hz": L.get("fmax_hz", ""), "flags": L.get("flags", "")}
        # ---- status
        rr = rep.loc[pid] if len(rep) and pid in rep.index else None
        status = rr["status"] if rr is not None else ("not run" if log is None else "no report row")
        if L.get("error"):
            status = f"{status}: {L['error']}"
        r["status"] = status
        # ---- recording / crop
        preset = (cfg.MICROEPI_MAT_PRESETS.get(raw) if sysname == "MicroEPI"
                  else cfg.EL_PRESETS.get(raw) if sysname == "Bern" else cfg.PAT_PRESETS.get(pid))
        tr = (preset or {}).get("time_range")
        r["fs_hz"] = L.get("fs_hz", "")
        r["recording_s"] = L.get("rec_s", "")
        r["crop_cfg"] = f"{tr[0]}-{tr[1]} s" if tr and tr[1] and tr[1] > 0 else ("none" if tr else "")
        r["crop_applied"] = L.get("crop_applied", "")
        r["raw_files_joined"] = L.get("raw_files_joined", "")
        # ---- channels
        r["ch_in"] = rr["n_channels_in"] if rr is not None else ""
        r["ch_neural"] = rr["n_channels_neural"] if rr is not None else ""
        r["unknown_dropped"] = rr["n_channels_unknown_dropped"] if rr is not None else L.get("unknown_dropped_log", "")
        r["aux_dropped"] = L.get("aux_dropped", "")
        r["export_micros"] = L.get("export_micros", "")
        bad = list(getattr(cfg, "bad_channels_manual", {}).get(pid, []))
        r["bad_listed_n"] = len(bad)
        r["bad_list"] = " ".join(bad)
        r["isout_listed_n"] = len(getattr(cfg, "LOOKUP_OUT_OF_BRAIN", {}).get(pid, []))
        r["hemi_stripped"] = "yes" if L.get("hemi_stripped") else ""
        r["grid_keep_prefixes"] = " ".join(getattr(cfg, "MIXED_GRID_KEEP_PREFIXES", {}).get(pid, ())) or ""
        # ---- reference
        wm_used = [w for w in (rr["wm_channels_used"].split("|") if rr is not None and rr["wm_channels_used"] else []) if w and not w.startswith("CAR:")]
        if L.get("car"):
            route = L["car"]
        elif sysname == "MicroEPI":
            route = f"WM reference, {len(wm_used)} contacts (selective: WM contacts stay as data)"
        elif pid in getattr(cfg, "MANUAL_WM_CHANNELS", {}):
            route = f"WM reference, {len(wm_used)} contacts (MANUAL list)"
        else:
            route = f"WM reference, {len(wm_used)} contacts (skipped as data)"
        r["reference"] = route
        r["wm_source"] = L.get("wm_source", "BIDS TSV" if rr is not None and wm_used else "")
        r["wm_found"] = L.get("wm_found", "")
        r["wm_n"] = len(wm_used)
        r["wm_list"] = " ".join(wm_used)
        r["wm_excluded_as_bad"] = (rr["wm_channels_excluded_as_bad"].replace("|", " ") if rr is not None else "")
        r["wm_not_reference"] = " ".join(getattr(cfg, "WM_NOT_REFERENCE", {}).get(pid, []))
        # ---- notch
        r["notch_method"] = getattr(cfg, "notch_method", {}).get(raw, getattr(cfg, "notch_method", {}).get(pid, "iir"))
        r["notch_scope"] = "per shaft" if L.get("notch_per_shaft") else ("per block" if L.get("notch_scope") else "")
        r["notch_Q_max"] = getattr(cfg, "notch_Q_max", {}).get(pid, "")
        r["notch_extra_bases"] = str(getattr(cfg, "notch_extra_bases", {}).get(pid, "")) if getattr(cfg, "notch_extra_bases", {}).get(pid) else ""
        r["unexplained_peaks"] = L.get("unexplained_peaks", "")
        r["dup_trigger_tables_skipped"] = L.get("dup_trigger_tables", "")
        # ---- per condition
        iqr_path = os.path.join(QC_ROOT, pid, "LM", "Report", f"{pid}_IQR.tsv")
        iqr = pd.read_csv(iqr_path, sep="\t").set_index("condition") if os.path.exists(iqr_path) else None
        tot_new = tot_old = 0
        change_notes = []
        shapes_all = set()
        for c in CONDS:
            pc = L["per_cond"][c]
            new = cube_names(NEW_ROOT, pid, c)
            old = cube_names(OLD_ROOT, pid, c)
            tot_new += len(new); tot_old += len(old)
            n_in = int(iqr.loc[c, "n_in"]) if iqr is not None and c in iqr.index else ""
            n_kept = int(iqr.loc[c, "n_kept"]) if iqr is not None and c in iqr.index else ""
            r[f"{c}_trials_in"] = n_in
            r[f"{c}_trials_kept_iqr"] = n_kept
            r[f"{c}_dropped_outside_crop"] = pc.get("dropped_outside_crop", "")
            r[f"{c}_trials_used"] = pc.get("trials_used", "")
            r[f"{c}_block_s"] = pc.get("block_s", "")
            r[f"{c}_notched"] = pc.get("notched", "")
            r[f"{c}_comb_left"] = pc.get("comb_left", "")
            r[f"{c}_offcomb_peaks"] = pc.get("offcomb", "")
            r[f"{c}_comb_fit"] = pc.get("comb_fit", "")
            r[f"{c}_removed_not_drawn"] = pc.get("removed_not_drawn", "")
            r[f"{c}_cubes"] = len(new)
            r[f"{c}_cubes_old"] = len(old)
            shapes = set()
            nan_files = 0
            for i, f in enumerate(sorted(new.values())):
                x = np.load(f, mmap_mode="r")
                shapes.add(x.shape)
                if i < 5 and not np.isfinite(np.asarray(x)).all():
                    nan_files += 1
            shapes_all |= shapes
            r[f"{c}_shape"] = "/".join(f"{a}x{b}" for a, b in sorted(shapes)) if shapes else ""
            r[f"{c}_nan_in_first5"] = nan_files if new else ""
            r[f"{c}_halves"] = len(glob.glob(os.path.join(NEW_ROOT, pid, "LM", "ERSP_halves", c, "*.npy")))
            r[f"{c}_clean_png"] = len(glob.glob(os.path.join(NEW_ROOT, pid, "LM", "ERSP_clean", c, "*.png")))
            r[f"{c}_qc_ersp_png"] = len(glob.glob(os.path.join(QC_ROOT, pid, "LM", "ERSP", c, "*.png")))
            r[f"{c}_qc_hg_png"] = len(glob.glob(os.path.join(QC_ROOT, pid, "LM", "HG", c, "*.png")))
            gone = set(old) - set(new)
            added = set(new) - set(old)
            if gone:
                change_notes.append(f"{c} -{len(gone)} [{fmt_groups(classify(gone, bad, wm_used, pid))}]")
            if added:
                change_notes.append(f"{c} +{len(added)} [{' '.join(sorted(added))}]")
        r["cubes_total"] = tot_new
        r["cubes_total_old"] = tot_old
        r["shapes"] = "/".join(f"{a}x{b}" for a, b in sorted(shapes_all)) if shapes_all else ""
        r["microwire_cubes_left"] = sum(1 for c in CONDS for n in cube_names(NEW_ROOT, pid, c) if MICRO_RE.match(n))
        # channels the cubes do not account for: neural - (WM skipped as data) - cubes per condition
        n_conds = sum(1 for c in CONDS if r[f"{c}_cubes"])
        if rr is not None and n_conds:
            neural = int(rr["n_channels_neural"])
            skipped_wm = 0 if (sysname == "MicroEPI" or L.get("car")) else len(wm_used)
            per_cond = tot_new / n_conds
            r["removed_other"] = int(round(neural - skipped_wm - per_cond))
            r["removed_vs_bad_listed"] = ("= bad list" if r["removed_other"] == len(bad)
                                         else f"{r['removed_other']} removed vs {len(bad)} listed")
        else:
            r["removed_other"] = ""; r["removed_vs_bad_listed"] = ""
        # exact reconciliation by NAME: the QC ERSP figures are drawn for every channel that
        # reached the ERSP stage (bad channels included; WM contacts skipped as data are not
        # drawn), so QC names - cube names = what the cube export left out, and each of those
        # names is either in the bad list or removed by something else (an aux-name rule, ...)
        first = next((c for c in CONDS if r[f"{c}_cubes"]), None)
        if first:
            qc_names = set()
            for f in glob.glob(os.path.join(QC_ROOT, pid, "LM", "ERSP", first, "*_TN.png")):
                m = re.search(r"_(?:WM|CAR)_ERSP_(.+)_TN\.png$", os.path.basename(f))
                if m:
                    qc_names.add(m.group(1))
            removed = qc_names - set(cube_names(NEW_ROOT, pid, first))
            bad_n = {cfg_norm(b) for b in bad}
            in_bad = sorted(n for n in removed if cfg_norm(n) in bad_n)
            not_bad = sorted(n for n in removed if cfg_norm(n) not in bad_n)
            r["aliased_shafts"] = " ".join(f"{k}->{v}" for k, v in (getattr(cfg, "CHANNEL_SHAFT_ALIAS", {}) or {}).get(pid, {}).items())
            r["qc_channels"] = len(qc_names)
            r["removed_by_bad_list"] = len(in_bad)
            r["removed_not_in_bad_list"] = " ".join(not_bad)
            seen = {cfg_norm(n) for n in qc_names} | {cfg_norm(w) for w in r["wm_excluded_as_bad"].split()}
            r["bad_listed_not_in_recording"] = " ".join(b for b in bad if cfg_norm(b) not in seen)
        else:
            r["qc_channels"] = r["removed_by_bad_list"] = r["removed_not_in_bad_list"] = r["bad_listed_not_in_recording"] = r["aliased_shafts"] = ""
        r["conditions_out"] = "/".join(c for c in CONDS if r[f"{c}_cubes"]) or "none"
        r["change_vs_old"] = "; ".join(change_notes) if change_notes else ("identical names" if tot_old else "no old tree")
        rows.append(r)

    df = pd.DataFrame(rows)
    out_tsv = os.path.join(QC_ROOT, "audit_140.tsv")
    df.to_csv(out_tsv, sep="\t", index=False)

    # ---- readable version
    def md_table(cols, names=None):
        names = names or cols
        sub = df[cols].copy()
        sub.columns = names
        lines = ["| " + " | ".join(names) + " |", "|" + "|".join("---" for _ in names) + "|"]
        for _, row in sub.iterrows():
            lines.append("| " + " | ".join(str(v) if v != "" else "·" for v in row.values) + " |")
        return "\n".join(lines)

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    md = [f"# 140 run audit - fmax 400 Hz - {stamp}", "",
          f"{len(df)} patients in cfg.patient_ids; {int((df.log != 'NOT RUN').sum())} with a log; "
          f"{int(df.status.eq('ok').sum())} ok. Cube trees: new `{NEW_ROOT}`, previous `{OLD_ROOT}`.", "",
          "## 1. Run, channels, reference", "",
          md_table(["patient", "system", "run_start", "run_min", "status", "crop_cfg", "ch_in", "ch_neural",
                    "unknown_dropped", "aux_dropped", "bad_listed_n", "reference", "wm_source", "wm_n",
                    "wm_excluded_as_bad", "wm_not_reference", "qc_channels", "removed_by_bad_list",
                    "removed_not_in_bad_list", "bad_listed_not_in_recording"],
                   ["patient", "sys", "start", "min", "status", "crop (cfg)", "in", "neural", "unk drop", "aux drop",
                    "bad listed", "reference", "WM source", "WM n", "WM excluded (bad)", "WM kept as data",
                    "ch at ERSP stage", "removed: bad list", "removed: not listed", "listed but not recorded"]),
          "", "## 2. Trials and cubes per condition", "",
          md_table(["patient"] + [f"{c}_{k}" for c in CONDS for k in ("trials_in", "trials_kept_iqr", "trials_used", "cubes", "cubes_old")]
                   + ["shapes", "microwire_cubes_left", "conditions_out"],
                   ["patient"] + [f"{c[:3]} {k}" for c in CONDS for k in ("in", "IQR kept", "used", "cubes", "old")]
                   + ["shape", "microwire cubes", "conditions"]),
          "", "## 3. Notch and QC", "",
          md_table(["patient", "notch_method", "notch_scope", "notch_Q_max"] + [f"{c}_{k}" for c in CONDS for k in ("notched", "comb_left", "offcomb_peaks")]
                   + ["unexplained_peaks", "audio_qc_ersp_png", "audio_qc_hg_png", "audio_halves", "audio_clean_png"],
                   ["patient", "method", "scope", "Q max"] + [f"{c[:3]} {k}" for c in CONDS for k in ("notched", "comb left", "off-comb")]
                   + ["unexplained", "QC ERSP png (aud)", "QC HG png (aud)", "halves (aud)", "CLEAN png (aud)"]),
          "", "## 4. Bad lists and WM references", ""]
    for _, row in df.iterrows():
        md.append(f"- **{row.patient}** bad ({row.bad_listed_n}): {row.bad_list or '·'}  \n"
                  f"  WM reference ({row.wm_n}): {row.wm_list or '·'}"
                  + (f"  \n  change vs old tree: {row.change_vs_old}" if row.change_vs_old != "identical names" else "  \n  cubes: same names as the old tree"))
    md_txt = "\n".join(md) + "\n"
    out_md = os.path.join(QC_ROOT, "audit_140.md")
    open(out_md, "w", encoding="utf-8").write(md_txt)
    print(md_txt)
    print(f"[done] {out_tsv}\n       {out_md}")


if __name__ == "__main__":
    main()
