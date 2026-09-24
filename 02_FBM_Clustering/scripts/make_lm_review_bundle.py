"""
make_lm_review_bundle.py - the review bundle of LM_visualizer.html: every recorded contact of
every patient of a 140 run, with its ERSP, its anatomy and why it is (or is not) in the data.

    "C:/Users/fanda/AppData/Local/Programs/Python/Python311/python.exe" \
        02_FBM_Clustering/scripts/make_lm_review_bundle.py                      # everything (~1 h)
        ... --patients EL043 PAT_6953        # rebuild these, keep the rest of the bundle
        ... --relabel                        # labels / status only, no cube is read (~2 min)
        ... --no-halves                      # skip the split-half reliability (faster)

WHAT. The visualizer's review mode (key r) lists a patient's contacts shaft by shaft, each with
its concatenated ERSP (audio | picture | reading, undistorted), its native anatomical label
and a flag. This script builds what that mode reads, straight from the 140 OUTPUT TREE - not
from the pooling cache - so it always shows the run as it is on disk:

    activity_viz/review/manifest.json    format of the ERSP files, build stamp, source tree
    activity_viz/review/patients.json    one row per patient of cfg.patient_ids: run status,
                                         trials per condition, reference route, notch, bad count
                                         (from 01_FBM_Analysis/outputs/04_ersp_LM/audit_140.tsv,
                                         written by 141_audit_140.py - run that first)
    activity_viz/review/contacts.json    one row per contact: names, shaft, fsaverage xyz,
                                         native label (BIDS electrodes TSV tissueLabel, or the
                                         Lookup workbook for EL051/EL052), fsaverage aparc label,
                                         STATUS, flags, per-condition numbers (split-half
                                         reliability, mains-row stripe index, mean |dB|, HG)
    activity_viz/review/ersp/*.bin       uint8 [3, 103, 150] for every contact with cubes - the
                                         same format as activity_viz/ersp/c<i>.bin (byte 0 = no
                                         data, 1..255 = -10..+10 dB, time averaged in pairs);
                                         each contact row names its file

STATUS of a contact (why it is or is not in the cubes):
    data          has ERSP cubes (MicroEPI WM contacts too: their reference keeps them as data)
    bad           in cfg.bad_channels_manual (whether it reached the ERSP stage or not)
    wm_ref        used in the white-matter reference and skipped as data (EL / PAT)
    unknown       first tissueLabel token "Unknown": dropped by the parcellation step
    aux           reached the ERSP stage but left out by a name rule (EKG-, photo, E1...);
                  not a contact, so not listed since 2026-09-23 - only counted (n_status.aux)
    not_recorded  in the electrodes table but nowhere in the run
    not_run       the patient has no 140 log

The "at the ERSP stage" set is the QC figure set (outputs/04_ersp_LM/<pid>/LM/ERSP/<cond>/),
which 140 draws for every channel it processes, bad ones included.

NAMES. Recording and anatomy table are joined with the key precompute_activity_cube.py uses
(norm_el: alphanumerics, upper-case), so A_L10 == AL10. Where a shaft is spelled differently
on the two sides (SHAFT_ALIAS below, and the I<->l swap of the insula shafts) the join goes
through the alias and the contact is flagged "name mismatch ... not linked in 140" - because
140 itself applies no alias: such a contact was analysed with no anatomy, outside the WM and
Unknown rules. Recording names that match nothing at all get label "?" and are listed at the
end of the log.

A per-patient rebuild (--patients) replaces those patients' rows and files and keeps every
other patient's as they are, so a 140 rerun of one patient costs one patient here.
"""
import argparse
import glob
import io
import json
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
A01 = os.path.join(REPO, "01_FBM_Analysis")
sys.path.insert(0, A01)
from functions import config as cfg   # noqa: E402

NASAC = "//nasac-m2.unige.ch/m-HumanNeuronLab"
CUBES = os.path.join(A01, "outputs", "04_ersp_LM_RAWONLY")
QC = os.path.join(A01, "outputs", "04_ersp_LM")
AUDIT = os.path.join(QC, "audit_140.tsv")
RECON = os.path.join(REPO, "02_FBM_Clustering", "outputs", "250_recon", "fsaverage")
BUNDLE = os.path.join(RECON, "activity_viz")
OUT = os.path.join(BUNDLE, "review")

CONDS = ["audio", "picture", "reading"]
NF, NT, T_DS, VLIM = 103, 300, 2, 10.0
F_STEP = 3.90625
CUBE_RE = re.compile(r"_(?:WM|CAR)_ERSP_(.+)_TN\.npy$")
PNG_RE = re.compile(r"_(?:WM|CAR)_ERSP_(.+)_TN\.png$")
HG_RE = re.compile(r"_(WM|CAR)_HGtrials_")
MICRO_RE = re.compile(r"^[A-Za-z]+m\d+$")

# Desikan-Killiany names, readable
DK = {
    "bankssts": "banks of STS", "caudalanteriorcingulate": "caudal anterior cingulate",
    "caudalmiddlefrontal": "caudal middle frontal", "cuneus": "cuneus", "entorhinal": "entorhinal",
    "frontalpole": "frontal pole", "fusiform": "fusiform", "inferiorparietal": "inferior parietal",
    "inferiortemporal": "inferior temporal", "insula": "insula", "isthmuscingulate": "isthmus cingulate",
    "lateraloccipital": "lateral occipital", "lateralorbitofrontal": "lateral orbitofrontal",
    "lingual": "lingual", "medialorbitofrontal": "medial orbitofrontal", "middletemporal": "middle temporal",
    "paracentral": "paracentral", "parahippocampal": "parahippocampal", "parsopercularis": "pars opercularis",
    "parsorbitalis": "pars orbitalis", "parstriangularis": "pars triangularis", "pericalcarine": "pericalcarine",
    "postcentral": "postcentral", "posteriorcingulate": "posterior cingulate", "precentral": "precentral",
    "precuneus": "precuneus", "rostralanteriorcingulate": "rostral anterior cingulate",
    "rostralmiddlefrontal": "rostral middle frontal", "superiorfrontal": "superior frontal",
    "superiorparietal": "superior parietal", "superiortemporal": "superior temporal",
    "supramarginal": "supramarginal", "temporalpole": "temporal pole",
    "transversetemporal": "transverse temporal (Heschl)", "unknown": "unknown",
}

# Recording shaft -> anatomy-table shaft, both normalised, where the two spell a shaft
# differently: RECON_ALIAS of 250_recon_fsaverage.ipynb, inverted. On top of it an unmatched
# recording shaft is tried with I<->l swapped (the TSVs write the insula shafts alL / alR,
# the recordings aI_L / aI_R).
SHAFT_ALIAS = getattr(cfg, "CHANNEL_SHAFT_ALIAS", None) or {   # 140's own table since 2026-09-22
    "EL034": {"MFGL": "MFG"}, "EL043": {"PINS": "PL"}, "EL045": {"PLATL": "PLANTL"},
    "EL046": {"AIL": "ALL", "PIL": "PLL"}, "PAT_6619": {"OFA": "OFAD", "OFP": "OFPD"},
}


def norm_el(s) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(s)).upper()


def shaft_of(name: str) -> str:
    return re.sub(r"[\s_-]*\d+$", "", str(name))


def num_of(name: str) -> int:
    m = re.search(r"(\d+)$", str(name))
    return int(m.group(1)) if m else 0


def resolve_alias(pid: str, name: str, anat: dict):
    """The anatomy key of a recording name whose shaft is spelled differently, or None."""
    sh, num = shaft_of(name), re.search(r"\d+$", str(name))
    if not num:
        return None
    k = norm_el(sh)
    cands = [SHAFT_ALIAS.get(pid, {}).get(k)]
    swapped = k.replace("I", "L")
    if swapped != k:
        cands.append(swapped)
    for c in cands:
        if c and (c + num.group(0)) in anat:
            return c + num.group(0)
    return None


def pid_of(raw) -> str:
    raw = str(raw)
    if raw.startswith("G-"):
        return cfg.MICROEPI_MAT_PRESETS[raw]["pat_name"]
    return f"PAT_{raw}" if raw.isdigit() else raw


def system_of(raw) -> str:
    raw = str(raw)
    return "MicroEPI" if raw.startswith("G-") else ("Bern" if raw.startswith("EL") else "HUG")


# ---- anatomy --------------------------------------------------------------------------
def parse_token(tok: str):
    """One tissueLabel token -> (tissue, region, hemi). tissue: WM | GM | sub | unknown."""
    t = str(tok).strip()
    m = re.match(r"^(wm|ctx)-(lh|rh)-(.+)$", t)
    if m:
        return ("WM" if m.group(1) == "wm" else "GM", DK.get(m.group(3), m.group(3)), "L" if m.group(2) == "lh" else "R")
    m = re.match(r"^(wm|ctx)_(lh|rh)_(.+)$", t)          # Destrieux, as the Lookup workbooks write it
    if m:
        return ("WM" if m.group(1) == "wm" else "GM", m.group(3).replace("_", " ").replace("-", " "), "L" if m.group(2) == "lh" else "R")
    m = re.match(r"^(Left|Right)-(.+)$", t)
    if m:
        name = m.group(2)
        if re.search(r"white-?matter", name, re.I):
            return ("WM", "white matter (unsegmented)", m.group(1)[0])
        return ("sub", name.replace("-", " "), m.group(1)[0])
    if t.lower() == "unknown":
        return ("unknown", "unknown", "")
    return ("?", t, "")


def native_from_tsv(row) -> dict:
    toks = str(row.get("tissueLabel", "")).split()
    ws = []
    for k in range(1, 8):
        v = row.get(f"tissueWeights_{k}", "")
        try:
            ws.append(float(v))
        except (TypeError, ValueError):
            ws.append(np.nan)
    ws = ws[:len(toks)]
    parts = [parse_token(t) for t in toks]
    wm = sum(w for (t, _, _), w in zip(parts, ws) if t == "WM" and w == w)
    gm = sum(w for (t, _, _), w in zip(parts, ws) if t == "GM" and w == w)
    sub = sum(w for (t, _, _), w in zip(parts, ws) if t == "sub" and w == w)
    unk = sum(w for (t, _, _), w in zip(parts, ws) if t == "unknown" and w == w)
    first = parts[0] if parts else ("?", "?", "")
    hemi = first[2] or str(row.get("hemisphere", "")).strip()[:1].upper()
    # the first named structure names the place; the first token decides the tissue
    region = first[1]
    for (t, r, _) in parts:
        if t in ("GM", "sub") or (t == "WM" and r != "white matter (unsegmented)"):
            region = r
            break
    return {"raw": " ".join(f"{t} {w:.2f}" if w == w else t for t, w in zip(toks, ws)) or "",
            "tissue": first[0], "region": region, "hemi": hemi,
            "wm": round(wm, 2), "gm": round(gm, 2), "sub": round(sub, 2), "unknown": round(unk, 2),
            # 140's rule (lf_io_utils.derive_wm_channels_from_electrodes_tsv): first token starts with
            # "wm-" and its weight is above 0.97 - "Left-UnsegmentedWhiteMatter" does not count
            "unknown_first": first[0] == "unknown",
            "wm_rule": bool(toks) and toks[0].startswith("wm-") and bool(ws) and ws[0] == ws[0] and ws[0] > 0.97}


def label_text(nat: dict) -> str:
    if not nat:
        return "?"
    h = (nat["hemi"] + " ") if nat["hemi"] else ""
    pct = []
    if nat["gm"]:
        pct.append(f"GM {nat['gm']*100:.0f} %")
    if nat["wm"]:
        pct.append(f"WM {nat['wm']*100:.0f} %")
    if nat["sub"]:
        pct.append(f"sub {nat['sub']*100:.0f} %")
    if nat["unknown"]:
        pct.append(f"unknown {nat['unknown']*100:.0f} %")
    return f"{h}{nat['region']}" + (" · " + " / ".join(pct) if pct else "")


def tsv_pattern(raw: str, pid: str) -> str:
    if raw.startswith("G-"):
        return cfg.MICROEPI_MAT_PRESETS[raw]["electrodes_tsv"].replace("\\", "/")
    if pid.startswith("EL"):
        return f"{NASAC}/DATARAW/BIDS_elec/SEEG-BERN/sub-{pid}/ieeg/*_electrodes.tsv"
    return f"{NASAC}/DATARAW/BIDS_elec/SEEG-HUG/sub-{pid.replace('PAT_', '')}/ieeg/*_electrodes.tsv"


def anatomy_table(raw: str, pid: str) -> tuple[dict, str]:
    """norm_el(name) -> {name_tsv, native, hemi}; and the source it came from."""
    out = {}
    if pid in getattr(cfg, "LOOKUP_ANATOMY_DIRS", {}) and pid in getattr(cfg, "LOOKUP_ANATOMY_PATIENTS", set()):
        d = cfg.LOOKUP_ANATOMY_DIRS[pid].replace("\\", "/")
        best, best_n = None, -1
        for f in sorted(glob.glob(d + "/*Lookup*.xls*")):
            try:
                df = pd.read_excel(f, sheet_name="channels")
            except Exception:
                continue
            if "natus" not in df.columns:
                continue
            n = int((~df["natus"].astype(str).str.strip().str.lower().isin(("unplugged", "nan", ""))).sum())
            if n > best_n:
                best, best_n = (f, df), n
        if best is None:
            return out, "no Lookup"
        f, df = best
        lead = df[df["type"].astype(str).str.lower().isin(("lead", "micro"))]
        lead = lead[~lead["natus"].astype(str).str.strip().str.lower().isin(("unplugged", "nan", ""))]
        for _, r in lead.iterrows():
            tok = str(r.get("destrieux_corrected", "") or r.get("destrieux", "") or "").strip()
            tissue, region, hemi = parse_token(tok) if tok and tok != "nan" else ("?", str(r.get("label", "")), "")
            is_wm = str(r.get("isWM", "")).strip().lower() in ("1", "1.0", "true")
            is_out = str(r.get("isOut", "")).strip().lower() in ("1", "1.0", "true")
            if is_wm and tissue != "WM":
                tissue = "WM"
            nat = {"raw": f"{tok} · {r.get('region', '')} / {r.get('label', '')}".strip(" ·/"),
                   "tissue": tissue, "region": region or str(r.get("label", "")),
                   "hemi": hemi or str(r.get("side", "")).strip()[:1].upper(),
                   "wm": 1.0 if is_wm else 0.0, "gm": 0.0 if is_wm else (1.0 if tissue == "GM" else 0.0), "sub": 1.0 if tissue == "sub" else 0.0,
                   "unknown": 0.0, "unknown_first": False, "wm_rule": is_wm and not is_out, "is_out": is_out}
            out[norm_el(r["natus"])] = {"name_tsv": str(r["natus"]), "native": nat, "hemi": nat["hemi"]}
        return out, os.path.basename(f)
    fs = glob.glob(tsv_pattern(raw, pid))
    if not fs:
        return out, "no electrodes TSV"
    t = pd.read_csv(fs[0], sep="\t", dtype=str).fillna("")
    for _, r in t.iterrows():
        nat = native_from_tsv(r)
        out[norm_el(r["name"])] = {"name_tsv": str(r["name"]), "native": nat,
                                   "hemi": nat["hemi"] or str(r.get("hemisphere", "")).strip()[:1].upper()}
    return out, os.path.basename(fs[0])


# ---- cubes ------------------------------------------------------------------------------
def cube_files(pid: str, cond: str) -> dict:
    out = {}
    for f in glob.glob(os.path.join(CUBES, pid, "LM", "ERSP_matrix", cond, "*_TN.npy")):
        m = CUBE_RE.search(os.path.basename(f))
        if m:
            out[m.group(1)] = f
    return out


def qc_names(pid: str) -> set:
    out = set()
    for cond in CONDS:
        for f in glob.glob(os.path.join(QC, pid, "LM", "ERSP", cond, "*_TN.png")):
            m = PNG_RE.search(os.path.basename(f))
            if m:
                out.add(m.group(1))
    return out


def hg_tag(pid: str) -> str:
    for cond in CONDS:
        fs = glob.glob(os.path.join(QC, pid, "LM", "HG", cond, "*_HGtrials_*.png"))
        if fs:
            m = HG_RE.search(os.path.basename(fs[0]))
            return m.group(1) if m else "WM"
    return "WM"


HARM_ROWS = [int(round(h / F_STEP)) for h in (50, 100, 150, 200, 250, 300, 350)]
HG_ROWS = slice(int(round(70 / F_STEP)), int(round(150 / F_STEP)) + 1)


def metrics(A: np.ndarray) -> dict:
    with np.errstate(invalid="ignore"):
        power = float(np.nanmean(np.abs(A)))
        hg = float(np.nanmean(A[HG_ROWS, :NT // 2]))
        rows = np.nanmean(A, axis=1)
        st = [abs(rows[r] - 0.5 * (rows[r - 2] + rows[r + 2])) for r in HARM_ROWS if r + 2 < A.shape[0]]
        stripe = float(np.nanmean(st)) if st else float("nan")
    return {"power": round(power, 3), "hg": round(hg, 3), "stripe": round(stripe, 3)}


def reliability(stem_path: str) -> float:
    h1 = stem_path.replace("ERSP_matrix", "ERSP_halves").replace("_TN.npy", "_TN_half1.npy")
    h2 = h1.replace("_half1.npy", "_half2.npy")
    if not (os.path.exists(h1) and os.path.exists(h2)):
        return float("nan")
    a, b = np.load(h1).ravel(), np.load(h2).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 100:
        return float("nan")
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def quantise(A: np.ndarray) -> np.ndarray:
    m = A.reshape(NF, NT // T_DS, T_DS)
    with np.errstate(invalid="ignore"):
        m = np.nanmean(m, axis=2)
    q = np.clip((m + VLIM) * (254.0 / (2 * VLIM)), 0, 254) + 1.0
    q[~np.isfinite(m)] = 0
    return q.astype(np.uint8)


def nan_to_none(v):
    return None if (isinstance(v, float) and v != v) else v


def load_json(path, default):
    return json.load(io.open(path, encoding="utf-8")) if os.path.exists(path) else default


# ---- one patient ----------------------------------------------------------------------------
def prep_dir_of(pid, audit):
    """The prep0 folder the run read, from the run's log (the audit names the log)."""
    log = str(audit.loc[pid, "log"]) if audit is not None and pid in audit.index else ""
    path = os.path.join(QC, "logs", log)
    if not log or not os.path.exists(path):
        return None
    m = re.search(r"(?:Prep dir|prep_dir):\s*(\S+)", io.open(path, encoding="utf-8", errors="replace").read())
    return m.group(1) if m else None


def read_trial_tables(prep_dir):
    """Every row of the run's trial tables (byte-identical copies skipped, like lf_trials)."""
    import hashlib
    seen, parts = set(), []
    for f in sorted(glob.glob(os.path.join(prep_dir, "*.tsv"))):
        h = hashlib.md5(open(f, "rb").read()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        try:
            t = pd.read_csv(f, sep="\t")
        except Exception:
            continue
        lc = {c.lower(): c for c in t.columns}
        if not {"sample", "sample_offsets", "trial_end"} <= set(lc):
            continue
        t = t.rename(columns={lc[k]: k for k in ("sample", "sample_offsets", "trial_end")})
        t["_idx"] = t[lc["trial_idx"]] if "trial_idx" in lc else ""
        t["_acc"] = t[lc["resp_accuracy"]].astype(str) if "resp_accuracy" in lc else ""
        parts.append(t)
    return pd.concat(parts) if parts else None


def fs_of(pid, audit, tables):
    """Sampling rate of the tables: the audit's fs, else implied by the IQR report's post_med."""
    v = str(audit.loc[pid, "fs_hz"]) if audit is not None and pid in audit.index else ""
    try:
        if v and float(v) > 0:
            return float(v)
    except ValueError:
        pass
    iqr = os.path.join(QC, pid, "LM", "Report", f"{pid}_IQR.tsv")
    if tables is None or not os.path.exists(iqr):
        return None
    r = pd.read_csv(iqr, sep="\t")
    post_med = float(r["post_med"].median())
    smp = float((tables["trial_end"] - tables["sample_offsets"]).median())
    if not post_med > 0:
        return None
    return float(min((512, 1024, 2048, 4096, 30000), key=lambda c: abs(c - smp / post_med)))


def write_trials(pid, audit):
    """review/trials/<pid>.json: every trial of the run's tables with the reason it was dropped,
    computed by the same function 140 uses (lf_trials.collect_trials, the cohort's settings)."""
    from functions import lf_trials as LT   # noqa: E402  (01_FBM_Analysis/functions is on sys.path)
    prep_dir = prep_dir_of(pid, audit)
    if not prep_dir or not os.path.isdir(prep_dir):
        return None
    tables = read_trial_tables(prep_dir)
    fs = fs_of(pid, audit, tables)
    if tables is None or fs is None:
        return None
    # the rule values THE RUN used come from its IQR report, not from the live config (which
    # may already carry the next run's values); min_stim_s is not in the report -> config
    rules = {"min_post_s": float(cfg.min_post_s), "max_post_s": float(cfg.max_post_s), "iqr_k": float(cfg.iqr_k), "source": "config"}
    iqr = os.path.join(QC, pid, "LM", "Report", f"{pid}_IQR.tsv")
    if os.path.exists(iqr):
        r0 = pd.read_csv(iqr, sep="\t").iloc[0]
        rules = {"min_post_s": float(r0["min_post_s"]), "max_post_s": float(r0["max_post_s"]), "iqr_k": float(r0["iqr_k"]), "source": "the run's IQR report"}
    all_trials = {}
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        LT.collect_trials(prep_dir, fs, outlier_method="IQR", iqr_k=rules["iqr_k"], report_path=None, patient_id=pid,
                          max_post_s=rules["max_post_s"], min_stim_s=cfg.min_stim_s, min_post_s=rules["min_post_s"],
                          all_out=all_trials, bad_spans=getattr(cfg, "bad_time_spans", {}).get(pid, []),
                          bad_spans_pre_s=abs(float(cfg.baseline_w[0])))
    meta = {int(s): (str(i), a) for s, i, a in zip(tables["sample"], tables["_idx"], tables["_acc"])}
    out = {"patient": pid, "fs": fs, "prep_dir": prep_dir.replace("\\", "/"),
           "rules": {"accuracy": "correct / valid / 1", "min_stim_s": cfg.min_stim_s, **rules},
           "columns": ["trial_idx", "onset_s", "stim_s", "post_s", "accuracy", "kept", "reason"], "conditions": {}}
    for cond, d in all_trials.items():
        rows = []
        for on, off, te, keep, why in zip(d["on"], d["off"], d["tend"], d["keep"], d["reason"]):
            idx, acc = meta.get(int(on), ("", ""))
            rows.append([idx, round(on / fs, 2), round((off - on) / fs, 2), round((te - off) / fs, 2), acc, int(bool(keep)), str(why)])
        out["conditions"][cond] = {"n_in": int(d["n_in"]), "n_kept": int(d["n_kept"]), "rows": rows}
    os.makedirs(os.path.join(OUT, "trials"), exist_ok=True)
    io.open(os.path.join(OUT, "trials", f"{pid}.json"), "w", encoding="utf-8").write(json.dumps(out, separators=(",", ":")))
    return out


def build_patient(raw, pid, audit, coords, aparc, args, prev, tmp_prefix):
    """The contact rows of one patient (ERSP files written under tmp_prefix + k) and its patients.json row."""
    sysname = system_of(raw)
    au = audit.loc[pid] if audit is not None and pid in audit.index else None
    ran = au is not None and au["log"] != "NOT RUN"
    anat, anat_src = anatomy_table(raw, pid)
    cubes = {c: cube_files(pid, c) for c in CONDS}
    data_names = set().union(*[set(v) for v in cubes.values()])
    data_keys = {norm_el(n) for n in data_names}
    qc = qc_names(pid)
    qc_keys = {norm_el(n) for n in qc}
    bad_raw = list(getattr(cfg, "bad_channels_manual", {}).get(pid, []))
    bad = {norm_el(b) for b in bad_raw}
    wm_used_raw = [w for w in (au["wm_list"].split() if au is not None else []) if w]
    wm_used = {norm_el(w) for w in wm_used_raw}
    wm_not_ref = {norm_el(w) for w in getattr(cfg, "WM_NOT_REFERENCE", {}).get(pid, [])}
    print(f"[{pid}] {sysname} anatomy: {anat_src} ({len(anat)} contacts) · cubes {[len(cubes[c]) for c in CONDS]} · QC {len(qc)} · bad {len(bad)} · WM ref {len(wm_used)}")

    # every name we know of, keyed by norm_el; the recording spelling first, the anatomy table's, the report's
    names = {}
    for n in sorted(data_names) + sorted(qc):
        names.setdefault(norm_el(n), n)
    for k, v in anat.items():
        names.setdefault(k, v["name_tsv"])
    for n in wm_used_raw:
        names.setdefault(norm_el(n), n)

    rows, unmatched, n_alias, n_tmp, n_aux = [], [], 0, 0, 0
    # shafts the recording's cubes or the anatomy table know; a bad-listed name on no such
    # shaft (EKG-, Chin-, y1..y4) is not a contact
    known_shafts = {norm_el(shaft_of(n)) for n in data_names} | {norm_el(shaft_of(v["name_tsv"])) for v in anat.values()}
    for key, name in names.items():
        a, aliased = anat.get(key), None
        if a is None:
            ak = resolve_alias(pid, name, anat)
            if ak:
                a, aliased = anat[ak], anat[ak]["name_tsv"]
                n_alias += 1
        nat = a["native"] if a else None
        in_data, in_qc = key in data_keys, key in qc_keys
        if not ran:
            status = "not_run"
        elif in_data:
            status = "data"
        elif key in bad:
            status = "bad"
        elif key in wm_used:
            status = "wm_ref"
        elif in_qc:
            n_aux += 1                      # trigger / photodiode / EKG names: counted, not shown
            continue
        elif nat and nat.get("unknown_first"):
            status = "unknown"
        elif nat and nat.get("is_out"):
            status = "bad"
        else:
            status = "not_recorded"
        flags = []
        if nat and nat.get("wm_rule"):
            flags.append("WM by rule")
        if key in wm_not_ref:
            flags.append("WM kept as data (WM_NOT_REFERENCE)")
        if status == "data" and key in wm_used:
            flags.append("in reference and kept as data")
        if key in bad and key in wm_used:
            flags.append("BAD-LISTED AND IN THE WM REFERENCE")
        if key in bad and status == "data":
            flags.append("BAD-LISTED BUT HAS CUBES")
        if MICRO_RE.match(name):
            flags.append("microwire")
        if aliased:
            flags.append(f"name mismatch: TSV {aliased} - not linked in 140")
        if nat is None and status != "not_recorded":
            if status != "data" and not MICRO_RE.match(name) and norm_el(shaft_of(name)) not in known_shafts:
                n_aux += 1                  # bad-listed non-electrode names (EKG-, Chin-, y1..y4): counted, not shown
                continue
            unmatched.append(name)
        co = coords.get((pid, key))
        row = {
            "patient": pid, "name": name, "shaft": shaft_of(name), "num": num_of(name),
            "name_tsv": a["name_tsv"] if a else None,
            "hemi": (a["hemi"] if a else (str(co["hemi"]) if co is not None else "")) or "",
            "x": nan_to_none(float(co["x"])) if co is not None else None,
            "y": nan_to_none(float(co["y"])) if co is not None else None,
            "z": nan_to_none(float(co["z"])) if co is not None else None,
            "dist_to_pial_mm": nan_to_none(round(float(co["dist_to_pial_mm"]), 1)) if co is not None else None,
            "label": label_text(nat) if nat else "?",
            "native": nat, "aparc": aparc.get((pid, key)),
            "status": status, "flags": flags, "conds": [c for c in CONDS if name in cubes[c]],
            "file": None, "m": {},
        }
        if status == "data" and args.relabel and (pid, name) in prev and prev[(pid, name)]["file"]:
            row["m"], row["file"] = prev[(pid, name)]["m"], prev[(pid, name)]["file"]
        elif status == "data":
            arr = np.zeros((len(CONDS), NF, NT // T_DS), dtype=np.uint8)
            for ci, c in enumerate(CONDS):
                f = cubes[c].get(name)
                if not f:
                    continue
                A = np.load(f).astype(np.float32)
                if A.shape != (NF, NT):
                    print(f"   [skip] {pid} {c} {name}: shape {A.shape}")
                    continue
                arr[ci] = quantise(A)
                mm = metrics(A)
                if not args.no_halves:
                    mm["rel"] = round(reliability(f), 3)
                row["m"][c] = {k: nan_to_none(v) for k, v in mm.items()}
            row["file"] = f"{tmp_prefix}{n_tmp}.bin"
            n_tmp += 1
            arr.tofile(os.path.join(OUT, row["file"]))
        rows.append(row)

    # one shaft, one spelling: contacts of a shaft that the recording names aH_L and the
    # anatomy table aHL (or alR for aI_R) are grouped together, under the recording's spelling
    disp = {}
    for r in rows:
        if r["status"] == "data":
            disp.setdefault(norm_el(r["shaft"]), r["shaft"])
    for r in rows:
        r["shaft_key"] = norm_el(r["shaft"])
        if r["shaft_key"] not in disp:
            cand = [k for k in disp if k.replace("I", "L") == r["shaft_key"] or SHAFT_ALIAS.get(pid, {}).get(k) == r["shaft_key"]]
            if len(cand) == 1:
                r["shaft_key"] = cand[0]
        r["shaft"] = disp.get(r["shaft_key"], r["shaft"])
    rows.sort(key=lambda r: (r["shaft_key"], r["num"], r["name"]))
    if n_alias:
        print(f"   [alias] {n_alias} recording names linked to the anatomy table through a shaft alias")
    if unmatched:
        print(f"   [no anatomy row] {len(unmatched)}: {' '.join(sorted(unmatched)[:30])}")

    trials = {}
    if au is not None:
        for c in CONDS:
            trials[c] = {"in": au.get(f"{c}_trials_in", ""), "kept": au.get(f"{c}_trials_kept_iqr", ""),
                         "used": au.get(f"{c}_trials_used", ""), "cubes": au.get(f"{c}_cubes", "")}
    prow = {
        "patient": pid, "raw_id": raw, "system": sysname, "ran": ran,
        "run_start": au["run_start"] if au is not None else "", "status": au["status"] if au is not None else "not run",
        "reference": au["reference"] if au is not None else "", "wm_n": int(au["wm_n"] or 0) if au is not None else 0,
        "wm_source": au["wm_source"] if au is not None else "", "bad_n": len(bad_raw), "bad_list": bad_raw,
        "notch": f"{au['notch_method']} · {au['notch_scope']}" if au is not None else "",
        "crop": au["crop_cfg"] if au is not None else "", "conditions": au["conditions_out"] if au is not None else "",
        "unexplained_peaks": au["unexplained_peaks"] if au is not None else "",
        "trials": trials, "hg_reref": hg_tag(pid), "anatomy_source": anat_src,
        "n_contacts": len(rows), "n_data": sum(1 for r in rows if r["status"] == "data"),
        "n_status": {**{s: sum(1 for r in rows if r["status"] == s) for s in ("data", "wm_ref", "bad", "unknown", "not_recorded", "not_run")}, "aux": n_aux},
    }
    return rows, prow


# ---- main ---------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-halves", action="store_true", help="skip the split-half reliability (faster)")
    ap.add_argument("--patients", nargs="*", help="rebuild only these patient ids (PAT_/EL form); the others are kept from the existing bundle")
    ap.add_argument("--relabel", action="store_true",
                    help="rewrite contacts.json / patients.json only: ERSP files and numbers are taken from the existing bundle, no cube is read")
    args = ap.parse_args()

    audit = pd.read_csv(AUDIT, sep="\t", dtype=str).fillna("").set_index("patient") if os.path.exists(AUDIT) else None
    if audit is None:
        print(f"[warn] {AUDIT} missing - run 01_FBM_Analysis/141_audit_140.py first; patients.json will be thin")
    coords = {}
    for f in glob.glob(os.path.join(RECON, "coords", "*_contacts_fsaverage.csv")):
        if "ALL_PATIENTS" in f:
            continue
        for _, r in pd.read_csv(f).iterrows():
            coords[(str(r["patient"]), norm_el(r["name"]))] = r
    aparc = {}
    ap_path = os.path.join(RECON, "aparc_lookup.csv")
    if os.path.exists(ap_path):
        for _, r in pd.read_csv(ap_path).iterrows():
            aparc[(str(r["patient"]), norm_el(r["electrode"]))] = str(r["aparc_label"])

    os.makedirs(os.path.join(OUT, "ersp"), exist_ok=True)
    old_rows = load_json(os.path.join(OUT, "contacts.json"), [])
    old_pats = {p["patient"]: p for p in load_json(os.path.join(OUT, "patients.json"), [])}
    old_by_pat = {}
    for r in old_rows:
        old_by_pat.setdefault(r["patient"], []).append(r)
    prev = {(r["patient"], r["name"]): r for r in old_rows} if args.relabel else {}
    partial = bool(args.patients)
    if not partial and not args.relabel:
        for old in glob.glob(os.path.join(OUT, "ersp", "*.bin")):
            os.remove(old)

    patients, contacts = [], []
    for raw in cfg.patient_ids:
        raw = str(raw)
        pid = pid_of(raw)
        if partial and pid not in args.patients:
            if pid in old_by_pat:                      # kept as it is, files included
                contacts.extend(old_by_pat[pid])
                patients.append(old_pats.get(pid, {"patient": pid, "ran": False, "status": "not in bundle", "trials": {}, "n_status": {}}))
            continue
        if partial and not args.relabel:               # this patient's old files go
            for r in old_by_pat.get(pid, []):
                if r.get("file") and os.path.exists(os.path.join(OUT, r["file"])):
                    os.remove(os.path.join(OUT, r["file"]))
        rows, prow = build_patient(raw, pid, audit, coords, aparc, args, prev, tmp_prefix=f"ersp/_tmp_{pid}_")
        # final file names: r<index in contacts.json> on a full build, <pid>_<j> on a partial one
        for j, r in enumerate(rows):
            if r["file"] and r["file"].startswith("ersp/_tmp_"):
                final = f"ersp/{pid}_{j}.bin" if partial else f"ersp/r{len(contacts) + j}.bin"
                os.replace(os.path.join(OUT, r["file"]), os.path.join(OUT, final))
                r["file"] = final
        contacts.extend(rows)
        patients.append(prow)

    # the per-trial reject tables, for every patient every time (cheap: the tsvs only)
    n_tr = 0
    for raw in cfg.patient_ids:
        pid = pid_of(str(raw))
        try:
            if write_trials(pid, audit) is not None:
                n_tr += 1
        except Exception as e:                          # a table the run could not read either
            print(f"   [trials] {pid}: {type(e).__name__}: {e}")
    print(f"[trials] {n_tr} patients -> review/trials/<pid>.json")

    manifest = {
        "built": datetime.now().strftime("%Y-%m-%d %H:%M"), "source_tree": CUBES.replace("\\", "/"),
        "n_patient": len(patients), "n_contact": len(contacts), "n_data": sum(1 for r in contacts if r["file"]),
        "conditions": CONDS, "n_cond": len(CONDS), "n_freq": NF, "f_hz": [0.0, (NF - 1) * F_STEP], "fmax_source_hz": (NF - 1) * F_STEP,
        "n_time": NT // T_DS, "time_downsample": T_DS, "n_time_source": NT,
        "dtype": "uint8", "order": ["cond", "freq", "time"], "vmin": -VLIM, "vmax": VLIM, "nan_byte": 0,
        "file": "each contact row's `file`",
        "hg_path": "{hg_root}/{patient}/LM/HG/{cond}/{patient}_{cond}_{hg_reref}_HGtrials_{name}.png",
        "psd_path": "{hg_root}/{patient}/LM/PSD_clean/{cond}/PSD/psd_by_shaft.png (per-shaft patients) or psd_allch_full.png",
        "iqr_path": "{hg_root}/{patient}/LM/Report/{patient}_{cond}_iqr_postDur_QC.png",
        "trials": "trials/{patient}.json: every trial of the run's tables with the reason it was dropped (lf_trials.collect_trials, the run's settings)",
        "metrics": {"power": "mean |dB| over the cube", "hg": "mean dB, 70-150 Hz, stimulus half",
                    "stripe": "mean over 50..350 Hz rows of |row - mean of rows +-2|, dB", "rel": "split-half Pearson r (ERSP_halves)"},
    }
    io.open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8").write(json.dumps(manifest, indent=1))
    io.open(os.path.join(OUT, "patients.json"), "w", encoding="utf-8").write(json.dumps(patients, indent=0))
    io.open(os.path.join(OUT, "contacts.json"), "w", encoding="utf-8").write(json.dumps(contacts, separators=(",", ":")))
    referenced = {r["file"] for r in contacts if r.get("file")}
    stray = [f for f in os.listdir(os.path.join(OUT, "ersp")) if "ersp/" + f not in referenced]
    for f in stray:                                    # a file no row names any more
        os.remove(os.path.join(OUT, "ersp", f))
    size = sum(os.path.getsize(os.path.join(OUT, f)) for f in referenced) / 1e6
    print(f"\n[done] {len(patients)} patients, {len(contacts)} contacts ({manifest['n_data']} with ERSP, {size:.0f} MB) -> {OUT}"
          + (f" · {len(stray)} stray files removed" if stray else ""))


if __name__ == "__main__":
    main()
