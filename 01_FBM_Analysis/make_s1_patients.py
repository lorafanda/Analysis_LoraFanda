#!/usr/bin/env python3
"""The per-patient table of the "01 · Signal → ERSP" tab: for every patient of the 03_ERSP
tree, the reference it ran on, the contacts removed and why, what was found, what is owed.

    python make_s1_patients.py        print the table as text (a check; nothing is written)
    make_s1_tab.py calls patients_html()

WHAT IS READ, at build time, and from where:
  the bad list, WM exclusions, second combs, width floors, joined files      functions/config.py
  the reference of the patient's last run, its Unknown drops                 outputs/03_ERSP/wm_reref_report.tsv
  the time of the last run                                                   outputs/03_ERSP/logs/<pid>_<stamp>.log
  the contacts that have cubes now                                           outputs/03_ERSP/<pid>/LM/ERSP_matrix/audio
  the contacts removed from the tree without a rerun, and when               outputs/03_ERSP/logs/*_deleted_<stamp>.tsv (143)
  trials used                                                                outputs/03_ERSP/audit_140.tsv (141)
(The audit's "listed but not recorded" column is NOT used: it compares the bad list with the ERSP
figure names, and since the bad list is applied before the ERSP stage it names every listed contact.)

WHAT IS WRITTEN BY HAND, because it cannot be read: WHY a contact is on the bad list (WHY) and
what a check found (FOUND). WHY is tied to the config: a WHY entry that names a contact which
is not on that patient's bad list fails the build, and a bad-listed name no entry covers is
printed as "reason not recorded" - so the table cannot say more, or less, than the list does.

WHAT IS DERIVED: "owed". A patient whose last run used a reference contact that is on the bad
list now needs a rerun (its reference changes, so every cube does); a bad-listed contact that
still has cubes needs 143 --bad-listed --delete.
"""
from __future__ import annotations

import glob
import html
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "functions"))
import config as cfg                                                  # noqa: E402

TREE = ROOT / "outputs" / cfg.ERSP_TREE
_e = html.escape
CONDS = ("audio", "picture", "reading")


def R(prefix: str, a: int, b: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(a, b + 1)]


# ---------------------------------------------------------------------------------------
# WHY - (date, names, reason). Only what the config's own comments or a dated check record.
# ---------------------------------------------------------------------------------------
WHY = {
    "EL033": [
        ("2026-10-02", ["pH_R7", "aH_R12"] + R("aH_L", 1, 12),
         "visual review of the 03_ERSP tree (striped, “notched” ERSPs); removed from the tree without a rerun. Judged on cubes that carried PHG_R4's noise — see the finding"),
        ("2026-10-02", ["PHG_R4"],
         "white-matter reference contact: after the reference its 60–390 Hz floor sits 18 / 15 / 8 dB above the median data contact (audio / picture / reading) and it carries 98–102 % of the line every data contact shares, at 9.0× their amplitude"),
    ],
    "EL035": [
        ("2026-10-02", ["CinG_R1", "CinG_R3", "CinG_R5", "CinG_R7"] + R("aI_R", 1, 7),
         "the two shafts whose notch left a line more than 3 dB above the floor at 250 / 350 Hz (6 of 24 bands each, audit of the 09-30 run); CinG_R2 / 4 / 6 / 8 are reference contacts and stay. Removed from the tree without a rerun"),
    ],
    "EL037": [
        ("2026-09-28", ["pH_R12"], "70–150 Hz floor +33 / +31 / +7 dB above the patient's median: dead in two of three blocks"),
        ("2026-09-28", ["aI_L2", "aI_L3", "aI_L4", "aI_L5", "A_L4", "pI_L14", "CinG_L2"], "Lora, after the rerun; their files removed by hand"),
    ],
    "EL038": [
        ("2026-10-02", ["pH_R1", "pH_R2", "LinG_R5", "LinG_R6", "A_L1"], "review of the 03_ERSP tree; removed from the tree without a rerun"),
    ],
    "EL040": [
        ("2026-10-02", ["aH_R13", "pPVH_L2"], "HFA / ERSP review of the 03_ERSP tree; removed from the tree without a rerun"),
        ("", R("FP-R", 10, 15), "the FP-R shank sits on the parcellation boundary (the Unknown rule kept FP-R12 / 15 and dropped FP-R11 on label order): the whole run is excluded"),
        ("", R("PlaT_L", 1, 3), "placeholder entry (“example” in the config)"),
    ],
    "EL043": [
        ("2026-09-28", R("sSMG", 8, 10), "12–70 Hz floor 10–14 dB above the patient's median in every condition, the excluded trials saturated in the HFA raster"),
    ],
    "EL045": [
        ("2026-09-28", ["PlaT_L2", "PlaT_L3", "PlaT_L4", "PlanTL4", "A_R8"],
         "70–150 Hz floor 7–15 dB above the patient's median in every condition; PlaT_L4 was one of the five reference contacts (PlanTL4 is the same contact in the electrodes table's spelling)"),
        ("2026-09-28", ["A_L2", "A_L3", "A_L4", "STG_L1", "STG_L2"], "the deepest 350 Hz stripe of the patient (−1.7 dB rows)"),
        ("2026-10-02", ["TTG_L4", "TTG_L7", "TTG_L8", "A_L1", "A_L5", "A_L6"] + R("EntG_L", 5, 11), "visual review of the 03_ERSP tree; removed from the tree without a rerun"),
    ],
    "EL046": [
        ("2026-09-24", ["pI_L1", "pI_L4", "aH_R12", "A_L1", "pH_L1", "A_L2"],
         "the out-of-brain list (<code>LOOKUP_OUT_OF_BRAIN</code>): pI_L1 / pI_L4 from the anatomy review, the other four added after the 09-24 rerun"),
    ],
    "EL048": [
        ("2026-10-02", ["aH_L1"],
         "white-matter reference contact: 3230 µV SD in the raw file against 60–75 for the others; after the reference every line from 50 to 400 Hz stood at the same amplitude and phase on the 69 other contacts and at 10.0×, sign flipped, on aH_L1"),
        ("2026-10-02", ["pH_R13"], "dead contact: 2362 µV SD in the raw file against 65–73 on its shaft neighbours (on the list since September, measured 10-02)"),
        ("", ["A_R9", "EntG_R12", "PHG_R15", "aH_L8", "aH_L9", "pH_R14", "pH_R15"], "outside the brain (<code>LOOKUP_OUT_OF_BRAIN</code>, isOut in the anatomy Lookup)"),
        ("", ["EKG", "EKG-"], "auxiliary leads"),
    ],
    "EL051": [
        ("2026-09-17", R("pH_R", 7, 13) + R("pSTG_", 3, 9) + R("VIM_", 12, 18) + ["pSPL_3"] + R("aSMG_", 4, 8), "Lora's list, names checked against the recording 09-18 (pSTG = the whole shaft as recorded)"),
        ("2026-09-29", R("VIM_", 1, 11) + ["aH_R13"], "VIM extended to the whole shaft; aH_R13"),
        ("", ["EKG+", "EKG-"], "auxiliary leads"),
    ],
    "EL052": [
        ("2026-09-17", R("A_L", 9, 13) + R("A_R", 9, 13) + R("aH_L", 7, 13) + ["aH_R1", "ANT_R15", "EntG_R11", "Pul_R11", "PrCG_R6", "PrCG_R7"], "Lora's list"),
        ("2026-09-29", ["ANT_R14", "aH_R12", "aH_R13", "pH_L12", "pH_L13"], "Lora"),
        ("2026-09-29", ["Pul_R16"], "white-matter candidate: 70–150 Hz floor 14 dB above the recording's median, a broadband smear from 100 to 350 Hz in every condition"),
        ("", ["Chin-", "EKG-", "EMG-"], "the minus poles of the chin EMG, EKG and EMG pairs (auxiliary leads the EL name filter keeps)"),
    ],
    "PAT_1327": [
        ("2026-09-24", ["CAG5", "CPG1", "IAG10", "PHG2", "PHG3"], "visual review of the first run; none is a reference contact"),
        ("2026-09-29", ["FOD7"], "Lora; its files removed by hand"),
    ],
    "PAT_3390": [("2026-10-02", ["PHG12"], "review of the 03_ERSP tree; removed from the tree without a rerun")],
    "PAT_3415": [
        ("2026-09-30", R("HLG", 1, 18), "the whole shaft sits 25–28 dB above the other depth contacts at 70–150 Hz (a connector, not tissue); HLG18 had been a ninth of the reference until 09-29"),
    ],
    "PAT_3455": [("2026-10-02", ["HAD9", "HAD10"], "review of the 03_ERSP tree; removed from the tree without a rerun")],
    "PAT_3965": [
        ("2026-10-02", ["cmd12"],
         "white-matter reference contact, cmd11's twin: 356 µV SD in the raw file against a median of 73, 60–390 Hz floor 22.6 dB above the median channel; after the reference it carried 99–101 % of the line every data contact shares, at 43× their amplitude"),
        ("2026-10-02", ["cmd11", "y1", "y2"], "measured in the raw file 10-02: 60–390 Hz floor 22–23 dB above the median channel (on the list before that)"),
    ],
    "PAT_3975": [
        ("2026-10-02", R("PHD", 1, 6) + R("HAG", 1, 3) + R("HAD", 1, 4) + ["FOD1"], "ictal activity (Lora, review of the 03_ERSP tree); none is a reference contact; removed from the tree without a rerun"),
    ],
    "PAT_5515": [("2026-09-15", ["FOD6", "FOD8"], "44× / 28× the median high gamma after the notch")],
    "PAT_6704": [
        ("", R("HADm", 1, 12) + R("PHDm", 1, 12) + R("TPDm", 1, 12) + R("FODm", 1, 12), "the microwires of the export, 12 per bundle (not macro contacts)"),
        ("", ["ainp1"], "analog input"),
    ],
    "PAT_6854": [
        ("", R("HADm", 1, 8) + R("ADm", 1, 8) + R("HAGm", 1, 8) + R("AGm", 1, 8), "the microwires of the export, 8 per bundle (not macro contacts)"),
    ],
    "EL030": [("", ["EntG_R18"], "placeholder entry (“example” in the config)")],
    "PAT_3780": [("", ["FAP9"], "placeholder entry (“example” in the config)")],
}

# ---------------------------------------------------------------------------------------
# FOUND - (date, text): what a check measured. Scratch simulations are named as such.
# ---------------------------------------------------------------------------------------
FOUND = {
    "EL048": [
        ("2026-10-02",
         "<b>one reference contact was behind the whole notch problem.</b> aH_L1 (white matter, one of 11 reference contacts) is broken in the recording; "
         "through the reference mean one eleventh of it went into every contact: all lines 50–400 Hz (28–57 dB above the floor before the notch), their trial-locked "
         "modulation (the 200 / 300 Hz ERSP bands, −0.6 / −0.8 dB below their neighbours over the whole trial in audio), the band at the top of the axis "
         "(a 400 Hz line the notch never tests for a patient with a second comb), the 84.5 / 115.5 Hz pair, the 12.5 Hz comb of the audio block — and broadband noise "
         "worth 2–4 dB of every contact's floor, the 70–150 Hz band included. "
         "Rerun 20:53 on the 10-contact reference: lines 6–26 dB before the notch, every band notched at ±1 Hz and back on the floor, "
         "200 / 300 Hz rows at −0.09 / +0.11 dB (audio), no row above 30 Hz more than 0.3 dB off its neighbours in any condition (20 / 5 / 19 rows before)"),
    ],
    "PAT_3965": [
        ("2026-10-02",
         "<b>the same fault, milder.</b> cmd12 (one of 44 reference contacts) carried all of the patient's line noise and 1.1–1.9 dB of every contact's floor. "
         "Scratch simulation without it: lines before the notch 17–26 dB → 1–3 dB, 130 notched bands → 49–52, floor −1.1 / −1.7 / −1.4 dB (70–150 / 150–250 / 250–390 Hz, audio block; the other two alike); "
         "the ERSP rows were clean before and stay clean. Bad-listed; <b>not rerun yet</b>"),
        ("2026-09-30", "trial z tree: audio trial #41 over z = 4 on HAG / AG / CAG only — a discharge, not a trial problem"),
    ],
    "EL033": [
        ("2026-10-02",
         "<b>the same fault, worst in the audio block.</b> PHG_R4 (one of 10 reference contacts) is noisy: scratch simulation without it — lines before the notch 17–37 dB → 1–12 dB; "
         "floor of the audio block −7.0 / −9.8 / −9.6 dB (70–150 / 150–250 / 250–390 Hz), picture −0.5 / −2.3 / −1.4, reading −0.2 / −0.3 / −0.4; "
         "26 of 56 notched bands were more than 3 dB above the floor in audio, none after. Bad-listed; <b>not rerun yet</b>"),
        ("2026-10-02",
         "the 13 contacts removed the same day, recomputed with PHG_R4 out (scratch): the horizontal notch stripes go from all of them. aH_L1 and aH_L2 stay abnormal on their own "
         "(70–150 Hz floor +6 to +12 dB over the other contacts, 1–30 Hz +12 to +20 dB, kurtosis 23–37); aH_L3 / aH_L4 are spiky (kurtosis 8–21); aH_L9–11 carry a floor that rises "
         "towards the outermost contact (+2 → +9 dB); pH_R7, aH_R12 and aH_L5–8 are within 2 dB of the other contacts. <b>Which of them come back is open</b>"),
        ("2026-09-30", "trial z tree: picture trial #46 over z = 4 on 67 of 75 channels (montage-wide)"),
    ],
    "EL043": [
        ("2026-10-02",
         "reference contact pIns2 (one of 28) carries 80–93 % of the 50 / 100 / 200 / 300 / 400 Hz line the contacts share (raw 50 Hz line +50 dB against a median of +22; its broadband is ordinary). "
         "Scratch simulation without it: lines before the notch 6–12 dB lower, floor −0.03 dB, the same notch result and the same ERSP rows — <b>left in the reference</b>"),
    ],
    "EL040": [
        ("2026-10-02",
         "reference contact CinG_R13 sits 10 dB above the median data contact at 12–30 Hz (5 dB at 30–70 Hz, 1 dB at 70–150 Hz; PSD tables of the 09-30 run). "
         "With 9 reference contacts that is ~12 % of a contact's beta power, on every contact. <b>Not checked further</b>"),
    ],
    "EL045": [
        ("2026-09-28", "PlaT_L4, 11 dB above the patient's floor, was a fifth of the reference; once it left, a 59.94 Hz comb (the screen's refresh) stood out on four shafts — notched as a second comb; "
                       "the 350 Hz row still sat 0.74 dB below its neighbours → ±5 Hz width floor"),
    ],
    "EL037": [
        ("2026-09-28", "350 Hz row 1.15 dB below its neighbours on every trial while the PSD test passed at ±1 Hz → ±5 Hz width floor (row bias +0.09 after)"),
    ],
    "EL038": [
        ("2026-09-24", "trigger tables rebuilt from the full session log: 14 audio + 2 reading trials that a DC6 drift and a pruned log copy had lost"),
    ],
    "EL051": [
        ("2026-09-29", "white-matter candidates checked one against the other five: reference = A_R6, A_R9, aH_R11; POp_R1 carries a +0.5 dB speech response and is analysed as data"),
    ],
    "EL052": [
        ("2026-09-29", "on its FreeSurfer reconstruction since 09-29 (whole-recording CAR before): reference = aI_R17, aI_R18, mI_R6, mI_R7, mI_R15, each checked against the others"),
    ],
    "PAT_3415": [
        ("2026-09-30", "depth contacts only: its 64 grid and 30 strip contacts leave with the Unknown drop; CAR and WM compared on the depth contacts, alike, WM kept; "
                       "IMG8 (+1.2 dB high gamma in picture) and IPG15 are white-matter contacts analysed as data"),
    ],
    "PAT_6704": [
        ("2026-09-15", "THD1 (+2–3 dB high gamma through the audio stimulus), THD3, THD4 are labelled white matter but respond: kept out of the reference, analysed as data"),
    ],
}

NOT_CHECKED = "reason not recorded"


# ---------------------------------------------------------------------------------------
def norm(s) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def key(name, pid) -> str:
    """The electrodes-table key of a recording name (cfg.CHANNEL_SHAFT_ALIAS), as 140 matches it."""
    k = norm(name)
    for a, b in (getattr(cfg, "CHANNEL_SHAFT_ALIAS", {}).get(pid, {}) or {}).items():
        m = re.match(rf"^{re.escape(a)}(\d+)$", k)
        if m:
            return f"{b}{m.group(1)}"
    return k


def compress(names) -> str:
    """aH_L1 aH_L2 aH_L3 pH_R7 -> 'aH_L1–3, pH_R7' (order of first appearance)."""
    groups, order = {}, []
    for n in names:
        m = re.match(r"^(.*?)(\d+)$", str(n))
        p, i = (m.group(1), int(m.group(2))) if m else (str(n), None)
        if p not in groups:
            groups[p] = []
            order.append(p)
        groups[p].append(i)
    out = []
    for p in order:
        nums = sorted(i for i in groups[p] if i is not None)
        if not nums:
            out.append(p)
            continue
        runs, a, b = [], nums[0], nums[0]
        for i in nums[1:]:
            if i == b + 1:
                b = i
            else:
                runs.append((a, b)); a = b = i
        runs.append((a, b))
        out.append(p + " / ".join(f"{a}" if a == b else f"{a}–{b}" for a, b in runs))
    return ", ".join(out)


def _stamp(name: str) -> float | None:
    m = re.search(r"(\d{8})_(\d{6})", name)
    return time.mktime(time.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")) if m else None


def last_run(pid: str) -> float | None:
    ts = [_stamp(os.path.basename(p)) for p in glob.glob(str(TREE / "logs" / f"{pid}_2*.log"))]
    ts = [t for t in ts if t]
    return max(ts) if ts else None


def deletions() -> dict:
    """{pid: [(stamp, [contacts])]} from the 143 logs."""
    out = {}
    for p in sorted(glob.glob(str(TREE / "logs" / "*_deleted_*.tsv"))):
        st = _stamp(os.path.basename(p))
        try:
            d = pd.read_csv(p, sep="\t", dtype=str).fillna("")
        except Exception:
            continue
        if "patient" not in d or "channel" not in d:
            continue
        for pid, g in d.groupby("patient"):
            ch = sorted({c for c in g["channel"] if c and " " not in c})
            if ch:
                out.setdefault(pid, []).append((st, ch, "non-neural" if "non_neural" in os.path.basename(p) else "bad-listed"))
    return out


def cubes_now(pid: str) -> dict:
    out = {}
    for c in CONDS:
        out[c] = sorted(os.path.basename(p).split("_ERSP_")[1][:-7] for p in glob.glob(str(TREE / pid / "LM" / "ERSP_matrix" / c / "*_TN.npy")))
    return out


def rows() -> list[dict]:
    rep = pd.read_csv(TREE / "wm_reref_report.tsv", sep="\t", dtype=str).fillna("")
    rep = rep.drop_duplicates("patient_id", keep="last").set_index("patient_id")
    aud_p = TREE / "audit_140.tsv"
    aud = pd.read_csv(aud_p, sep="\t", dtype=str).fillna("").set_index("patient") if aud_p.exists() else None
    dels = deletions()
    micro = {v["pat_name"]: k for k, v in getattr(cfg, "MICROEPI_MAT_PRESETS", {}).items()}
    out = []
    for pid in sorted(p.name for p in TREE.iterdir() if (p / "LM").is_dir()):
        bad = list(cfg.bad_channels_manual.get(pid, []))
        why = WHY.get(pid, [])
        named = [n for _, names, _ in why for n in names]
        stray = [n for n in named if n not in bad]
        if stray:
            raise SystemExit(f"make_s1_patients: WHY[{pid}] names contacts that are not on its bad list: {stray}")
        uncovered = [n for n in bad if n not in named]
        run = last_run(pid)
        cubes = cubes_now(pid)
        n_c = [len(cubes[c]) for c in CONDS]
        r = rep.loc[pid] if pid in rep.index else None
        wm = [w for w in (r["wm_channels_used"].split("|") if r is not None else []) if w]
        bad_keys = {key(n, pid): n for n in bad}
        ref_now_bad = [bad_keys[w] for w in wm if w in bad_keys]
        still = sorted({n for c in CONDS for n in cubes[c]} & set(bad))
        after = [(st, ch, kind) for st, ch, kind in dels.get(pid, []) if run is None or st > run]
        removed_after = sorted({c for _, ch, kind in after if kind == "bad-listed" for c in ch})
        a = aud.loc[pid] if (aud is not None and pid in aud.index) else None
        special = []
        if pid in micro:
            special.append(f"MicroEPI {micro[pid]}: macro contacts only, white-matter contacts kept as data")
        if pid in getattr(cfg, "RAW_CONCAT", {}):
            special.append(f"{len(cfg.RAW_CONCAT[pid])} recordings joined")
        if pid in getattr(cfg, "STRIP_HEMI_PATIENTS", set()):
            special.append("contact names without the _L / _R infix")
        if pid in getattr(cfg, "WM_NOT_REFERENCE", {}):
            special.append("white-matter contacts analysed as data: " + compress(cfg.WM_NOT_REFERENCE[pid]))
        if pid in getattr(cfg, "notch_extra_bases", {}):
            special.append("second notch comb " + ", ".join(f"{b:g} Hz" for b in cfg.notch_extra_bases[pid]))
        for f0, hw in (getattr(cfg, "notch_interp_min_hw_hz", {}).get(pid, {}) or {}).items():
            special.append(f"notch width floor ±{hw:g} Hz at {f0:g} Hz")
        owed = []
        if ref_now_bad:
            owed.append(f"<b>rerun 140</b>: {compress(ref_now_bad)} was in the reference of the last run and is bad-listed now — the reference changes, so every cube does")
        if still:
            owed.append(f"<code>143 --bad-listed --delete</code>: {compress(still)} bad-listed, cubes still in the tree")
        out.append(dict(pid=pid, raw=micro.get(pid, ""), run=run, n_c=n_c, wm=wm, bad=bad, why=why, uncovered=uncovered,
                        unk=(r["n_channels_unknown_dropped"] if r is not None else ""),
                        trials=("/".join(str(a[f"{c}_trials_used"] or "–") for c in CONDS) if a is not None else ""),
                        removed_after=removed_after, removed_at=max((st for st, _, k in after if k == "bad-listed"), default=None),
                        special=special, found=FOUND.get(pid, []), owed=owed))
    return out


def _t(ts) -> str:
    return time.strftime("%m-%d %H:%M", time.localtime(ts)) if ts else "–"


def patients_html() -> str:
    R_ = rows()
    head = ("<tr><th>patient</th><th>last run · contacts · trials a / p / r</th><th>reference of that run</th>"
            "<th>removed contacts, and why</th><th>found · handling</th><th>owed</th></tr>")
    body = []
    for x in R_:
        n = x["n_c"]
        ncell = f"{n[0]}" if len(set(n)) == 1 else "/".join(map(str, n))
        wm = x["wm"]
        ref = f"<b>WM {len(wm)}</b>" + (": " + _e(" ".join(wm[:14])) + (f" … (+{len(wm) - 14})" if len(wm) > 14 else "") if wm else "")
        if x["unk"] and x["unk"] != "0":
            ref += f"<br>Unknown dropped: {_e(x['unk'])}"
        rem = []
        for date, names, why in x["why"]:
            rem.append(f"<b>{_e(compress(names))}</b> — {why}" + (f" <span class=\"runid\">{date}</span>" if date else ""))
        if x["uncovered"]:
            rem.append(f"<b>{_e(compress(x['uncovered']))}</b> — {NOT_CHECKED}")
        if x["removed_after"]:
            rem.append(f"<i>removed from the tree after the last run, no rerun ({_t(x['removed_at'])}): {_e(compress(x['removed_after']))}</i>")
        fnd = [f"{txt} <span class=\"runid\">{date}</span>" for date, txt in x["found"]] + [_e(s) for s in x["special"]]
        body.append(
            f"<tr style=\"vertical-align:top\"><td><b>{_e(x['pid'])}</b>{('<br>' + _e(x['raw'])) if x['raw'] else ''}</td>"
            f"<td>{_t(x['run'])}<br><b>{ncell}</b> contacts{('<br>' + _e(x['trials'])) if x['trials'] else ''}</td>"
            f"<td>{ref}</td>"
            f"<td>{'<br>'.join(rem) if rem else '–'}</td>"
            f"<td>{'<br>'.join(fnd) if fnd else '–'}</td>"
            f"<td>{'<br>'.join(x['owed']) if x['owed'] else '–'}</td></tr>")
    return f'<table class="sum" style="margin:8px 0;font-size:12px">{head}{"".join(body)}</table>'


def main() -> int:
    for x in rows():
        print(f"\n{x['pid']} {x['raw']}  run {_t(x['run'])}  contacts {x['n_c']}  trials {x['trials']}  WM {len(x['wm'])}  unk {x['unk']}")
        for date, names, why in x["why"]:
            print(f"   - {compress(names)}: {re.sub('<[^>]+>', '', why)} [{date}]")
        if x["uncovered"]:
            print(f"   - {compress(x['uncovered'])}: {NOT_CHECKED}")
        if x["removed_after"]:
            print(f"   removed after the run ({_t(x['removed_at'])}): {compress(x['removed_after'])}")
        for s in x["special"]:
            print(f"   · {s}")
        for date, txt in x["found"]:
            print(f"   * [{date}] {re.sub('<[^>]+>', '', txt)[:150]}")
        for o in x["owed"]:
            print(f"   OWED {re.sub('<[^>]+>', '', o)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
