"""audit_freshness.py - is every published artefact built from the newest input?

    "C:/Users/fanda/AppData/Local/Programs/Python/Python311/python.exe" \
        02_FBM_Clustering/audit_freshness.py [--quick] [--no-badlist]

READS ONLY. Nothing is written, nothing is rebuilt. Walks the chain

    03_ERSP  ->  concat_source_v16  ->  21 clustering runs  ->  coverage_viz  -\
             \                                              -> centroids      -+-> main
              -> activity_viz (cube, review/ersp, review/qc) ------------------+-> activity-visualizer

and prints, per artefact, whether anything it was built FROM has moved since. Every row is
OK / STALE / WARN with the two timestamps that decide it, so a verdict can be checked by
hand. The last section is what is built but NOT PUSHED, which is the failure that shows a
correct local tree and a wrong live page.

WHY THIS EXISTS. On 2026-10-15 the LM visualizer was serving HG rasters from 2026-09-27
while every panel beside them was v16: make_lm_review_bundle.py had rebuilt contacts.json,
patients.json and review/ersp/ and left review/qc/ alone, and the bundle manifest stamped
the whole folder with the rebuild date. Nothing in the tree said so.

--quick skips the per-patient 03_ERSP walk (the slow part on the share) and takes the
source clock from the dataset cache instead; --no-badlist skips the bad-list product check.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ERSP = os.path.join(R, "01_FBM_Analysis", "outputs", "03_ERSP")
DS = os.path.join(R, "02_FBM_Clustering", "outputs", "_dataset")
CLUST = os.path.join(R, "02_FBM_Clustering", "outputs", "clustering")
FS = os.path.join(R, "02_FBM_Clustering", "outputs", "250_recon", "fsaverage")
COV = os.path.join(FS, "coverage_viz")
CEN = os.path.join(CLUST, "paper_web", "centroids")
AV = os.path.join(FS, "activity_viz")
RV = os.path.join(AV, "review")
AV_WT = r"C:\Users\fanda\av_branch"
SITE = r"C:\Users\fanda\lorafanda.github.io"

# the feature sets the notebooks fit as of 2026-10-13 (240/241/242 FEATURE_SETS)
WANT_FSETS = ["concat_bands15", "concat_bands5", "concat_bands2", "concat_hfa"]

ROWS: list[tuple] = []
ts = lambda t: dt.datetime.fromtimestamp(t).strftime("%m-%d %H:%M") if t else "-"


def say(verdict, what, detail=""):
    ROWS.append((verdict, what, detail))
    mark = {"OK": "  ok  ", "STALE": " STALE", "WARN": " warn ", "INFO": "      "}[verdict]
    print(f"{mark}  {what:<46} {detail}")


def age(what, built, newest_src, src_name, grace_s=0):
    """One freshness row: STALE when an input is newer than the artefact."""
    if not built:
        return say("STALE", what, f"MISSING (newest {src_name} {ts(newest_src)})")
    if newest_src and newest_src > built + grace_s:
        return say("STALE", what, f"built {ts(built)} < {src_name} {ts(newest_src)}")
    say("OK", what, f"built {ts(built)}  (>= {src_name} {ts(newest_src)})")


def newest(pattern, recursive=True):
    f = [x for x in glob.glob(pattern, recursive=recursive) if os.path.isfile(x)]
    return (max(os.path.getmtime(x) for x in f), len(f)) if f else (0, 0)


def head(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def usable(cwd):
    """GIT NEEDS A REAL LOCAL DIRECTORY AS cwd. The server reaches this repo as
    \\\\nasac-m2.unige.ch\\... and CreateProcess refuses a UNC cwd with WinError 267; the
    push worktrees exist on the laptop only, and the server has no remote. Every git-backed
    row is laptop-only and says so rather than raising."""
    return bool(cwd) and os.path.isdir(cwd) and not str(cwd).replace("/", "\\").startswith("\\\\")


def git(args, cwd):
    if not usable(cwd):
        return None
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
    except OSError:
        return None
    # NOT .strip(): porcelain lines start with a space for "modified, not staged", and
    # stripping the whole block eats the first line's, which then cuts a character off
    # that file's name.
    return r.stdout.rstrip("\n")


# ---------------------------------------------------------------- 1. the source clock
def source_clock(quick):
    head("1.  SOURCE  -  01_FBM_Analysis/outputs/03_ERSP")
    pats = sorted(os.path.basename(p.rstrip("/\\"))
                  for p in glob.glob(os.path.join(ERSP, "*", "")))
    pats = [p for p in pats if not p.startswith("logs")]
    if quick:
        say("INFO", "per-patient walk", "skipped (--quick)")
        return {}, 0, 0
    cube_t, hfa_t, spread = 0, 0, []
    per = {}
    for p in pats:
        lm = os.path.join(ERSP, p, "LM")
        c = [x for x in glob.glob(os.path.join(lm, "ERSP_matrix", "**", "*"), recursive=True)
             if os.path.isfile(x)]
        h = [x for x in glob.glob(os.path.join(lm, "PerTrial", "HFA", "**", "*.png"),
                                  recursive=True) if os.path.isfile(x)]
        ct = max((os.path.getmtime(x) for x in c), default=0)
        ht = max((os.path.getmtime(x) for x in h), default=0)
        per[p] = dict(cube=ct, hfa=ht, n_cube=len(c), n_hfa=len(h))
        cube_t, hfa_t = max(cube_t, ct), max(hfa_t, ht)
        # 140 never clears a patient folder, so cubes from two different runs can sit
        # side by side and both enter the concat. A wide mtime spread is that smell.
        if c:
            lo = min(os.path.getmtime(x) for x in c)
            if ct - lo > 6 * 3600:
                spread.append((p, lo, ct, len(c)))
    say("INFO", f"{len(pats)} patients", f"newest cube {ts(cube_t)} \u00b7 newest HFA png {ts(hfa_t)}")
    if spread:
        for p, lo, hi, n in spread:
            say("WARN", f"{p}: cubes span two runs",
                f"{ts(lo)} .. {ts(hi)} over {n} files - orphans enter the concat")
    else:
        say("OK", "no patient has cubes from two runs", "mtime spread < 6 h everywhere")
    return per, cube_t, hfa_t


# ------------------------------------------------- 2. bad lists vs products on disk
def badlist_check(per):
    head("2.  BAD LISTS  -  are the products of bad-listed contacts gone?")
    sys.path.insert(0, os.path.join(R, "01_FBM_Analysis", "functions"))
    try:
        import config as cfg
    except Exception as e:                                        # noqa: BLE001
        return say("WARN", "config.py", f"could not import: {e}")
    bad = getattr(cfg, "bad_channels_manual", {})
    say("INFO", "bad_channels_manual",
        f"{len(bad)} patients, {sum(len(v) for v in bad.values())} channels")
    left = []
    for p, chans in sorted(bad.items()):
        lm = os.path.join(ERSP, p, "LM")
        if not os.path.isdir(lm):
            continue
        names = {os.path.basename(x) for x in
                 glob.glob(os.path.join(lm, "**", "*"), recursive=True)}
        for ch in chans:
            # the contact name sits between underscores in every product file name
            hits = [n for n in names if f"_{ch}." in n or f"_{ch}_" in n or n.startswith(ch + ".")]
            if hits:
                left.append((p, ch, len(hits), sorted(hits)[0]))
    if left:
        for p, ch, n, ex in left[:25]:
            say("STALE", f"{p} {ch}", f"{n} product file(s) still on disk, e.g. {ex}")
        if len(left) > 25:
            say("STALE", f"... and {len(left) - 25} more", "")
    else:
        say("OK", "every bad-listed contact", "no product files left in 03_ERSP")


# ------------------------------------------------------------- 3. the dataset cache
def cache_check(cube_t):
    head("3.  DATASET CACHE  -  02_FBM_Clustering/outputs/_dataset")
    vers = sorted(glob.glob(os.path.join(DS, "concat_source_v*")))
    say("INFO", "caches on disk", ", ".join(os.path.basename(v) for v in vers))
    cur = vers[-1] if vers else None
    if not cur:
        return say("STALE", "newest cache", "MISSING"), None
    pj = os.path.join(cur, "params.json")
    age(f"{os.path.basename(cur)}", os.path.getmtime(pj), cube_t, "03_ERSP cube")
    try:
        import pandas as pd
        m = pd.read_parquet(os.path.join(cur, "df_meta.parquet"))
        say("INFO", "cache contents",
            f"{len(m)} rows \u00b7 {m['patient_id'].nunique()} patients")
    except Exception as e:                                        # noqa: BLE001
        say("WARN", "df_meta.parquet", str(e)[:60])
    return os.path.basename(cur), os.path.getmtime(pj)


# ------------------------------------------------------------------- 4. the fits
def runs_check(cache_name, cache_t):
    head("4.  CLUSTERING RUNS  -  were they all fit on that cache?")
    runs, newest_run = [], 0
    for mf in sorted(glob.glob(os.path.join(CLUST, "*", "concat_*", "runs", "*",
                                            "manifest.json"))):
        try:
            j = json.load(open(mf, encoding="utf-8"))
        except Exception:                                         # noqa: BLE001
            continue
        d = os.path.dirname(mf)
        n = sum(1 for _ in open(os.path.join(d, "labels.csv"), encoding="utf-8")) - 1 \
            if os.path.exists(os.path.join(d, "labels.csv")) else -1
        runs.append(dict(method=j.get("method"), fset=j.get("feature_set"),
                         run=j.get("run_id"), cache=(j.get("cache") or {}).get("name"),
                         n=n, t=os.path.getmtime(mf), dir=d))
        newest_run = max(newest_run, os.path.getmtime(mf))
    say("INFO", "runs with a manifest", f"{len(runs)} \u00b7 newest {ts(newest_run)}")
    wrong = [r for r in runs if r["cache"] != cache_name]
    old_n = sorted({r["n"] for r in runs if r["cache"] == cache_name})
    if wrong:
        byc = {}
        for r in wrong:
            byc.setdefault(r["cache"], []).append(r)
        for c, rs in sorted(byc.items(), key=lambda kv: str(kv[0])):
            say("WARN", f"{len(rs)} run(s) fit on {c or 'no cache recorded'}",
                "earlier cohort - fine as history, not for the paper"
                if c else "manifest has no cache block - pre-provenance run")
    say("OK" if len(old_n) == 1 else "STALE", f"rows per run on {cache_name}",
        f"n = {old_n}" + ("" if len(old_n) == 1 else "  <- the fits disagree"))
    cur = [r for r in runs if r["cache"] == cache_name]
    have = sorted({r["fset"] for r in cur})
    miss = [f for f in WANT_FSETS if f not in have]
    say("OK" if not miss else "STALE", "feature sets fit on this cache",
        ", ".join(have) + (f"   MISSING {miss}" if miss else ""))
    for f in sorted(have):
        s = [r for r in cur if r["fset"] == f]
        ms = sorted({r["method"] for r in s})
        say("OK" if len(ms) == 3 else "WARN", f"   {f}",
            f"{len(s)} run(s): {', '.join(ms)} \u00b7 newest {ts(max(r['t'] for r in s))}")
    # the per-run statistics the paper quotes
    name = "cluster_statistics.json"
    f = [os.path.join(r["dir"], name) for r in cur
         if os.path.exists(os.path.join(r["dir"], name))]
    say("OK" if len(f) == len(cur) else "WARN", f"   {name}",
        f"{len(f)} of {len(cur)} runs \u00b7 newest {ts(max(os.path.getmtime(x) for x in f))}"
        if f else f"not in any of the {len(cur)} current runs")
    # THE LOPO PREDICTION TABLE is not per run: make_cluster_statistics.py writes it to
    # clustering/statistics/<feature set>_K<k>/. The `seed` column is the tell that it was
    # written AFTER fit_with_predictor (2026-10-14) - before that, predict_any refit cNMF
    # with a different seed and the cNMF numbers were two unrelated solutions compared to
    # each other, scoring below chance. A table without the column is that broken version.
    import csv as _csv
    found = sorted(glob.glob(os.path.join(CLUST, "statistics", "*", "stats_lopo_predict.csv")))
    if not found:
        say("STALE", "   stats_lopo_predict.csv", "none written yet")
    for p in found:
        rows = list(_csv.DictReader(open(p, encoding="utf-8")))
        fixed = bool(rows) and "seed" in rows[0]
        tag = os.path.basename(os.path.dirname(p))
        say("OK" if fixed else "STALE", f"   LOPO {tag}",
            f"{ts(os.path.getmtime(p))} \u00b7 {len(rows)} rows \u00b7 "
            + ("fixed predictor" if fixed else "NO seed column - the pre-10-14 broken predictor"))
    return newest_run, cur


# ------------------------------------------------------- 5. the published bundles
# ------------------------------------------------------------------------ 5. the recon
def recon_check(cache_name):
    """Where every coordinate, neighbour radius and anatomical label comes from.

    Nothing audited this until 2026-10-16, and that is exactly how the 10-12 regeneration
    went unnoticed for four days: 36 contacts had moved up to 70.6 mm, 12 had been renamed
    (EL051 pSTG_R* -> pSTG_*), and the outputs were never committed. None of them happened
    to be cohort electrodes, so nothing published moved - but that was luck, not a check.
    """
    head("5.  RECON  -  the coordinates the bundles and every brain plot are built on")
    coords = os.path.join(FS, "coords")
    status = os.path.join(os.path.dirname(FS), "_step1_tkrRAS_status.tsv")
    allcsv = os.path.join(coords, "ALL_PATIENTS_contacts_fsaverage.csv")

    # did step 1 run, and did every patient come back OK?
    if os.path.exists(status):
        with open(status, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        bad = [r.get("pid") for r in rows if (r.get("status") or "").upper() != "OK"]
        say("OK" if rows and not bad else "STALE", "   step 1 tkrRAS",
            f"{len(rows)} patients, run {ts(os.path.getmtime(status))}"
            + (f"   NOT OK: {', '.join(x for x in bad[:6] if x)}" if bad else ", all OK"))
    else:
        say("STALE", "   step 1 tkrRAS", "MISSING _step1_tkrRAS_status.tsv")

    # the fsaverage table every downstream artefact reads
    t_all = os.path.getmtime(allcsv) if os.path.exists(allcsv) else 0
    if t_all:
        with open(allcsv, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        pats = {r.get("patient") for r in rows}
        say("INFO", "   ALL_PATIENTS_contacts_fsaverage",
            f"{len(rows)} contacts, {len(pats)} patients, written {ts(t_all)}")
    else:
        say("STALE", "   ALL_PATIENTS_contacts_fsaverage", "MISSING")

    # THE ROW THAT WAS MISSING. A regenerated recon that is never committed leaves git HEAD
    # describing coordinates nobody is using any more.
    rel = os.path.relpath(coords, R).replace("\\", "/")
    out = git(["status", "--porcelain", "--", rel], R)
    if out is None:
        say("INFO", "   coords committed?", "git cannot run here (UNC repo) - laptop only")
    else:
        dirty = [l[3:] for l in out.splitlines() if l.strip()]
        say("OK" if not dirty else "STALE", "   coords committed?",
            "committed" if not dirty else
            f"{len(dirty)} file(s) modified or untracked"
            f"   e.g. {os.path.basename(dirty[0])}")

    # contacts.json is DERIVED from those coordinates, so it may not be older than them
    contacts = os.path.join(COV, "contacts.json")
    age("   coverage contacts.json", os.path.getmtime(contacts) if os.path.exists(contacts) else 0,
        t_all, "coords", grace_s=120)

    # and does every electrode in the cohort actually have a reconstruction? That is
    # check_recon_name_match.py's table; read it rather than redo the matching here.
    nm = os.path.join(CLUST, "name_match", cache_name or "", "name_match_per_patient.csv")
    if os.path.exists(nm):
        with open(nm, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        n = lambda r, k: int(float(r.get(k) or 0))
        in_cohort = sum(n(r, "in_cohort") for r in rows)
        miss = [r for r in rows if n(r, "unmatched_in_cohort") > 0]
        age("   name match table", os.path.getmtime(nm), t_all, "coords", grace_s=120)
        say("OK" if not miss else "STALE", "   cohort electrodes with a recon",
            f"{in_cohort} in the cohort, all matched" if not miss else
            f"{sum(n(r, 'unmatched_in_cohort') for r in miss)} of {in_cohort} unmatched: "
            + ", ".join(r.get("patient", "?") for r in miss[:5]))
    else:
        say("WARN", "   name match table",
            f"not run for {cache_name} - check_recon_name_match.py writes it")


def bundles_check(cube_t, hfa_t, newest_run, cur_runs):
    head("6.  BUNDLES  -  what the two visualizer pages read")

    print("\n-- cluster_visualizer (main) --")
    mf = os.path.join(COV, "manifest.json")
    if os.path.exists(mf):
        age("coverage_viz/manifest.json", os.path.getmtime(mf), newest_run, "newest fit")
        j = json.load(open(mf, encoding="utf-8"))
        listed = {r.get("run") for r in j.get("runs", [])}
        # only the feature sets the page actually publishes can be "missing". TRACKS is
        # the list that decides what the bundle carries, so read it rather than assume
        # every run on the cache belongs: narrowing it to the current four is a decision,
        # not a gap, and the retired sets were reading as MISSING 9.
        live_fsets = set()
        try:
            import importlib.util as _ilu
            _s = _ilu.spec_from_file_location("_mcb", os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_coverage_bundle.py"))
            _m = _ilu.module_from_spec(_s)
            _s.loader.exec_module(_m)
            live_fsets = {t[0].split("/")[1] for t in _m.TRACKS}
        except Exception as e:
            print(f"   (could not read TRACKS: {e})")
        disk = {r["run"] for r in cur_runs
                if not live_fsets or r.get("fset") in live_fsets}
        retired = len(cur_runs) - len(disk)
        say("OK" if disk <= listed else "STALE", "   runs the manifest lists",
            f"{len(listed)} listed, {len(disk)} on this cache in a published feature set"
            + (f"   MISSING {len(disk - listed)}" if disk - listed else "")
            + (f"  ({retired} run(s) of retired sets ignored)" if retired else ""))
        say("INFO", "   cohorts", str({k: len(v) for k, v in j.get("cohorts", {}).items()}))
    else:
        say("STALE", "coverage_viz/manifest.json", "MISSING")
    t, n = newest(os.path.join(COV, "**", "*"))
    say("INFO", "coverage_viz tree", f"{n} files \u00b7 newest {ts(t)}")

    ix = os.path.join(CEN, "index.json")
    if os.path.exists(ix):
        age("centroids/index.json", os.path.getmtime(ix), newest_run, "newest fit")
        j = json.load(open(ix, encoding="utf-8"))
        have = sorted(j.get("feature_sets", {}))
        miss = [f for f in WANT_FSETS if f not in have]
        say("OK" if not miss else "STALE", "   feature sets in the bundle",
            ", ".join(have) + (f"   MISSING {miss}" if miss else ""))
        say("OK" if j.get("row_keys") else "WARN", "   row_keys (sample-panel labels)",
            "published" if j.get("row_keys") else "absent - samples show 'row N'")
        say("INFO", "   runs in the bundle", str(len(j.get("runs", []))))
    else:
        say("STALE", "centroids/index.json", "MISSING")

    print("\n-- LM_visualizer (activity-visualizer branch) --")
    for name, src, src_t in (("contacts.json", "03_ERSP cube", cube_t),
                             ("patients.json", "03_ERSP cube", cube_t),
                             ("manifest.json", "03_ERSP cube", cube_t)):
        f = os.path.join(RV, name)
        age(f"review/{name}", os.path.getmtime(f) if os.path.exists(f) else 0, src_t, src)
    t, n = newest(os.path.join(RV, "ersp", "*.bin"), recursive=False)
    age("review/ersp/ (per-contact ERSP)", t, cube_t, "03_ERSP cube")
    say("INFO", "   ", f"{n} .bin files")
    t, n = newest(os.path.join(RV, "qc", "**", "*.webp"))
    age("review/qc/ (HG rasters, PSD, IQR)", t, hfa_t, "03_ERSP HFA png")
    qm = os.path.join(RV, "qc", "manifest.json")
    if os.path.exists(qm):
        j = json.load(open(qm, encoding="utf-8"))
        say("INFO", "   qc/manifest.json",
            f"built {j.get('built')} \u00b7 n_hg {j.get('n_hg')} \u00b7 {n} webp on disk")
    t, n = newest(os.path.join(AV, "cube_f16.*.bin"), recursive=False)
    age("activity_viz cube_f16 parts", t, cube_t, "03_ERSP cube")
    say("INFO", "   ", f"{n} parts")


# --------------------------------------------------------------- 6. pushed or not
def pushed_check():
    head("7.  PUSHED?  -  a correct local tree and a wrong live page")

    # GIT NEEDS A REAL LOCAL DIRECTORY AS cwd. The server reaches this repo as
    # \\nasac-m2.unige.ch\... and CreateProcess refuses a UNC cwd with WinError 267; the
    # push worktrees exist on the laptop only, and the server has no remote. So this whole
    # section is laptop-only, and on the server it says so rather than raising.
    if not usable(R):
        say("INFO", "section 7 skipped", "the repo is a UNC path, so git cannot run here: "
                                         "this is the server. Sections 1-5 and 7 above are "
                                         "machine-independent; run the audit on the laptop "
                                         "for the push state (and push from there).")
        return

    # the LM bundle: the worktree IS the branch, so its qc stamp is what the page serves
    live = os.path.join(RV, "qc", "manifest.json")
    wt = os.path.join(AV_WT, "02_FBM_Clustering", "outputs", "250_recon", "fsaverage",
                      "activity_viz", "review", "qc", "manifest.json")
    if os.path.exists(live) and os.path.exists(wt):
        a = json.load(open(live, encoding="utf-8")).get("built")
        b = json.load(open(wt, encoding="utf-8")).get("built")
        say("OK" if a == b else "STALE", "activity-visualizer branch: review/qc",
            f"page serves {b}, local tree has {a}"
            + ("" if a == b else "   -> push_review_bundle.py"))
    elif os.path.exists(live):
        say("WARN", "activity-visualizer worktree", f"not found at {AV_WT} - laptop only")
    dirty = git(["status", "--porcelain", "--",
                 "02_FBM_Clustering/outputs/250_recon/fsaverage/activity_viz"], AV_WT)
    if dirty is not None:
        say("OK" if not dirty else "STALE", "   worktree vs its branch",
            "clean" if not dirty else f"{len(dirty.splitlines())} file(s) not committed")

    # the cluster bundle: committed to main of the analysis repo, served from raw.github.
    # push_cluster_bundle.py leaves out per-run files of runs the coverage manifest does
    # not list - retired runs stay on disk and are not the page's business - so a path
    # naming one of those is not an unpushed file and must not be counted as one.
    live = set()
    mf = os.path.join(COV, "manifest.json")
    if os.path.exists(mf):
        for r in json.load(open(mf, encoding="utf-8")).get("runs", []):
            if r.get("run"):
                live.add(str(r["run"]))
    for rel, label in ((os.path.relpath(COV, R).replace("\\", "/"), "coverage_viz"),
                       (os.path.relpath(CEN, R).replace("\\", "/"), "centroids")):
        out = git(["status", "--porcelain", "--", rel], R)
        if out is None:
            continue
        mine, retired = [], 0
        for l in out.splitlines():
            if not l.strip():
                continue
            p = l[3:]
            # a run stamp in the path (…__20261013_212310… or …/runs/<x>__<stamp>/)
            stamps = [s for s in re.findall(r"\d{8}_\d{6}", p)]
            if stamps and not any(s in live for s in stamps):
                retired += 1
            else:
                mine.append(p)
        say("OK" if not mine else "STALE", f"main: {label}",
            (f"committed" if not mine else
             f"{len(mine)} file(s) of a LISTED run not committed -> push_cluster_bundle.py")
            + (f"  ({retired} file(s) of retired runs ignored)" if retired else ""))
    ahead = git(["rev-list", "--count", "origin/main..main"], R)
    if ahead is not None:
        say("OK" if ahead in ("0", "") else "STALE", "main vs origin/main",
            "in sync" if ahead in ("0", "") else f"{ahead} commit(s) unpushed")
    if os.path.isdir(SITE):
        out = git(["status", "--porcelain"], SITE)
        if out is None:
            return
        n = len([l for l in out.splitlines() if l.strip() and not l.strip().startswith("??")])
        say("OK" if not n else "STALE", "site repo (pages)",
            "clean" if not n else f"{n} tracked file(s) modified: "
            + ", ".join(l[3:] for l in out.splitlines() if not l.startswith("??"))[:70])


# ------------------------------------------------------------------ 7. the figures
def figures_check(newest_run, cur_runs):
    head("8.  FIGURES")
    pf = os.path.join(CLUST, "paper_figures")
    if os.path.isdir(pf):
        png = glob.glob(os.path.join(pf, "*.png"))
        t, n = newest(os.path.join(pf, "*.png"), recursive=False)
        say("INFO", "paper_figures/", f"{n} png · newest {ts(t)}")
        # a figure's own file name carries the run it was built on, so the stale ones name
        # themselves: anything whose stamp is not among the current fits is from a dead run
        # Judge the NEWEST file per figure. Earlier versions sit beside it with their own
        # run stamps in their names; those are history, not a stale figure, and counting
        # them marked every figure stale.
        cur_stamps = {r["run"].replace("_", "-") for r in cur_runs} if cur_runs else set()
        byfig = {}
        for x in png:
            if " - Copy" in x:
                continue
            byfig.setdefault(os.path.basename(x).split("_")[0], []).append(x)
        for fig, xs in sorted(byfig.items()):
            x = max(xs, key=os.path.getmtime)
            b = os.path.basename(x)
            stamp = b.split("_run")[-1].split(".")[0] if "_run" in b else ""
            if not stamp:
                say("INFO", f"   {fig}", f"{ts(os.path.getmtime(x))} · no run in the name · {b[:46]}")
            else:
                say("OK" if stamp in cur_stamps else "STALE", f"   {fig}",
                    f"{ts(os.path.getmtime(x))} · run {stamp}"
                    + ("" if stamp in cur_stamps else " ← NOT a run of this cohort")
                    + f" · {b[:40]}")
    aud = os.path.join(R, "02_FBM_Clustering", "outputs", "site_figures_audit.csv")
    if os.path.exists(aud):
        import csv
        rows = list(csv.DictReader(open(aud, encoding="utf-8")))
        bad = [r for r in rows if "STALE" in (r.get("verdict") or "")]
        say("WARN" if bad else "OK", "site_figures_audit.csv",
            f"{len(bad)} of {len(rows)} figures marked STALE \u00b7 written {ts(os.path.getmtime(aud))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the per-patient 03_ERSP walk")
    ap.add_argument("--no-badlist", action="store_true")
    a = ap.parse_args()
    print(f"audit_freshness  {dt.datetime.now():%Y-%m-%d %H:%M}   repo {R}")
    per, cube_t, hfa_t = source_clock(a.quick)
    if not a.no_badlist and not a.quick:
        badlist_check(per)
    cache_name, cache_t = cache_check(cube_t)
    if a.quick:
        cube_t = hfa_t = cache_t          # the cache stands in for the source clock
    newest_run, cur = runs_check(cache_name, cache_t)
    recon_check(cache_name)
    bundles_check(cube_t, hfa_t, newest_run, cur)
    pushed_check()
    figures_check(newest_run, cur)
    head("SUMMARY")
    for v in ("STALE", "WARN"):
        rs = [r for r in ROWS if r[0] == v]
        print(f"{v}: {len(rs)}")
        for _, what, detail in rs:
            print(f"    {what:<46} {detail}")
    print(f"OK: {sum(1 for r in ROWS if r[0] == 'OK')}")
    return 1 if any(r[0] == "STALE" for r in ROWS) else 0


if __name__ == "__main__":
    sys.exit(main())
