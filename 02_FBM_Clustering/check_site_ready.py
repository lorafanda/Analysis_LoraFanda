#!/usr/bin/env python3
"""
check_site_ready.py - is everything the site shows actually built, and built from the
runs that are current?

    python check_site_ready.py
    python check_site_ready.py --k 8

WHY THIS EXISTS. The site is regenerated from files on disk. Nothing in that chain fails
loudly when a file is simply older than the run it claims to describe: the page renders,
the figure appears, and it is last month's fit. This walks the whole chain and says, for
every item, whether it is there and whether it came from the run that is newest now.

    OK       present, and drawn from the current run
    STALE    present, but from a superseded run or an older cohort
    MISSING  not built

It reads only. It never writes, never regenerates and never deletes; the last section
prints the commands that would fix what it found.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "functions"))
import lf_concat as CC          # noqa: E402
import lf_runs as LR            # noqa: E402

CLUST = ROOT / "outputs" / "clustering"
FIGS = CLUST / "paper_figures"
EXPL = CLUST / "explainers"
WEB = CLUST / "paper_web"
ATLAS = CLUST / "atlas"
FSETS = ["concat_hg", "concat_rawds", "concat_bands5", "concat_bands5z"]
TAG = dict(zip(FSETS, "abcd"))
METHODS = ["cnmf", "kmeans", "hierarchical"]
VARIANTS = ["", "_minP030", "_weighted"]      # the display-rule versions asked for

rows = []


def add(section, item, state, detail=""):
    rows.append((section, item, state, detail))


def rel(p: Path) -> str:
    """A short path for the report. The site repo is not under ROOT, so this falls back
    rather than raising, which is what crashed the check on the server."""
    for base in (ROOT, ROOT.parent, Path.home()):
        try:
            return str(p.relative_to(base))
        except ValueError:
            continue
    return str(p)


def newest(method, fset):
    try:
        return LR.newest_run(method, fset)
    except Exception:
        return None


def run_tag(rd):
    return "_run" + rd.name.replace("_", "-")


def find_fig(stem_prefix, run_tag_wanted, suffix=".png"):
    """The figure for this stem: exact-with-run-tag, else any older name."""
    want = FIGS / f"{stem_prefix}{run_tag_wanted}{suffix}"
    if want.exists():
        return want, "OK"
    others = sorted(FIGS.glob(f"{stem_prefix}_run*{suffix}")) + \
        ([FIGS / f"{stem_prefix}{suffix}"] if (FIGS / f"{stem_prefix}{suffix}").exists() else [])
    if others:
        return others[-1], "STALE"
    return None, "MISSING"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()
    K = a.k
    cache = CC.DEFAULT_CONCAT_CACHE
    cohort_tag = (re.search(r"_v(\d+)$", cache.name) or [None, "?"])[1]
    print(f"  cohort cache: {cache.name}")

    # ---- 1. the runs themselves ------------------------------------------------
    n_ref = None
    for m in METHODS:
        for fs in FSETS:
            rd = newest(m, fs)
            if rd is None:
                add("runs", f"{m}/{fs}", "MISSING", "no run at all")
                continue
            import numpy as np
            n = int(np.load(rd / "X_train.npy", mmap_mode="r").shape[0])
            n_ref = n_ref or n
            has_k = ((rd / "loadings_by_k" / f"G_k{K:02d}.npy").exists()
                     or f"k_{K}" in open(rd / "cluster_labels_by_k.csv",
                                         encoding="utf-8").readline())
            prm = (json.loads((rd / "manifest.json").read_text(encoding="utf-8"))
                   .get("params") or {})
            sp = prm.get("space", "unit-norm" if m == "cnmf" else "(not recorded)")
            bad = []
            if n != n_ref:
                bad.append(f"n={n} but {n_ref} elsewhere")
            if not has_k:
                bad.append(f"no K={K}")
            if m != "cnmf" and sp != "unit-norm":
                bad.append(f"space={sp}")
            add("runs", f"{m}/{fs}", "STALE" if bad else "OK",
                rd.name + ("  " + "; ".join(bad) if bad else f"  n={n}, {sp}"))

    # ---- 2. the figures --------------------------------------------------------
    for which, exists in (("FIG0_cohort", (FIGS / f"FIG0_cohort_v{cohort_tag}.png")),
                          ("FIG0_cohort_supplement",
                           (FIGS / f"FIG0_cohort_supplement_v{cohort_tag}.png"))):
        if exists.exists():
            add("FIG 0", which, "OK", exists.name)
        else:
            older = [f for f in sorted(FIGS.glob(f"{which}*.png"))
                     if f.stem == which or f.stem.startswith(which + "_v")]
            add("FIG 0", which, "STALE" if older else "MISSING",
                older[-1].name if older else f"want {exists.name}")

    for m in METHODS:
        for fs in FSETS:
            rd = newest(m, fs)
            if rd is None:
                continue
            for var in (VARIANTS if m == "cnmf" else [""]):
                stem = f"FIG1{TAG[fs]}_{fs}_{m}_K{K}{var}"
                f, st = find_fig(stem, run_tag(rd))
                add("FIG 1", stem, st, f.name if f else "")
                if f is not None and st == "OK":
                    for sc in ("_caption.txt", "_patients.csv", "_generalization.csv"):
                        g = f.with_name(f.stem + sc)
                        if not g.exists():
                            add("FIG 1", stem + sc, "MISSING", "sidecar")

    for fs in FSETS:
        rd = newest("cnmf", fs)
        if rd is None:
            continue
        for stem, sec in ((f"FIG2_agreement_{fs}_K{K}", "FIG 2"),
                          (f"FIG3_lana_{fs}_K{K}", "FIG 3")):
            f, st = find_fig(stem, run_tag(rd))
            add(sec, stem, st, f.name if f else "")
            if f is not None and st == "OK" and not f.with_name(f.stem + "_caption.txt").exists():
                add(sec, stem + "_caption.txt", "MISSING", "sidecar")

    rd_hg = newest("cnmf", "concat_hg")
    if rd_hg is not None:
        rt = run_tag(rd_hg)
        for stem in (f"FIG4_correspondence_K{K:02d}", f"FIG4_K{K:02d}",
                     f"FIG4_K{K:02d}", ):
            pass
        for stem, sfx in ((f"FIG4_correspondence_K{K:02d}", ""),
                          (f"FIG4_K{K:02d}", ""), (f"FIG4_K{K:02d}", "_weighted")):
            want = FIGS / f"{stem}{rt}{sfx}.png"
            add("FIG 4", stem + sfx, "OK" if want.exists() else "MISSING", want.name)
        for csv in ("matched", "summary"):
            want = FIGS / f"FIG4_{csv}_K{K:02d}{rt}.csv"
            add("FIG 4", f"FIG4_{csv}", "OK" if want.exists() else "MISSING", want.name)

    # ---- 3. what 252 owes FIG 3 ------------------------------------------------
    tabs = sorted(ATLAS.glob("lana_*/runs/*/recon/*with_fsaverage.csv"))
    runs = sorted(p for p in ATLAS.glob("lana_*/runs/*") if p.is_dir())
    add("252 recon", "LanA fsaverage tables",
        "OK" if tabs else "MISSING",
        f"{len(tabs)} table(s) under {len(runs)} LanA run(s)"
        + ("" if tabs else " - 252_clustering_recon.ipynb writes them"))

    # ---- 4. the site bundles ---------------------------------------------------
    # WHEN THE FIT HAPPENED, not when the folder was last touched: 249, a stability
    # sweep and 252 all write into a run directory long after the fit, which would make
    # anything built in between look stale.
    newest_run_time = max(((newest(m, fs) / "manifest.json").stat().st_mtime
                           for m in METHODS for fs in FSETS
                           if newest(m, fs) and (newest(m, fs) / "manifest.json").exists()),
                          default=0)
    SITE = Path.home() / "lorafanda.github.io"
    if not SITE.is_dir():
        print(f"  no site checkout at {SITE} - the site rows are reported n/a "
              "(this is the analysis machine, not the publishing one)")
    for label, p in (("centroid bundle", WEB / "centroids" / "index.json"),
                     ("coverage bundle",
                      ROOT / "outputs" / "250_recon" / "fsaverage" / "coverage_viz"
                      / "manifest.json"),
                     ("visualizer order", SITE / "cluster_visualizer_order.json"),
                     ("cluster visualizer", SITE / "cluster_visualizer.html"),
                     ("web block, KISS", EXPL / "webblock_kiss.html"),
                     # the figures block is spliced straight into the page, so the page
                     # itself is what has to be newer than the runs it shows
                     ("analysis_status.html", SITE / "analysis_status.html")):
        if not p.exists():
            # a site file cannot be missing on a machine with no site checkout
            add("site", label, "n/a" if not SITE.is_dir() else "MISSING", rel(p))
        else:
            st = "OK" if p.stat().st_mtime >= newest_run_time else "STALE"
            add("site", label, st,
                time.strftime("%m-%d %H:%M", time.localtime(p.stat().st_mtime)))

    # the explainers the KISS tab actually references, from its own ENTRIES - E.10 is
    # built by 01_FBM_Analysis and would never be found by globbing this folder
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("kiss", ROOT / "make_kiss_tab.py")
        kiss = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kiss)
        for e in kiss.ENTRIES:
            for num, _t, img, _alt in e.get("figs", []):
                f = (ROOT.parent / img) if "/" in img else (EXPL / img)
                add("explainers", f"{num}  {Path(img).name}",
                    "OK" if f.exists() else "MISSING", "" if f.exists() else str(f))
    except Exception as exc:
        add("explainers", "make_kiss_tab.ENTRIES", "MISSING", f"could not read: {exc}")

    # ---- report ----------------------------------------------------------------
    width = max(len(r[1]) for r in rows) + 2
    bad = 0
    order = list(dict.fromkeys(r[0] for r in rows))      # sections in first-seen order
    rows.sort(key=lambda r: order.index(r[0]))
    sec = None
    for section, item, state, detail in rows:
        if section != sec:
            print(f"\n  {section}")
            sec = section
        if state not in ("OK", "n/a"):
            bad += 1
        mark = {"OK": "  ok  ", "STALE": " STALE", "MISSING": " MISS ",
                "n/a": "  n/a "}[state]
        print(f"   {mark}  {item:<{width}}{detail}")

    n_na = sum(1 for r in rows if r[2] == "n/a")
    print(f"\n  {len(rows) - bad - n_na} of {len(rows) - n_na} checks pass"
          + (f", {n_na} not applicable here." if n_na else "."))
    if bad:
        print("\n  What is still owed, in order:")
        print("    252_clustering_recon.ipynb           (FIG 3 cannot run without it)")
        print("    the figure block from the last message")
        print("    make_centroid_bundle.py / make_coverage_bundle.py / "
              "make_cluster_visualizer.py")
        print("    make_paper_figures_webblock.py --insert   and   "
              "make_paper_tab.py --insert")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
