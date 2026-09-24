#!/usr/bin/env python3
"""
audit_site_figures.py - which figures on analysis_status.html are out of date, and what
regenerates them.

Every <img data-fig="..."> on the status page points at a file in this repo, which the
page loads from raw.githubusercontent of main. Next to each one the page carries a
<div class="runid"> naming the script that made it - the site's own convention - so the
audit can say not just "this is old" but "run this".

A figure is judged against what it DEPENDS on, not against the calendar:

  cohort   drawn from the concat cache or a clustering run
           -> stale if older than the newest concat_source_v<N>
  ersp     drawn from the 140 cubes
           -> stale if older than the newest cube in 04_ersp_LM_RAWONLY
  method   a documentation drawing of the pipeline
           -> never "stale" by date; stale only when the method itself changed, which a
              human has to decide (the 0-500 -> 0-400 Hz axis is the current example)

    python audit_site_figures.py                 print the summary
    python audit_site_figures.py --csv           also write outputs/site_figures_audit.csv

Re-run it after every rerun: the point is to shrink the STALE list to nothing before a
cohort is published.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import urllib.parse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SITE = Path.home() / "lorafanda.github.io" / "analysis_status.html"
OUT = ROOT / "02_FBM_Clustering" / "outputs"

DEP = [(r"outputs/clustering/", "cohort"), (r"outputs/pooling/", "cohort"),
       (r"outputs/classification/", "cohort"), (r"outputs/figures/", "cohort"),
       (r"_status_png/s1_", "ersp"), (r"microepi_status/", "ersp"),
       (r"outputs/compare_140/", "ersp"), (r"preprocessing_docs/", "method")]


def newest(pattern: Path, glob: str) -> float:
    ts = [p.stat().st_mtime for p in pattern.glob(glob)]
    return max(ts) if ts else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true")
    a = ap.parse_args()

    cache = sorted((ROOT / "02_FBM_Clustering" / "outputs" / "_dataset").glob("concat_source_v*"),
                   key=lambda p: int(p.name.rsplit("v", 1)[1]) if p.name.rsplit("v", 1)[1].isdigit() else -1)
    V = cache[-1].joinpath("params.json").stat().st_mtime
    cubes = ROOT / "01_FBM_Analysis" / "outputs" / "04_ersp_LM_RAWONLY"
    E = max((max((f.stat().st_mtime for f in (p / "LM" / "ERSP_matrix").rglob("*.npy")), default=0)
             for p in cubes.iterdir() if p.is_dir()), default=0)
    print(f"cohort  {cache[-1].name}  built {dt.datetime.fromtimestamp(V):%Y-%m-%d %H:%M}")
    print(f"cubes   newest in 04_ersp_LM_RAWONLY  {dt.datetime.fromtimestamp(E):%Y-%m-%d %H:%M}\n")

    s = SITE.read_text(encoding="utf-8", errors="replace")
    rows = []
    for m in re.finditer(r"<figure[^>]*>(.*?)</figure>", s, re.S):
        blk = m.group(1)
        figs = re.findall(r'data-fig="([^"]+)"', blk)
        if not figs:
            continue
        txt = lambda p, d="": (re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", p.group(1))).strip() if p else d)
        num = txt(re.search(r'<div class="fignum">([^<]+)</div>', blk))
        title = txt(re.search(r"<h4[^>]*>(.*?)</h4>", blk, re.S))[:70]
        how = txt(re.search(r'<div class="runid">(.*?)</div>', blk, re.S))[:80]
        sec = re.findall(r"<h[23][^>]*>(.*?)</h[23]>", s[:m.start()], re.S)
        for rel in figs:
            relp = urllib.parse.unquote(rel)
            f = ROOT / relp
            dep = next((d for rx, d in DEP if re.search(rx, relp)), "other")
            t = f.stat().st_mtime if f.exists() else float("nan")
            if not f.exists():
                v = "0 MISSING on disk"
            elif dep == "cohort":
                v = "3 current" if t >= V else "1 STALE - cohort"
            elif dep == "ersp":
                v = "3 current" if t >= E else "2 STALE - cubes"
            else:
                v = "4 method drawing"
            rows.append(dict(fig=num, title=title, section=txt(None) if not sec else
                             re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", sec[-1])).strip()[:44],
                             dep=dep, verdict=v, how=how, rel=relp,
                             date="" if t != t else dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d")))
    D = pd.DataFrame(rows).drop_duplicates(subset=["rel"])
    tracked = set(subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split("\n"))
    D["committed"] = D.rel.isin(tracked)
    D["folder"] = D.rel.str.replace(r"^[^/]+/outputs/", "", regex=True).str.split("/").str[:2].str.join("/")

    print(f"{len(D)} figures\n")
    print(D.verdict.value_counts().sort_index().to_string())
    print("\n--- stale, by output folder ---")
    st = D[D.verdict.str.startswith(("1", "2"))]
    print(st.groupby(["verdict", "folder"]).agg(n=("rel", "size"), oldest=("date", "min"),
                                                newest=("date", "max"))
            .sort_values(["verdict", "n"], ascending=[True, False]).to_string())
    bad = D[~D.committed]
    if len(bad):
        print(f"\n--- not committed, so blank on the live site ({len(bad)}) ---")
        print(bad[["fig", "rel"]].to_string(index=False))
    if a.csv:
        p = OUT / "site_figures_audit.csv"
        D.sort_values(["verdict", "folder", "fig"]).to_csv(p, index=False)
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
