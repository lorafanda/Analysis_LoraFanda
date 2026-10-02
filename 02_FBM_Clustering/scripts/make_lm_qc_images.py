"""
make_lm_qc_images.py - compact WebP copies of the run's QC figures for the LM visualizer's
review mode, so the HG rasters, the trial-rejection histograms and the PSDs show on the
LIVE page (lorafanda.github.io) without a local server.

    "C:/Users/fanda/AppData/Local/Programs/Python/Python311/python.exe" \
        02_FBM_Clustering/scripts/make_lm_qc_images.py [--patients EL038 PAT_6704] [--workers 8]

Reads activity_viz/review/contacts.json (every contact with ERSP) and patients.json (the
reference tag in the HG file names) and writes, under activity_viz/review/qc/:
    <pid>/HG/<cond>/<name>.webp     the HFA trial raster (01_FBM_Analysis/outputs/03_ERSP/<pid>/LM/HFA/),
                                    800 px wide, quality 65: ~45 kB instead of ~350 kB
    <pid>/iqr_<cond>.webp           Report/<pid>_<cond>_iqr_postDur_QC.png, 900 px
    <pid>/psd_<cond>.webp           PSD_clean/<cond>/PSD/psd_by_shaft.png (per-shaft patients) else psd_allch_full.png
    manifest.json                   sizes, counts, build stamp
Incremental: a target newer than its source is kept. Targets whose contact is no longer in
contacts.json are removed. The originals stay where they are; a local "QC root" in the page
still shows them at full resolution. push_review_bundle.py pushes the qc/ folder with the rest.
"""
import argparse
import glob
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QC = os.path.join(REPO, "01_FBM_Analysis", "outputs", "03_ERSP")   # one tree since 2026-09-30
REVIEW = os.path.join(REPO, "02_FBM_Clustering", "outputs", "250_recon", "fsaverage", "activity_viz", "review")
OUT = os.path.join(REVIEW, "qc")
CONDS = ("audio", "picture", "reading")


def _pdir(lm, name):
    """A per-trial folder of one patient: <lm>/PerTrial/<name> (2026-10-02), or <lm>/<name> for a
    patient not moved yet and for the frozen 04_* tree. Same rule as config.product_dir."""
    new, old = os.path.join(lm, "PerTrial", name), os.path.join(lm, name)
    return new if (os.path.isdir(new) or not os.path.isdir(old)) else old


HG_W, HG_Q = 800, 65
FIG_W, FIG_Q = 900, 70
PSD_W = 1100


def convert(src, dst, width, quality):
    """PNG -> WebP at `width` px; returns (status, bytes). status: made / kept / missing."""
    if not os.path.exists(src):
        return "missing", 0
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return "kept", os.path.getsize(dst)
    im = Image.open(src).convert("RGB")
    im.thumbnail((width, 100000), Image.LANCZOS)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=quality, method=4)
    with open(dst, "wb") as f:            # write in one go, the share truncates on failed writes
        f.write(buf.getvalue())
    return "made", buf.tell()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", nargs="*", help="only these patient ids (PAT_/EL form)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    contacts = json.load(open(os.path.join(REVIEW, "contacts.json"), encoding="utf-8"))
    patients = {p["patient"]: p for p in json.load(open(os.path.join(REVIEW, "patients.json"), encoding="utf-8"))}
    jobs, expected = [], set()
    for c in contacts:
        if c["status"] != "data" or (a.patients and c["patient"] not in a.patients):
            continue
        P = patients.get(c["patient"], {})
        pid, ref = c["patient"], P.get("hg_reref", "WM")
        # a non-cohort patient (patients.json: tree / hfa_dir / hfa_stem) is read from the
        # frozen 04_ersp_LM tree with its HG/ folder; the bundle keeps its own HG/ layout
        qc_root = os.path.join(os.path.dirname(QC), P["tree"]) if P.get("tree") and P["tree"] != os.path.basename(QC) else QC
        hstem = P.get("hfa_stem", "HFAtrials")
        hdir = _pdir(os.path.join(qc_root, pid, "LM"), os.path.basename(P.get("hfa_dir", "PerTrial/HFA")))
        for cond in c["conds"]:
            src = os.path.join(hdir, cond, f"{pid}_{cond}_{ref}_{hstem}_{c['name']}.png")
            dst = os.path.join(OUT, pid, "HG", cond, f"{c['name']}.webp")
            jobs.append((src, dst, HG_W, HG_Q)); expected.add(os.path.normcase(dst))
    for pid, P in patients.items():
        if not P.get("ran") or (a.patients and pid not in a.patients):
            continue
        qc_root = os.path.join(os.path.dirname(QC), P["tree"]) if P.get("tree") and P["tree"] != os.path.basename(QC) else QC
        for cond in CONDS:
            jobs.append((os.path.join(_pdir(os.path.join(qc_root, pid, "LM"), "Report"), f"{pid}_{cond}_iqr_postDur_QC.png"),
                         os.path.join(OUT, pid, f"iqr_{cond}.webp"), FIG_W, FIG_Q))
            by_shaft = os.path.join(qc_root, pid, "LM", "PSD_clean", cond, "PSD", "psd_by_shaft.png")
            src = by_shaft if os.path.exists(by_shaft) else os.path.join(qc_root, pid, "LM", "PSD_clean", cond, "PSD", "psd_allch_full.png")
            jobs.append((src, os.path.join(OUT, pid, f"psd_{cond}.webp"), PSD_W if src == by_shaft else FIG_W, FIG_Q))
    t0 = time.time()
    counts, total = {"made": 0, "kept": 0, "missing": 0}, 0
    with ThreadPoolExecutor(a.workers) as ex:
        for k, (st, n) in enumerate(ex.map(lambda j: convert(*j), jobs), 1):
            counts[st] += 1; total += n
            if k % 1000 == 0:
                print(f"  {k}/{len(jobs)}  {time.time() - t0:.0f} s  {total / 1e6:.0f} MB", flush=True)
    stale = 0
    if not a.patients:
        for f in glob.glob(os.path.join(OUT, "*", "HG", "*", "*.webp")):
            if os.path.normcase(f) not in expected:
                os.remove(f); stale += 1
    n_hg = sum(1 for _ in glob.glob(os.path.join(OUT, "*", "HG", "*", "*.webp")))
    size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(OUT, "**", "*.webp"), recursive=True)) / 1e6
    manifest = {"built": datetime.now().strftime("%Y-%m-%d %H:%M"), "n_hg": n_hg, "hg_px": HG_W, "hg_quality": HG_Q,
                "fig_px": FIG_W, "psd_px": PSD_W, "size_mb": round(size, 1),
                "hg": "{pid}/HG/{cond}/{name}.webp", "iqr": "{pid}/iqr_{cond}.webp", "psd": "{pid}/psd_{cond}.webp"}
    os.makedirs(OUT, exist_ok=True)
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8"), indent=1)
    print(f"[done] {len(jobs)} figures: {counts['made']} made, {counts['kept']} kept, {counts['missing']} missing sources, "
          f"{stale} stale removed; qc/ holds {n_hg} HG rasters, {size:.0f} MB, {time.time() - t0:.0f} s -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
