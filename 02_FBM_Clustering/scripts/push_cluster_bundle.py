"""
push_cluster_bundle.py - commit the cluster visualizer's two data bundles to main in commits
the repo's pre-commit guard accepts (< 25 MB of files each, < 5 MB per file), then push.

    "C:/Users/fanda/AppData/Local/Programs/Python/Python311/python.exe" \
        02_FBM_Clustering/scripts/push_cluster_bundle.py            # plan only
        ... --commit                                                 # commit, no push
        ... --commit --push                                          # commit and push main

WHAT. cluster_visualizer.html fetches its data from raw.githubusercontent.com/.../main/, so the
coverage bundle (outputs/250_recon/fsaverage/coverage_viz: manifest, contacts, coverage maps,
neighbour index, runs/<run>/) and the centroid bundle (outputs/clustering/paper_web/centroids:
index, x_<feature set>.bin, <run>.json / .w.bin) must be ON MAIN before the page can show a
new cohort. make_coverage_bundle.py and make_centroid_bundle.py write them; this commits them.
Until 2026-10-11 this was done by hand ("Cluster visualizer bundle on cohort v11 (i/10)").

WHICH FILES. Every file under the two folders that git sees as modified or untracked, EXCEPT
per-run files of runs that are not in the coverage manifest's run list (older runs' folders
stay on disk and are not the page's business). The repo's index belongs to Lora (thousands
of staged files), so every commit is `git commit --only` with an explicit pathspec file -
nothing else that is staged or modified goes in.

CHUNKS. Files are packed greedily into commits of at most CHUNK_BYTES (22 MB, under the
guard's 25 MB); a file over 5 MB stops the run before anything is committed.
"""
import argparse
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COV = "02_FBM_Clustering/outputs/250_recon/fsaverage/coverage_viz"
CEN = "02_FBM_Clustering/outputs/clustering/paper_web/centroids"
CHUNK_BYTES = 22 * 1048576
MAX_FILE = 5 * 1048576
TRAILER = "\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\n"


def git(*args, check=True):
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        print(f"!! git {' '.join(args[:4])} failed:\n{(r.stderr or r.stdout)[-1500:]}", flush=True)
        sys.exit(1)
    return r


def changed_files():
    """(path, status) of every modified / untracked file under the two bundle folders."""
    out = []
    for line in git("status", "--short", "--untracked-files=all", "--", COV, CEN).stdout.splitlines():
        st, path = line[:2], line[3:].strip()
        if path.startswith('"') and path.endswith('"'):
            path = path[1:-1]
        if "D" in st:
            continue                       # a deleted file is not pushed by this script
        out.append((path, st))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true", help="make the commits (default: print the plan)")
    ap.add_argument("--push", action="store_true", help="push main afterwards (needs --commit)")
    ap.add_argument("--label", default=None, help="cohort label for the messages (default: the cache name of the first run's manifest, e.g. 'cohort v15')")
    ap.add_argument("--with-stats", action="store_true",
                    help="also commit each manifest run's cluster_statistics.json (249's output; the report's per-cluster cards read it from main - "
                         "nothing else of a run folder goes, the chip and raster PNGs are thousands of files)")
    ap.add_argument("--with-chips", action="store_true",
                    help="also commit each manifest run's centroid chips at the manifest's sweep K values (cluster_centroids/k_K/*.png, "
                         "~70 PNGs per run for K 5-12): the report's K chips read them from main (2026-10-13)")
    a = ap.parse_args()

    man = json.load(open(os.path.join(REPO, COV, "manifest.json"), encoding="utf-8"))
    runs = [r if isinstance(r, str) else (r.get("id") or r.get("run")) for r in man["runs"]]
    label = a.label
    if not label:
        try:
            first = man["runs"][0]
            cache = (first.get("cache") or {}).get("name") if isinstance(first, dict) else None
            label = f"cohort {cache.split('_v')[-1] and 'v' + cache.split('_v')[-1]}" if cache else "the current cohort"
        except Exception:
            label = "the current cohort"

    def wanted(path):
        name = os.path.basename(path)
        per_run = (f"/{COV}/runs/" in f"/{path}") or (path.startswith(CEN + "/") and name.split("__")[0] in ("cnmf", "kmeans", "hierarchical"))
        return (any(rid in path for rid in runs)) if per_run else True

    files, skipped = [], []
    for path, st in changed_files():
        (files if wanted(path) else skipped).append(path)
    if a.with_stats:
        # run id "cnmf__concat_hg__20261012_184011" -> outputs/clustering/cnmf/concat_hg/runs/20261012_184011
        for rid in runs:
            method, fset, stamp = rid.split("__")
            rel = f"02_FBM_Clustering/outputs/clustering/{method}/{fset}/runs/{stamp}/cluster_statistics.json"
            if not os.path.exists(os.path.join(REPO, rel)):
                print(f"  (no cluster_statistics.json yet for {rid} - 249 has not reached it)")
            elif git("status", "--short", "--", rel).stdout.strip():
                files.append(rel)
            else:
                print(f"  ({rid}: cluster_statistics.json already on main)")
    if a.with_chips:
        import glob
        ks = man.get("sweep_ks") or list(range(5, 13))
        for rid in runs:
            method, fset, stamp = rid.split("__")
            for k in ks:
                for p in sorted(glob.glob(os.path.join(REPO, f"02_FBM_Clustering/outputs/clustering/{method}/{fset}/runs/{stamp}/cluster_centroids/k_{k}/*.png"))):
                    rel = os.path.relpath(p, REPO).replace("\\", "/")
                    if git("status", "--short", "--", rel).stdout.strip():
                        files.append(rel)
    if not files:
        print("nothing to commit: main already holds these bundles")
        return
    sizes = {p: os.path.getsize(os.path.join(REPO, p)) for p in files}
    too_big = [p for p, s in sizes.items() if s > MAX_FILE]
    if too_big:
        print("!! over the 5 MB guard, nothing committed:", *too_big, sep="\n   ")
        sys.exit(1)

    # shared files first (manifest last of them, so a reader of main never sees a manifest
    # whose runs are not there yet), then run by run in manifest order
    shared = sorted(p for p in files if not any(rid in p for rid in runs))
    shared = [p for p in shared if not p.endswith("manifest.json")]
    per_run = [p for rid in runs for p in sorted(files) if rid in p]
    manifests = [p for p in files if p.endswith("manifest.json")]
    ordered = shared + per_run + manifests
    assert len(ordered) == len(files), (len(ordered), len(files))

    chunks, cur, size = [], [], 0
    for p in ordered:
        if cur and size + sizes[p] > CHUNK_BYTES:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(p)
        size += sizes[p]
    if cur:
        chunks.append(cur)

    total = sum(sizes.values()) / 1048576
    print(f"{len(files)} files, {total:.1f} MB, {len(chunks)} commit(s) of <= {CHUNK_BYTES / 1048576:.0f} MB; "
          f"{len(skipped)} file(s) of other runs left out; runs in the manifest: {len(runs)}")
    for i, c in enumerate(chunks, 1):
        print(f"  {i:2d}/{len(chunks)}: {len(c):3d} files {sum(sizes[p] for p in c) / 1048576:5.1f} MB   {c[0].split('/')[-2]}/{os.path.basename(c[0])} ..")
    if not a.commit:
        print("(plan only: add --commit, and --push to push main)")
        return

    lst = os.path.join(os.environ.get("TEMP", REPO), "cluster_bundle_pathspec.txt")
    for i, c in enumerate(chunks, 1):
        with open(lst, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(c) + "\n")
        git("add", "--pathspec-from-file=" + lst)          # no "--" before it: git would read the option as a pathspec
        msg = f"Cluster visualizer bundle on {label} ({i}/{len(chunks)}): " + ", ".join(
            sorted({p.split("/")[-2] if "/runs/" in p else os.path.basename(p).split(".")[0] for p in c})[:4]) + TRAILER
        git("commit", "-q", "--only", "--pathspec-from-file=" + lst, "-m", msg)
        print(f"  committed {i}/{len(chunks)} ({len(c)} files)", flush=True)
    print(git("log", "-1", "--format=%h %s").stdout.strip()[:100])
    if a.push:
        git("push", "origin", "main")
        print("pushed main")
    else:
        print("(not pushed: add --push)")


if __name__ == "__main__":
    main()
