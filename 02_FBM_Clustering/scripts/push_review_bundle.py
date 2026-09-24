"""
push_review_bundle.py - commit activity_viz/review/ to the activity-visualizer branch in commits
the repo's pre-commit guard accepts (< 25 MB of new files each, < 5 MB per file), then push.

    "C:/Users/fanda/AppData/Local/Programs/Python/Python311/python.exe" \
        02_FBM_Clustering/scripts/push_review_bundle.py [--no-push]

The branch lives in a worktree at C:/Users/fanda/av_branch (created here if missing). The
review folder of this checkout is copied over the worktree's, files whose content is already
on the branch are skipped, changed and new files are committed in chunks, files the bundle no
longer has are removed in a last commit. A second run after a failed push continues where it
stopped. The site's LM_visualizer.html reads the branch through raw.githubusercontent, so the
page shows the new bundle a minute after the push.
"""
import argparse
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WT = r"C:\Users\fanda\av_branch"
BRANCH = "activity-visualizer"
REL = "02_FBM_Clustering/outputs/250_recon/fsaverage/activity_viz"     # the whole bundle since 2026-09-22 (cube parts, ersp/, review/)
CHUNK_BYTES = 22e6
TRAILER = "\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\n"


def git(*args, cwd=WT, check=True):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        print(f"!! git {' '.join(args[:3])} failed:\n{(r.stderr or r.stdout)[-1500:]}", flush=True)
        sys.exit(1)
    return r


def git_paths(verb, paths):
    """git <verb> over many paths - through a pathspec file, the Windows command line is 32 kB."""
    lst = os.path.join(WT, ".git_pathspec.txt") if os.path.isdir(os.path.join(WT, ".git")) else os.path.join(os.environ.get("TEMP", WT), "review_pathspec.txt")
    with open(lst, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(paths) + "\n")
    r = git(verb, "--pathspec-from-file=" + lst, *(["-q"] if verb == "rm" else []))
    os.remove(lst)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args()
    if not os.path.isdir(os.path.join(WT, ".git")) and not os.path.isfile(os.path.join(WT, ".git")):
        print(f"creating worktree {WT} on {BRANCH}")
        git("fetch", "origin", BRANCH, cwd=REPO)
        git("worktree", "add", WT, BRANCH, cwd=REPO)
    git("checkout", BRANCH)
    git("pull", "--ff-only", "origin", BRANCH, check=False)

    src, dst = os.path.join(REPO, *REL.split("/")), os.path.join(WT, *REL.split("/"))
    os.makedirs(dst, exist_ok=True)
    # what the branch has, by content
    have = {}
    for line in git("ls-files", "-s", "--", REL).stdout.splitlines():
        mode, blob, stage, path = line.split(None, 3)
        have[path] = blob
    changed, total = [], 0
    for root, _, files in os.walk(src):
        for f in files:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, REPO).replace("\\", "/")
            blob = subprocess.run(["git", "hash-object", p], capture_output=True, text=True, cwd=REPO).stdout.strip()
            if have.get(rel) == blob:
                continue
            target = os.path.join(WT, *rel.split("/"))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(p, target)
            changed.append((rel, os.path.getsize(p)))
            total += os.path.getsize(p)
    wanted = {os.path.relpath(os.path.join(r, f), REPO).replace("\\", "/") for r, _, fs in os.walk(src) for f in fs}
    gone = sorted(set(have) - wanted)
    print(f"{len(changed)} files to commit ({total / 1e6:.0f} MB), {len(have) - len(gone)} unchanged on the branch, {len(gone)} to remove", flush=True)

    chunks, cur, size = [], [], 0
    for rel, sz in sorted(changed):
        if cur and size + sz > CHUNK_BYTES:
            chunks.append(cur); cur, size = [], 0
        cur.append(rel); size += sz
    if cur:
        chunks.append(cur)
    for i, paths in enumerate(chunks, 1):
        git_paths("add", paths)
        msg = f"activity_viz: the LM visualizer's bundle, part {i}/{len(chunks)}" + TRAILER
        git("commit", "-q", "-m", msg)
        print(f"  committed part {i}/{len(chunks)} ({len(paths)} files)", flush=True)
    if gone:
        git_paths("rm", gone)
        git("commit", "-q", "-m", "activity_viz: files the rebuilt bundle no longer has" + TRAILER)
        print(f"  removed {len(gone)} files")
    if not chunks and not gone:
        print("nothing to commit: the branch already holds this bundle")
    if a.no_push:
        print("(--no-push: not pushed)")
        return 0
    git("push", "origin", BRANCH)
    print("pushed", BRANCH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
