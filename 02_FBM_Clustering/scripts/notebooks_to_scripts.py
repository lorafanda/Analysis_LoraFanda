"""
notebooks_to_scripts.py - write 240 / 241 / 242 as terminal scripts, from the notebooks.

    python 02_FBM_Clustering/scripts/notebooks_to_scripts.py            # rewrites the three .py next to the notebooks
    python 02_FBM_Clustering/scripts/notebooks_to_scripts.py OUT_DIR    # elsewhere, to compare

The scripts are the notebooks' five code cells in order (settings, load, fit, held-out, done)
with only what a terminal needs changed: ROOT from __file__, the Agg backend, print() for
display(), and three flags (--ks, --feature-set, --no-heldout). Any other code cell in a
notebook (a pip line, a diagnostic) is reported and left out. Run it again after editing a
notebook, so the two never drift apart (2026-10-09).
"""
import json
import os
import re
import sys

CLUST = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else CLUST
NBS = ["240_cluster_kmeans.ipynb", "241_cluster_hierarchical.ipynb", "242_cluster_cnmf.ipynb"]

HEADER = '''"""
{py} - {nb} as a terminal script (2026-10-09).

    cd 02_FBM_Clustering
    python {py}                              # exactly the notebook: K 5..30, four feature sets, then the held-out curve
    python {py} --no-heldout                 # the fits only (section 3, make_heldout_variance.py, is the long part)
    python {py} --ks 5 6 7 8 9 10 11 12      # a shorter K range
    python {py} --feature-set concat_hg      # one feature set

The notebook's code cells, in their order, with only what a terminal needs changed: ROOT is
this file's folder instead of the kernel's cwd, matplotlib draws off-screen (Agg), the held-out
table is printed instead of display()ed, and three flags. Everything else - the cache
resolution, the skip-if-run-exists rule, the fit parameters - is the notebook's. Regenerate
from the notebook when it changes; do not edit the two apart.
{order}"""
'''
ORDER = {
    "240_cluster_kmeans.ipynb": "240 and 241 can run in parallel (separate run directories); 242 needs 240's k-means\nrun of the same cohort first (run_decomposition.py takes its matrix from it).\n",
    "241_cluster_hierarchical.ipynb": "240 and 241 can run in parallel (separate run directories); 242 needs 240's k-means\nrun of the same cohort first (run_decomposition.py takes its matrix from it).\n",
    "242_cluster_cnmf.ipynb": "Run AFTER 240: run_decomposition.py takes its matrix from the newest k-means run of the\ncurrent cache, and the last check here asserts the two electrode sets are the same.\n",
}

ARGPARSE = '''
# ---- terminal flags (the notebook has none; defaults reproduce it) ----------------------
ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
ap.add_argument('--ks', type=int, nargs='+', default=K_RANGE, help='K values to fit (default: the notebook\\'s 5..30)')
ap.add_argument('--feature-set', nargs='+', default=FEATURE_SETS, choices=FEATURE_SETS, help='feature sets to fit (default: all four)')
ap.add_argument('--no-heldout', action='store_true', help='skip section 3, the bi-cross-validated held-out curve (make_heldout_variance.py)')
args = ap.parse_args()
K_RANGE = sorted(set(args.ks))
FEATURE_SETS = [f for f in FEATURE_SETS if f in args.feature_set]
'''


def convert(nb_name):
    cells = json.load(open(os.path.join(CLUST, nb_name), encoding="utf-8"))["cells"]
    py = nb_name.replace(".ipynb", ".py")
    parts = [HEADER.format(py=py, nb=nb_name, order=ORDER[nb_name])]
    n_code = 0
    for c in cells:
        src = "".join(c["source"]).strip("\n")
        if c["cell_type"] == "markdown":
            lines = [l.strip() for l in src.splitlines() if l.strip()]
            if lines and lines[0].startswith("## "):
                parts.append("\n# " + "=" * 88 + "\n# " + lines[0] + "\n# " + "=" * 88 + "\n")
            continue
        if not src.strip():
            continue
        # only the notebook's five sections; anything else (a pip line, a diagnostic) is reported and left out
        markers = ("ROOT = Path.cwd()", "build_concat_dataset(", "SKIP_IF_RUN_EXISTS = True",
                   "make_heldout_variance.py", "print('DONE:'")
        if not any(m in src for m in markers):
            print(f"  {nb_name}: left out a code cell that is not one of the five sections: {src.splitlines()[0][:70]!r}")
            continue
        n_code += 1
        if "ROOT = Path.cwd()" in src:                                      # the settings cell
            src = src.replace("import os, sys, json, subprocess, time", "import argparse, os, sys, json, subprocess, time", 1)
            src = src.replace("ROOT = Path.cwd()", "ROOT = Path(__file__).resolve().parent", 1)
            src = src.replace("import matplotlib.pyplot as plt", "import matplotlib\nmatplotlib.use('Agg')          # terminal: draw off-screen\nimport matplotlib.pyplot as plt", 1)
            src = re.sub(r"SCRIPT_NAME = '(\d+_[a-z_]+)\.ipynb'", r"SCRIPT_NAME = '\1.py'", src, count=1)
            assert "RANDOM_STATE= cfg.RANDOM_STATE" in src, nb_name
            src = src.replace("RANDOM_STATE= cfg.RANDOM_STATE", "RANDOM_STATE= cfg.RANDOM_STATE\n" + ARGPARSE, 1)
            assert "SCRIPT_NAME = '" + py + "'" in src, nb_name
        if "make_heldout_variance.py" in src and "display(pk)" in src:   # the held-out cell
            src = src.replace("display(pk)", "print(pk.to_string(index=False))", 1)
            body = "\n".join(("    " + l) if l.strip() else l for l in src.splitlines())
            src = ("if args.no_heldout:\n"
                   "    print('\\nsection 3 (held-out variance, make_heldout_variance.py) skipped: --no-heldout')\n"
                   "else:\n" + body)
        assert "display(" not in src and not src.lstrip().startswith("%"), nb_name
        parts.append("\n" + src + "\n")
    text = "".join(parts)
    assert n_code == 5, (nb_name, n_code)
    assert "Path.cwd()" not in text and ".ipynb'" not in text.split('"""', 2)[2], nb_name
    compile(text, py, "exec")
    out = os.path.join(OUT, py)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"{py}: {n_code} cells -> {len(text.splitlines())} lines -> {out}")


for nb in NBS:
    convert(nb)
