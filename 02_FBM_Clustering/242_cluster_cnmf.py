"""
242_cluster_cnmf.py - 242_cluster_cnmf.ipynb as a terminal script (2026-10-09).

    cd 02_FBM_Clustering
    python 242_cluster_cnmf.py                              # exactly the notebook: K 5..30, four feature sets, then the held-out curve
    python 242_cluster_cnmf.py --no-heldout                 # the fits only (section 3, make_heldout_variance.py, is the long part)
    python 242_cluster_cnmf.py --ks 5 6 7 8 9 10 11 12      # a shorter K range
    python 242_cluster_cnmf.py --feature-set concat_hg      # one feature set

The notebook's code cells, in their order, with only what a terminal needs changed: ROOT is
this file's folder instead of the kernel's cwd, matplotlib draws off-screen (Agg), the held-out
table is printed instead of display()ed, and three flags. Everything else - the cache
resolution, the skip-if-run-exists rule, the fit parameters - is the notebook's. Regenerate
from the notebook when it changes; do not edit the two apart.
Run AFTER 240: run_decomposition.py takes its matrix from the newest k-means run of the
current cache, and the last check here asserts the two electrode sets are the same.
"""

import argparse, os, sys, json, subprocess, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')          # terminal: draw off-screen
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
assert (ROOT / 'functions').exists(), f'run this from 02_FBM_Clustering, not {ROOT}'
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'functions'))
# cfg is lf_blob_clustering_config here - 02_FBM_Clustering has no config.py;
# that name belongs to 01_FBM_Analysis. RANDOM_STATE lives in this one.
import lf_concat as CC, lf_cluster_run as R, lf_runs as LR
import lf_blob_clustering_config as cfg

SCRIPT_NAME = '242_cluster_cnmf.py'
METHOD      = 'cnmf'
METHOD_LBL  = 'convex NMF'
# the cache lf_concat points at - the newest concat_source_v<N> built by
# rebuild_concat_cache.py. Pinned here until 2026-09-06, which meant a refit
# after a rebuild silently used the previous cohort. Set LF_CONCAT_CACHE to
# reproduce a published run on its own cache.
CACHE_DIR   = CC.DEFAULT_CONCAT_CACHE
FEATURE_SETS= ['concat_hg', 'concat_rawds', 'concat_bands5', 'concat_bands5z']
K_RANGE     = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
N_ITER      = 1000          # convex NMF fit iterations (final fits)
RANDOM_STATE= cfg.RANDOM_STATE

# ---- terminal flags (the notebook has none; defaults reproduce it) ----------------------
ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
ap.add_argument('--ks', type=int, nargs='+', default=K_RANGE, help='K values to fit (default: the notebook\'s 5..30)')
ap.add_argument('--feature-set', nargs='+', default=FEATURE_SETS, choices=FEATURE_SETS, help='feature sets to fit (default: all four)')
ap.add_argument('--no-heldout', action='store_true', help='skip section 3, the bi-cross-validated held-out curve (make_heldout_variance.py)')
args = ap.parse_args()
K_RANGE = sorted(set(args.ks))
FEATURE_SETS = [f for f in FEATURE_SETS if f in args.feature_set]


INPUT_DIR = Path(r'\\nasac-m2.unige.ch\m-HumanNeuronLab\ANALYSIS\FLM\Analysis_LoraFanda\01_FBM_Analysis\outputs\03_ERSP')
if not INPUT_DIR.exists():
    INPUT_DIR = (ROOT.parent / '01_FBM_Analysis' / 'outputs' / '03_ERSP').resolve()

print('method      :', METHOD_LBL)
print('feature sets:', FEATURE_SETS)
print('K range     :', K_RANGE[0], '..', K_RANGE[-1], f'({len(K_RANGE)} values)')
print('cache       :', CACHE_DIR)

# ========================================================================================
# ## 1 - Load the shared cohort
# ========================================================================================

# The cache MUST already exist. rebuild_concat_cache.py builds it once, serially.
# Building it here would mean three kernels racing to write the same files, and
# prepare_dataset's cache key carries no patient list, so a half-written cache would
# be picked up as a hit by whichever kernel arrived second.
assert CACHE_DIR.exists() and (CACHE_DIR / 'X_3d.npy').exists(), (
    f'{CACHE_DIR} missing - run:  python rebuild_concat_cache.py --apply')

df_contacts, X_concat = CC.build_concat_dataset(
    INPUT_DIR, conditions=('audio','picture','reading'),
    require_high_activity=True, cache_dir=CACHE_DIR, verbose=True)

X = {'concat_hg':     CC.concat_hg_features(X_concat, hg_band=(70.0,150.0), fmax=398.4375),
     'concat_rawds':  CC.concat_rawds_features(X_concat, n_blocks=3, fmax_hz=398.4375),
     # concat_bands5 is the SAME builder with 5 band edges instead of 15, each a
     # union of contiguous 15-band edges. It averages the ORIGINAL frequency bins,
     # so a wide band is weighted by the bins it contains rather than giving a 3 Hz
     # sub-band the same say as a 40 Hz one - and the two sets stay nested.
     'concat_bands5': CC.concat_bands5_features(X_concat, n_blocks=3, fmax_hz=398.4375),
     # the same five bands with each one z-scored to equal weight. NOT dB any
     # more - the units are SD within a band - and it is a COHORT-level rescale,
     # so it changes with the cohort and must be rebuilt when the cohort does.
     'concat_bands5z': CC.concat_bands5z_features(X_concat, n_blocks=3, fmax_hz=398.4375)}

print(f'\ncohort: {len(df_contacts)} electrodes · '
      f'{df_contacts.patient_id.nunique()} patients')
for k_, v in X.items():
    print(f'  {k_:<14} {v.shape}')

# ========================================================================================
# ## 2 - Fit convex NMF at every K
# ========================================================================================

# convex NMF: run_decomposition creates the run, sweep_decomposition fills in
# loadings_by_k / components_by_k / cluster_labels_by_k for EVERY K.
#
# REFITTING A FEATURE SET DOES NOT UPDATE ITS RUN - IT SUPERSEDES IT. Run ids are
# timestamps and lf_runs.newest_run takes the last, so a second pass over concat_hg
# leaves the first run's statistics, stability sweeps and coverage-manifest entry
# stranded on a run nothing resolves any more. Set SKIP_IF_RUN_EXISTS = False only
# when you actually mean to replace one.
SKIP_IF_RUN_EXISTS = True

def sh(cmd):
    print('$', ' '.join(str(c) for c in cmd), flush=True)
    r = subprocess.run([sys.executable, *[str(c) for c in cmd]],
                       cwd=str(ROOT), env={**os.environ, 'PYTHONIOENCODING':'utf-8'})
    assert r.returncode == 0, f'failed: {cmd}'

skipped = []
for fs in FEATURE_SETS:
    if SKIP_IF_RUN_EXISTS:
        try:
            rd_ = LR.newest_run('cnmf', fs)
            have_ = {int(p.stem[3:]) for p in (rd_/'loadings_by_k').glob('G_k*.npy')}
            # THE COHORT, NOT ONLY THE K RANGE. run_decomposition.py takes its matrix from
            # the newest k-means run, so a cnmf run made from an older cohort covers every K
            # and still has the wrong electrodes (v4 = 1693 rows, v5 = 1675). Skip only if
            # the rows and the patient|electrode keys match the newest k-means run.
            km_ = LR.newest_run('kmeans', fs)
            key_ = lambda d: pd.read_csv(d/'labels.csv', usecols=['patient_id', 'electrode']).astype(str).agg('|'.join, axis=1).tolist()
            same_ = (np.load(rd_/'X_train.npy', mmap_mode='r').shape[0] == np.load(km_/'X_train.npy', mmap_mode='r').shape[0]
                     and key_(rd_) == key_(km_))
            if set(K_RANGE).issubset(have_) and same_:
                skipped.append(fs)
                print(f'{fs:<15} SKIPPED - {rd_.name} already covers this K range')
                continue
            print(f'{fs:<15} refitting - {rd_.name} '
                  + ('was fitted on a different electrode set than the newest k-means run' if not same_
                     else f'is missing K={sorted(set(K_RANGE) - have_)}'))
        except (FileNotFoundError, LR.StaleRunError):
            pass
    t0 = time.time()
    sh(['run_decomposition.py', '--feature-set', fs])
    # PUBLISH before sweeping. run_decomposition writes decomposition/<fset>/;
    # publish_decomposition turns that into a REGISTERED run under cnmf/<fset>/runs/.
    # Without it sweep_decomposition --run defaults to 'newest', which is whatever
    # cnmf run already existed - so the K sweep lands in a stale run and lf_runs keeps
    # resolving cnmf to a cohort that no longer matches k-means and Ward.
    sh(['publish_decomposition.py', '--feature-set', fs])
    sh(['sweep_decomposition.py', '--feature-set', fs,
        '--ks', *[str(k) for k in K_RANGE], '--n-iter', str(N_ITER)])
    print(f'{fs} done in {time.time()-t0:.0f}s')

if skipped:
    print(f'\n{len(skipped)} already fitted and left alone: {skipped}')

import numpy as np
for fs in FEATURE_SETS:
    rd = LR.newest_run('cnmf', fs)
    have = sorted(int(p.stem[3:]) for p in (rd/'loadings_by_k').glob('G_k*.npy'))
    n_cnmf = np.load(rd/'X_train.npy', mmap_mode='r').shape[0]
    n_km   = np.load(LR.newest_run('kmeans', fs)/'X_train.npy', mmap_mode='r').shape[0]
    tag = '(skipped)' if fs in skipped else ''
    print(f'{fs:<15} {rd.name}   loadings K {have[0]}..{have[-1]} ({len(have)})   '
          f'n={n_cnmf} vs kmeans n={n_km} {tag}')
    assert set(K_RANGE).issubset(have), f'{fs}: missing loadings_by_k'
    # the check that would have caught the stale-source bug immediately
    assert n_cnmf == n_km, (f'{fs}: cnmf fitted on {n_cnmf} electrodes but k-means on '
                            f'{n_km} - the decomposition used a stale source run')

# ========================================================================================
# ## 3 - Held-out variance, K=5..30
# ========================================================================================

if args.no_heldout:
    print('\nsection 3 (held-out variance, make_heldout_variance.py) skipped: --no-heldout')
else:
    # Held-out variance, K=5..30, HOME SPACE only, bi-cross-validated.
    # --tag keeps this kernel's output in its own file so the three notebooks running in
    # parallel never write to the same CSV.
    cmd = [sys.executable, 'make_heldout_variance.py',
           '--from-cache', str(CACHE_DIR),
           '--feature-set', *FEATURE_SETS,
           '--method', METHOD, '--tag', METHOD,
           '--ks', *[str(k) for k in K_RANGE],
           '--spaces', 'home', '--n-iter', '300']
    print('$', ' '.join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT), env={**os.environ, 'PYTHONIOENCODING':'utf-8'})
    assert r.returncode == 0

    pk = pd.read_csv(ROOT/'outputs'/'clustering'/'bsf_comparison'/f'heldout_peaks_{METHOD}.csv')
    print(pk.to_string(index=False))

# ========================================================================================
# ## 4 - Done
# ========================================================================================

print('DONE:', METHOD_LBL)
print('When ALL THREE of 240 / 241 / 242 have finished, run 249.')
