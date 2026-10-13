"""
240_cluster_kmeans.py - 240_cluster_kmeans.ipynb as a terminal script (2026-10-09).

    cd 02_FBM_Clustering
    python 240_cluster_kmeans.py                              # exactly the notebook: K 5..30, four feature sets, then the held-out curve
    python 240_cluster_kmeans.py --no-heldout                 # the fits only (section 3, make_heldout_variance.py, is the long part)
    python 240_cluster_kmeans.py --ks 5 6 7 8 9 10 11 12      # a shorter K range
    python 240_cluster_kmeans.py --feature-set concat_hg      # one feature set

The notebook's code cells, in their order, with only what a terminal needs changed: ROOT is
this file's folder instead of the kernel's cwd, matplotlib draws off-screen (Agg), the held-out
table is printed instead of display()ed, and three flags. Everything else - the cache
resolution, the skip-if-run-exists rule, the fit parameters - is the notebook's. Regenerate
from the notebook when it changes; do not edit the two apart.
240 and 241 can run in parallel (separate run directories); 242 needs 240's k-means
run of the same cohort first (run_decomposition.py takes its matrix from it).
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

SCRIPT_NAME = '240_cluster_kmeans.py'
METHOD      = 'kmeans'
METHOD_LBL  = 'k-means'
# THE SPACE THE FIT HAPPENS IN. unit-norm since 2026-09-06: each electrode scaled
# to unit length before the fit, the space convex NMF has always used - so the three
# methods see the same matrix and a squared-Euclidean fit cannot sort electrodes by
# amplitude first. X_train.npy stays raw dB; the manifest records the space.
FIT_SPACE   = 'unit-norm'   # 'dB' reproduces the runs made before 2026-09-06
# the cache lf_concat points at - the newest concat_source_v<N> built by
# rebuild_concat_cache.py. Pinned here until 2026-09-06, which meant a refit
# after a rebuild silently used the previous cohort. Set LF_CONCAT_CACHE to
# reproduce a published run on its own cache.
CACHE_DIR   = CC.DEFAULT_CONCAT_CACHE
FEATURE_SETS= ['concat_bands15', 'concat_bands5', 'concat_bands2', 'concat_hfa']
# ---- the HFA time-bin sweep (2026-10-13) -----------------------------------------------
# concat_hfa at other time resolutions, same 70-150 Hz band, same anti-aliased resize.
# Not in FEATURE_SETS: they are fitted only when named with --feature-set, e.g.
#     python 240_cluster_kmeans.py --feature-set concat_hfa300 concat_hfa30 --ks 5 6 7 8 9 10
TIMEBIN_SETS = {'concat_hfa300': 300, 'concat_hfa200': 200, 'concat_hfa150': 150, 'concat_hfa100': 100, 'concat_hfa30': 30, 'concat_hfa10': 10}

K_RANGE     = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
N_ITER      = 1000          # convex NMF fit iterations (final fits)
RANDOM_STATE= cfg.RANDOM_STATE

# ---- terminal flags (the notebook has none; defaults reproduce it) ----------------------
ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
ap.add_argument('--ks', type=int, nargs='+', default=K_RANGE, help='K values to fit (default: the notebook\'s 5..30)')
ap.add_argument('--feature-set', nargs='+', default=FEATURE_SETS, choices=FEATURE_SETS + list(TIMEBIN_SETS), help='feature sets to fit (default: all four)')
ap.add_argument('--no-heldout', action='store_true', help='skip section 3, the bi-cross-validated held-out curve (make_heldout_variance.py)')
args = ap.parse_args()
K_RANGE = sorted(set(args.ks))
FEATURE_SETS = [f for f in list(FEATURE_SETS) + list(TIMEBIN_SETS)
                if f in args.feature_set]


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

X = {# 2026-10-13: the four sets differ in FREQUENCY RESOLUTION ONLY - 15 / 5 / 2 / 1
     # bands on one shared 30-bin time grid, 1350 / 450 / 180 / 90 features. concat_hg
     # (the same HFA band at the cube's 300 bins) and concat_rawds / concat_bands5z are
     # still in lf_concat and every run fitted on them still resolves; they are just not
     # in this comparison any more.
     'concat_bands15': CC.concat_bands15_features(X_concat, n_blocks=3, fmax_hz=398.4375),
     # the same builder with 5 band edges instead of 15, each a union of contiguous
     # 15-band edges. It averages the ORIGINAL frequency bins, so a wide band is
     # weighted by the bins it contains rather than giving a 3 Hz sub-band the same say
     # as a 40 Hz one - and the two sets stay nested.
     'concat_bands5': CC.concat_bands5_features(X_concat, n_blocks=3, fmax_hz=398.4375),
     # low gamma 30-50 Hz and HFA 70-150 Hz, nothing else. 50-70 Hz is left out rather
     # than averaged in: it is the least informative stretch of the spectrum (eta2
     # 0.11-0.14, bottom two of fifteen).
     'concat_bands2': CC.concat_bands2_features(X_concat, n_blocks=3, fmax_hz=398.4375),
     # 70-150 Hz on the same 30-bin grid as the band sets - NOT concat_hg, which is the
     # same band at 300 bins and is what the time-bin sweep varies.
     'concat_hfa': CC.concat_hfa_features(X_concat, n_blocks=3, fmax_hz=398.4375)}
# the sweep sets are built only when asked for: each one is a full pass over the
# cohort's cubes, and the four standard sets are what a plain run is for
for _fs in FEATURE_SETS:
    if _fs in TIMEBIN_SETS:
        X[_fs] = CC.concat_hfa_features(X_concat, n_blocks=3, fmax_hz=398.4375,
                                        time_bins_out=TIMEBIN_SETS[_fs])


print(f'\ncohort: {len(df_contacts)} electrodes · '
      f'{df_contacts.patient_id.nunique()} patients')
for k_, v in X.items():
    print(f'  {k_:<14} {v.shape}')

# ========================================================================================
# ## 2 - Fit k-means at every K
# ========================================================================================

# one loop, every feature set, every K in one fit_and_save call
#
# REFITTING A FEATURE SET DOES NOT UPDATE ITS RUN - IT SUPERSEDES IT. Run ids are
# timestamps and lf_runs.newest_run takes the last, so a second fit of concat_hg
# leaves the first one's statistics, stability sweeps and coverage-manifest entry
# stranded on a run nothing resolves any more. Set SKIP_IF_RUN_EXISTS = False only
# when you actually mean to replace one.
SKIP_IF_RUN_EXISTS = True

results, skipped = {}, []
for fs in FEATURE_SETS:
    if SKIP_IF_RUN_EXISTS:
        try:
            rd_ = LR.newest_run(METHOD, fs)
            lab_ = pd.read_csv(rd_ / 'cluster_labels_by_k.csv')
            have_ = {int(c[2:]) for c in lab_.columns if c.startswith('k_')}
            sp_ = (json.load(open(rd_ / 'manifest.json', encoding='utf-8')).get('params') or {}).get('space', 'dB')
            # THE COHORT TOO. A run that covers every K in the right space can still be the
            # previous cohort's (the cache is rebuilt under a new version, the run is not).
            # Skip only if its electrodes are the ones just loaded, in the same order - the
            # comparison 242 makes between its cnmf run and the newest k-means run.
            key_ = lambda d: d[['patient_id', 'electrode']].astype(str).agg('|'.join, axis=1).tolist()
            same_ = key_(pd.read_csv(rd_ / 'labels.csv', usecols=['patient_id', 'electrode'])) == key_(df_contacts)
            if set(K_RANGE).issubset(have_) and sp_ == FIT_SPACE and same_:
                skipped.append(fs)
                print(f'{fs:<15} SKIPPED - {rd_.name} already covers this K range')
                continue
            print(f'{fs:<15} refitting - {rd_.name} '
                  + ('was fitted on a different electrode set than the cohort just loaded' if not same_
                     else f'was fitted in {sp_} space, not {FIT_SPACE}' if sp_ != FIT_SPACE
                     else f'is missing K={sorted(set(K_RANGE) - have_)}'))
        except (FileNotFoundError, LR.StaleRunError):
            pass
    t0 = time.time()
    params = ({'k_range': K_RANGE, 'random_state': RANDOM_STATE, 'n_init': 20, 'space': FIT_SPACE}
              if METHOD == 'kmeans' else
              {'linkage': 'ward', 'metric': 'euclidean', 'k_range': K_RANGE, 'space': FIT_SPACE})
    m = R.fit_and_save(
        X[fs], df_keep=df_contacts, method=METHOD, feature_set=fs, params=params,
        method_label=METHOD_LBL,
        # .get, not [fs]: a bare lookup makes a new feature set a KeyError
        # AFTER the features are built and the loop has started.
        feature_set_label={'concat_hg':     'Concatenated HG [a|p|r]',
                           'concat_rawds':  'Concatenated 15-band [a|p|r]',
                           'concat_bands5': 'Concatenated 5-band [a|p|r]',
                           'concat_bands5z':'Concatenated 5-band, z-scored [a|p|r]'}.get(fs, fs),
        feature_names=CC.concat_feature_names(fs), notebook=SCRIPT_NAME)
    results[fs] = m
    print(f'{fs:<15} best_k={m["summary"]["best_k"]}   ({time.time()-t0:.0f}s)')

if skipped:
    print(f'\n{len(skipped)} already fitted and left alone: {skipped}')

# verify EVERY feature set, skipped ones included - a skip is only safe if the run it
# skipped for really does hold the K range
for fs in FEATURE_SETS:
    rd = LR.newest_run(METHOD, fs)
    lab = pd.read_csv(rd / 'cluster_labels_by_k.csv')
    have = sorted(int(c[2:]) for c in lab.columns if c.startswith('k_'))
    tag = '(skipped)' if fs in skipped else ''
    print(f'{fs:<15} {rd.name}   K columns {have[0]}..{have[-1]} ({len(have)}) {tag}')
    assert set(K_RANGE).issubset(have), f'{fs}: missing K columns'

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
