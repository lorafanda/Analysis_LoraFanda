"""Step 0 of member_measures (optional; Lora, 2026-11-03): unthresholded NeuroSynth association maps
and contrasts, computed with NiMARE from the NeuroSynth database, for the naming task.

    pip install nimare                      (laptop Python 3.11, once)
    python 0_nimare_maps.py --counts        study counts for every candidate term, nothing computed
    python 0_nimare_maps.py                 the maps (minutes each)

THE 200-STUDY RULE: a term is used only if at least MIN_STUDIES studies carry it (NeuroSynth's own
labelling: tf-idf of the term in the abstract >= 0.001); a pooled term counts the union of its
parts; a contrast needs MIN_STUDIES on BOTH sides. Everything below the rule is listed in
study_counts.csv and skipped.

THE STATISTIC: NiMARE's MKDA chi-square, the test behind NeuroSynth's association map, with
NeuroSynth's 6-mm kernel, WITHOUT a threshold:
    single term   studies with the term vs every other study in the database
    contrast      studies with term X vs studies with term Y (both sides >= MIN_STUDIES; the
                  studies carrying both are dropped from both sides)
Output: 04_FBM_Pooling/neurosynth_nimare/<name>/<name>_association-test_z.nii.gz, one folder per
map, and study_counts.csv / maps.csv (name, stage, kind, n studies) beside them. Step 4 with
--ns nimare reads this folder.

NOT TESTED until NiMARE is installed: the NiMARE calls follow its documented API (fetch_neurosynth,
convert_neurosynth_to_dataset, MKDAChi2); map names differ across NiMARE versions, so both the old
and the new name of the association map are tried."""
import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = ROOT / "04_FBM_Pooling" / "neurosynth_nimare"
DATA = ROOT / "04_FBM_Pooling" / "neurosynth_nimare" / "_database"
MIN_STUDIES = 200

# (name, stage, kind, terms or (terms X, terms Y)); a list of terms = pooled into one study set
# Revised 2026-11-04 from the first --counts run: every term below exists in NeuroSynth v7's vocabulary
# and every map and both sides of every contrast have >= 200 studies (counts in the comments, from the
# tf-idf >= 0.001 rule; contrasts after dropping the shared studies). The first list's picture naming
# (179), speech perception (97) and speech production (107) were below the rule.
SINGLE = [
    ("pictures", "Picture naming", "term", ["picture", "pictures"]),                       # 749
    ("objects", "Picture naming", "term", ["object", "objects"]),                          # 1160
    ("listening", "Auditory definition", "term", ["listening", "listened", "spoken"]),     # 496
    ("auditory", "Auditory definition", "term", ["auditory", "auditory cortex"]),          # 1252
    ("sentences", "Sentence completion", "term", ["sentence", "sentences"]),               # 478
    ("reading", "Sentence completion", "term", ["reading", "read"]),                       # 615
    ("naming", "Shared core", "term", ["naming", "fluency", "covert"]),                    # 421
    ("semantic", "Shared core", "term", ["semantic"]),                                     # 1031
    ("phonological", "Shared core", "term", ["phonological"]),                             # 377
    ("production", "Shared core", "term", ["production", "speech production", "overt"]),   # 457
]
CONTRASTS = [
    ("naming_vs_objects", "Picture naming", "contrast",                                    # 344 vs 1083
     (["naming", "fluency", "covert"], ["object", "objects"])),
    ("listening_vs_auditory", "Auditory definition", "contrast",                           # 254 vs 1010
     (["listening", "listened", "spoken"], ["auditory", "auditory cortex"])),
    ("sentences_vs_words", "Sentence completion", "contrast",                              # 315 vs 1259
     (["sentence", "sentences"], ["word", "words"])),
    ("naming_vs_production", "Shared core", "contrast",                                    # 296 vs 332
     (["naming", "fluency", "covert"], ["production", "speech production", "overt"])),
    ("semantic_vs_memory_retrieval", "Shared core", "contrast",                            # 967 vs 265
     (["semantic"], ["memory retrieval", "encoding retrieval"])),
    ("production_vs_motor", "Shared core", "contrast",                                     # 390 vs 628
     (["production", "speech production", "overt"], ["motor control", "motor cortex", "primary motor"])),
    # added 2026-11-05: language studies vs resting-state studies (~1030 vs ~1350 after dropping the ~70
    # shared). NOT task vs rest inside one study: resting-state abstracts mostly report connectivity peaks
    # (default mode and others), so this map is "where language studies report foci more than resting-state
    # studies do"
    ("language_vs_resting_state", "Language network", "contrast",
     (["language", "languages", "language network"], ["resting state", "resting"])),
]
# added 2026-11-05 (curiosity): one term per task stage vs the same resting-state studies, built from SINGLE's
# terms so the two always agree; the same caveat as language_vs_resting_state
REST = ["resting state", "resting"]
CONTRASTS += [(f"{n}_vs_resting_state", st, "contrast", (terms, REST))
              for n, st, _, terms in SINGLE
              if n in ("pictures", "listening", "sentences", "naming", "semantic", "production")]

ap = argparse.ArgumentParser()
ap.add_argument("--counts", action="store_true", help="only count studies per term")
ap.add_argument("--kernel-r", type=float, default=6.0, help="MKDA sphere radius in mm (NeuroSynth: 6)")
ap.add_argument("--only", nargs="+", default=None,
                help="fit only these maps; the other rows of maps.csv are kept (e.g. --only language_vs_resting_state)")
args = ap.parse_args()

from nimare.extract import fetch_neurosynth                          # noqa: E402
from nimare.io import convert_neurosynth_to_dataset                  # noqa: E402
from nimare.meta.cbma.mkda import MKDAChi2                           # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
pkl = DATA / "neurosynth_dataset.pkl.gz"
if pkl.exists():
    from nimare.dataset import Dataset
    dset = Dataset.load(str(pkl))
else:
    files = fetch_neurosynth(data_dir=str(DATA), version="7", overwrite=False, source="abstract", vocab="terms")
    db = files[0]
    dset = convert_neurosynth_to_dataset(coordinates_file=db["coordinates"], metadata_file=db["metadata"],
                                         annotations_files=db["features"])
    dset.save(str(pkl))
labels = {c.split("__", 1)[1]: c for c in dset.annotations.columns if "__" in c}


def studies(terms):
    """The union of the studies labelled with any of `terms`, and the terms not in the vocabulary."""
    ids, absent = set(), []
    for term in terms:
        if term in labels:
            ids |= set(dset.get_studies_by_label(labels=[labels[term]], label_threshold=0.001))
        else:
            absent.append(term)
    return ids, absent


rows = []
for name, stage, kind, terms in SINGLE:
    ids, absent = studies(terms)
    rows.append(dict(name=name, stage=stage, kind=kind, terms=" + ".join(terms), n_studies=len(ids),
                     n_other=None, absent=", ".join(absent), used=len(ids) >= MIN_STUDIES))
for name, stage, kind, (tx, ty) in CONTRASTS:
    ix, ax_ = studies(tx)
    iy, ay = studies(ty)
    both = ix & iy
    nx, ny = len(ix - both), len(iy - both)
    rows.append(dict(name=name, stage=stage, kind=kind, terms=" + ".join(tx) + "  VS  " + " + ".join(ty),
                     n_studies=nx, n_other=ny, absent=", ".join(ax_ + ay), used=min(nx, ny) >= MIN_STUDIES))
counts = pd.DataFrame(rows)
counts.to_csv(OUT / "study_counts.csv", index=False)
pd.set_option("display.width", 220)
print(counts[["name", "kind", "n_studies", "n_other", "used", "absent"]].to_string(index=False))
if args.counts:
    raise SystemExit(0)


def association_map(res):
    for key in ("z_desc-association", "z_desc-specificity"):         # new / old NiMARE names
        try:
            return res.get_map(key)
        except (ValueError, KeyError):
            continue
    raise SystemExit(f"no association map among {list(res.maps)}")


done = []
todo = counts[counts.used]
if args.only:
    unknown = sorted(set(args.only) - set(counts.name))
    if unknown:
        raise SystemExit(f"--only: unknown map(s) {unknown}")
    todo = todo[todo.name.isin(args.only)]
    if todo.empty:
        raise SystemExit("--only: none of these maps passes the 200-study rule")
for r in todo.itertuples():
    if r.kind == "term":
        terms = next(t for n, _, _, t in SINGLE if n == r.name)
        ids_in, _ = studies(terms)
        ids_out = sorted(set(dset.ids) - ids_in)
        d1, d2 = dset.slice(sorted(ids_in)), dset.slice(ids_out)
    else:
        tx, ty = next(t for n, _, _, t in CONTRASTS if n == r.name)
        ix, _ = studies(tx)
        iy, _ = studies(ty)
        both = ix & iy
        d1, d2 = dset.slice(sorted(ix - both)), dset.slice(sorted(iy - both))
    print(f"  {r.name}: {len(d1.ids)} vs {len(d2.ids)} studies ...", flush=True)
    res = MKDAChi2(kernel__r=args.kernel_r).fit(d1, d2)
    img = association_map(res)
    d = OUT / r.name
    d.mkdir(exist_ok=True)
    img.to_filename(str(d / f"{r.name}_association-test_z.nii.gz"))
    done.append(dict(name=r.name, stage=r.stage, kind=r.kind, n_studies=r.n_studies, n_other=r.n_other,
                     terms=r.terms))
done = pd.DataFrame(done)
if args.only and (OUT / "maps.csv").exists():                        # keep the maps fitted before
    old = pd.read_csv(OUT / "maps.csv")
    done = pd.concat([old[~old.name.isin(done.name)], done], ignore_index=True)
done.to_csv(OUT / "maps.csv", index=False)
(OUT / "params.json").write_text(json.dumps(dict(min_studies=MIN_STUDIES, kernel_r=args.kernel_r,
                                                 label_threshold=0.001, database="neurosynth v7 abstract terms"),
                                            indent=1))
print(f"wrote {len(done)} maps to {OUT}")
