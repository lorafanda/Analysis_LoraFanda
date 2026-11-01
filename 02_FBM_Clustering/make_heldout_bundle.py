#!/usr/bin/env python3
"""make_heldout_bundle.py - the held-out placement of every run, in the cluster visualizer's
own contact space, so cluster_visualizer.html can show it on the brain.

    python make_heldout_bundle.py            # plan: which runs have tables, the join and label checks
    python make_heldout_bundle.py --write    # write coverage_viz/runs/<run>/heldout.json + heldout_index.json

WHAT IT READS, per run of the coverage manifest (outputs/250_recon/fsaverage/coverage_viz/)
  outputs/clustering/mmd/<cache>/<feature set>/<method>/   the MMD script's tables at the paper's K:
      confusion_lopo_*.csv, confusion_halves_*.csv (2026-10-30), per_type.csv, pairs_summary.csv, manifest
  outputs/clustering/consensus/<cache>/<feature set>/<method>/items_k<K>.csv and M_k<K>.npy
      the patient-resampling sweep: per electrode its own and best-other consensus; per type the
      mean of M over the pairs inside it (Lora's definition, 2026-10-30)
  the run itself through 00_Paper2_Figures.load_run (labels, loadings, patients, the join keys)

WHAT IT WRITES  runs/<run dir>/heldout.json
  contacts  idx (global contact indices of contacts.json), type (the paper's partition), lopo_hit,
            lopo_goes, halves_hit, halves_goes, loading (convex NMF), cons_own, cons_other
  types     per cluster: n, n_patients, recall / precision / confused_with (lopo and halves), IQR of
            the recall over folds, consensus m(k), silhouette z, name
  lopo, halves   the K x K count matrices and the overall accuracy
  mmd       the 21 pair distances on held-out patients, the cloud's sector value
  params, order (figure order at K), k, cache, run, written
and heldout_index.json at the bundle root: run id -> file.

CHECKS (printed; the plan stops on a failure)
  - every electrode of the tables joins exactly one contact of the run's cohort
  - the table's type equals the bundle's sweep labels at K (labels_by_k.json) for every contact;
    the agreement with the PUBLISHED labels (contacts.json runs) is reported, not required: the
    published cut is the 300-iteration fit, the paper's partition is the sweep's
  - the main cell's overall accuracies and per-type recalls equal the validation story's numbers.json
Nothing else is written; push with scripts/push_cluster_bundle.py (it picks up new files under coverage_viz).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("p2fig1", ROOT / "00_Paper2_Figures.py")
P2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P2)
sys.path.insert(0, str(ROOT / "functions"))
import lf_runs as LR                                      # noqa: E402

COV = ROOT / "outputs" / "250_recon" / "fsaverage" / "coverage_viz"
MANIFEST = COV / "manifest.json"
SITE_ORDER = Path.home() / "lorafanda.github.io" / "cluster_visualizer_order.json"
STORY_NUMBERS = ROOT / "outputs" / "clustering" / "paper2_validation" / "numbers.json"
K_BY_FSET = {"concat_bands15": 7, "concat_bands5": 8, "concat_bands2": 10, "concat_hfa": 6}


def norm(s) -> str:
    return "" if s is None else str(s).replace("_", "").replace("-", "").upper()


def r3(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write the json files (default: plan and check)")
    a = ap.parse_args()
    t0 = time.time()
    cache = LR.current_cache()
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    contacts = json.loads((COV / man["contacts_file"]).read_text(encoding="utf-8"))
    order_json = json.loads(SITE_ORDER.read_text(encoding="utf-8"))["runs"] if SITE_ORDER.exists() else {}
    story = json.loads(STORY_NUMBERS.read_text(encoding="utf-8"))["values"] if STORY_NUMBERS.exists() else None

    # the contact space: the coordinate table without NaN rows, in file order (make_coverage_bundle)
    co = pd.read_csv(P2.COORDS).dropna(subset=["x", "y", "z"]).reset_index(drop=True)
    if len(co) != len(contacts["xyz"]):
        raise SystemExit(f"coordinate table has {len(co)} rows, contacts.json {len(contacts['xyz'])}: rebuild the bundle first")
    xyz_c = np.asarray(contacts["xyz"], float)
    if not np.allclose(np.round(co[["x", "y", "z"]].to_numpy(float), 1), xyz_c, atol=0.06):
        raise SystemExit("coordinate table and contacts.json disagree on the contact order")
    key2idx = {}
    for i, (p, n) in enumerate(zip(co["patient"], co["name"])):
        key2idx.setdefault(f"{p}|{norm(n)}", i)
    print(f"cache {cache}   {len(co)} contacts   {len(man['runs'])} runs in the manifest")

    index, failures = {}, []
    for r in man["runs"]:
        method, fset = r["id"].split("__")[0], r["id"].split("__")[1]
        K = K_BY_FSET.get(fset)
        cell = ROOT / "outputs" / "clustering" / "mmd" / cache / fset / method
        if K is None or not (cell / "confusion_lopo_electrodes.csv").exists():
            print(f"  -  {r['id']:48s} no held-out tables yet ({cell.relative_to(ROOT) if K else 'no K'})")
            continue
        cm = json.loads((cell / "manifest.json").read_text(encoding="utf-8"))
        if int(cm["k"]) != K:
            print(f"  !! {r['id']}: the MMD cell is at K {cm['k']}, the paper's K is {K}; skipped")
            continue
        d = P2.load_run(fset, K, method)
        d["fset"] = fset
        meta = d["meta"]
        keys = [f"{p}|{norm(e)}" for p, e in zip(meta.patient_id, meta.electrode)]
        idx = np.array([key2idx.get(k, -1) for k in keys])
        cohort = set(contacts["cohorts"][r["cohort_id"]])
        bad = [(k, i) for k, i in zip(keys, idx) if i < 0 or i not in cohort]
        if bad:
            failures.append(f"{r['id']}: {len(bad)} electrodes do not join a cohort contact, e.g. {bad[:3]}")
            continue
        # the tables, joined by key (not by row order)
        lo = pd.read_csv(cell / "confusion_lopo_electrodes.csv")
        ha = pd.read_csv(cell / "confusion_halves_electrodes.csv")
        for t in (lo, ha):
            t["key"] = [f"{p}|{norm(e)}" for p, e in zip(t.patient, t.electrode)]
        lo = lo.set_index("key").reindex(keys)
        ha = ha.set_index("key").reindex(keys)
        if lo.type.isna().any() or ha.type.isna().any():
            failures.append(f"{r['id']}: the confusion tables do not cover every electrode of the run")
            continue
        typ = lo.type.to_numpy(int)
        if not np.array_equal(typ, d["lab"]):
            failures.append(f"{r['id']}: the table's types differ from the run's labels at K {K}")
            continue
        # label checks against the bundle
        sweep = json.loads((COV / r["dir"] / "labels_by_k.json").read_text(encoding="utf-8")).get(str(K))
        agree_sweep = (float(np.mean([bool(sweep[i]) and int(sweep[i][0]) == int(t) for i, t in zip(idx, typ)]))
                       if sweep else float("nan"))
        pub = contacts["runs"].get(r["id"])
        agree_pub = (float(np.mean([bool(pub[i]) and int(pub[i][0]) == int(t) for i, t in zip(idx, typ)]))
                     if (pub and int(r["k"]) == K) else None)
        if sweep and agree_sweep < 0.999:
            failures.append(f"{r['id']}: the table's types agree with the sweep labels at K {K} on only {100*agree_sweep:.1f} %")
            continue
        # consensus: per electrode and per type (Lora's definition: mean M over pairs inside the type)
        cdir = ROOT / "outputs" / "clustering" / "consensus" / cache / fset / method
        cons_own = cons_other = None
        cons_type = {}
        it_f, M_f = cdir / f"items_k{K:02d}.csv", cdir / f"M_k{K:02d}.npy"
        if it_f.exists():
            it = pd.read_csv(it_f)
            it["key"] = [f"{p}|{norm(e)}" for p, e in zip(it.patient_id, it.electrode)]
            it = it.set_index("key").reindex(keys)
            cons_own = it.own_consensus.to_numpy(float)
            cons_other = it.best_other_consensus.to_numpy(float)
        if M_f.exists():
            Mc = np.load(M_f).astype(float)
            if Mc.shape[0] == len(typ):
                for j in range(K):
                    m = typ == j
                    sub = Mc[np.ix_(m, m)]
                    n_ = int(m.sum())
                    cons_type[j] = float((sub.sum() - np.trace(sub)) / (n_ * (n_ - 1))) if n_ > 1 else None
        # per type
        pt = pd.read_csv(cell / "per_type.csv")
        pt = pt[pt.version == "argmax"].set_index("cluster")
        lt = pd.read_csv(cell / "confusion_lopo_per_type.csv").set_index("cluster")
        ht = pd.read_csv(cell / "confusion_halves_per_type.csv").set_index("cluster")
        CL = pd.read_csv(cell / "confusion_lopo_counts.csv", index_col=0).to_numpy(float)
        CH = pd.read_csv(cell / "confusion_halves_counts.csv", index_col=0).to_numpy(float)
        own = d["Gn"].max(1) if not d["hard"] else None
        types = []
        for j in range(K):
            m = typ == j
            types.append(dict(
                cluster=j, name=P2.type_name(fset, K, j), n=int(m.sum()), n_patients=int(pd.Series(d["patient"][m]).nunique()),
                lopo_recall=r3(lt.loc[j, "recall"]), lopo_q25=r3(lt.loc[j, "recall_q25"]), lopo_q75=r3(lt.loc[j, "recall_q75"]),
                lopo_precision=r3(lt.loc[j, "precision"]), lopo_confused_with=int(lt.loc[j, "confused_with"]), lopo_confused_share=r3(lt.loc[j, "confused_share"]),
                halves_recall=r3(ht.loc[j, "recall"]), halves_q25=r3(ht.loc[j, "recall_q25"]), halves_q75=r3(ht.loc[j, "recall_q75"]),
                halves_precision=r3(ht.loc[j, "precision"]), halves_confused_with=int(ht.loc[j, "confused_with"]), halves_confused_share=r3(ht.loc[j, "confused_share"]),
                consensus=r3(cons_type.get(j)), sil_z=r3(pt.loc[j, "sil_z"]) if j in pt.index else None,
                sil=r3(pt.loc[j, "silhouette"]) if j in pt.index else None))
        ps = pd.read_csv(cell / "pairs_summary.csv")
        ps = ps[ps.version == "argmax"]
        mmd = dict(cloud=r3(ps.cloud_median.iloc[0]), bandwidth=r3(cm["bandwidth"]),
                   pairs=[dict(i=int(x.i), j=int(x.j), mmd2=r3(x.mmd2_ho), ratio=r3(x.ratio), share_p=r3(x.share_p_ho_lt05)) for _, x in ps.iterrows()])
        order = [int(c) for c in (order_json.get(r["id"], {}).get(str(K)) or range(K))]
        out = dict(
            run=r["id"], run_dir=str(d["run"].name), feature_set=fset, method=method, k=K, cache=cache,
            n=int(len(typ)), n_patients=int(d["n_patients"]), written=time.strftime("%Y-%m-%d %H:%M"),
            params=dict(n_lopo_seeds=cm.get("n_lopo_seeds"), n_splits=cm["n_splits"], n_perm=cm["n_perm"],
                        n_perm_heldout=cm["n_perm_heldout"], n_cloud=cm["n_cloud"]),
            order=order, types=types,
            lopo=dict(matrix=CL.astype(int).tolist(), overall=r3(np.trace(CL) / CL.sum())),
            halves=dict(matrix=CH.astype(int).tolist(), overall=r3(np.trace(CH) / CH.sum())),
            mmd=mmd,
            label_agreement=dict(sweep=r3(agree_sweep), published=r3(agree_pub) if agree_pub is not None else None),
            contacts=dict(
                idx=[int(i) for i in idx], type=[int(t) for t in typ],
                lopo_hit=[r3(v) for v in lo.hit_rate], lopo_goes=[int(v) for v in lo.goes_to],
                halves_hit=[r3(v) for v in ha.hit_rate], halves_goes=[int(v) for v in ha.goes_to],
                loading=([r3(v) for v in own] if own is not None else None),
                cons_own=([r3(v) for v in cons_own] if cons_own is not None else None),
                cons_other=([r3(v) for v in cons_other] if cons_other is not None else None)))
        # the story check, on the main cell
        note = ""
        if story and fset == "concat_bands15" and method == "cnmf" and story.get("conf", {}).get("lopo"):
            sl, sh = story["conf"]["lopo"], story["conf"]["halves"]
            ok = (abs(sl["overall"] - out["lopo"]["overall"]) < 5e-4 and abs(sh["overall"] - out["halves"]["overall"]) < 5e-4
                  and all(abs(sl["recall"][str(t["cluster"])] - t["lopo_recall"]) < 5e-4 for t in types))
            note = "   story numbers.json: " + ("MATCH" if ok else "MISMATCH (rebuild the story: make_validation_story.py)")
            if not ok:
                failures.append(f"{r['id']}: the held-out numbers differ from the validation story's")
        rec = ", ".join(f"c{t['cluster']} {t['lopo_recall']:.2f}" for t in sorted(types, key=lambda t: -t["lopo_recall"]))
        print(f"  ok {r['id']:48s} K {K}  lopo {100*out['lopo']['overall']:.0f} %  halves {100*out['halves']['overall']:.0f} %  "
              f"sweep-label agreement {100*agree_sweep:.1f} %" + (f", published {100*agree_pub:.1f} %" if agree_pub is not None else "")
              + f"   [{rec}]{note}")
        if a.write:
            p = COV / r["dir"] / "heldout.json"
            tmp = Path(str(p) + ".tmp")
            tmp.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
            if json.loads(tmp.read_text(encoding="utf-8"))["n"] != out["n"]:
                raise SystemExit(f"write check failed for {p}")
            tmp.replace(p)
        index[r["id"]] = dict(file=f"{r['dir']}/heldout.json", k=K, feature_set=fset, method=method,
                              lopo_overall=out["lopo"]["overall"], halves_overall=out["halves"]["overall"], written=out["written"])
    if failures:
        print("\nFAILED:", *failures, sep="\n  ")
        return 1
    if a.write:
        (COV / "heldout_index.json").write_text(json.dumps(dict(generated=time.strftime("%Y-%m-%d %H:%M"), cache=cache, runs=index), indent=1),
                                                encoding="utf-8")
        print(f"\n{len(index)} heldout.json written + heldout_index.json   ({time.time()-t0:.0f} s)")
    else:
        print(f"\nplan only: {len(index)} run(s) would be written; add --write   ({time.time()-t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
