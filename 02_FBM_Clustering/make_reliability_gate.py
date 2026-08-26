#!/usr/bin/env python3
"""
make_reliability_gate.py - replace the amplitude gate with a reliability gate.

THE PROBLEM. The responsiveness gate thresholds on AMPLITUDE (`n_high_activity`),
which cannot separate a small-but-repeatable response from a large-but-random one.
It keeps 1267 of 2946 contacts. FIG C.10 showed the 1679 it removes are not a
different response type - they carry the same feature signature as the contacts they
would have joined (r = 0.75-0.95), only weaker - and FIG C.9 showed that including
them lets k-means and Ward build a cluster that is 74-78% made of them. So the gate
is removing the right kind of thing for the wrong reason, and lifting it is not the
fix either.

THE REPLACEMENT, and it is what both reference papers do rather than something new:

  * Norman-Haignere et al. (2019), Results "Electrode decomposition": their 271
    electrodes are the survivors of SPLIT-HALF r > 0.2. They reuse the same quantity
    to noise-correct variance explained (Fig 1E), so it buys two things at once.
  * Castellucci et al. (2026), STAR Methods "Detection of significant neural
    responses": significance against a shuffled-alignment null plus a duration
    criterion, accepting activation OR suppression. Keeps 72.6% of contacts.

Both select on REPRODUCIBILITY. Neither selects on amplitude.

WHAT THIS SCRIPT DOES. Reads the odd/even half-cubes written by stage 01, correlates
them per contact-condition, applies the Spearman-Brown correction, and compares the
resulting gate against the amplitude gate contact by contact. The interesting number
is not how many survive - it is the DISAGREEMENT: contacts the amplitude gate keeps
that do not reproduce, and contacts it throws away that do.

SPEARMAN-BROWN. A correlation between two half-length measurements underestimates the
reliability of the full-length one. r_full = 2r / (1 + r) corrects for that. NH's 0.2
threshold is on the raw split-half correlation, so both are reported and the gate is
applied on the raw value to match them.

BAND. Reliability is computed over the whole time-frequency cube AND over the
high-gamma band alone, and the whole-cube version is the one that matches the gate.
An earlier version of this note said the amplitude gate is HG-based. It is not:
prepare_dataset counts bins over threshold across the FULL 0-400 Hz cube (129 x 300 =
38,700 bins), so a whole-cube reliability is the like-for-like comparison and the
HG-band version is the secondary one. The HG band is what the clustering FEATURES use,
which is a different thing from what the gate measures.

    python make_reliability_gate.py
    python make_reliability_gate.py --r-min 0.2 --band hg
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "functions"))

ERSP = ROOT.parent / "01_FBM_Analysis" / "outputs" / "04_ersp_LM_RAWONLY"
CLUST = ROOT / "outputs" / "clustering"
OUT = CLUST / "reliability_gate"

# house palette, same as the gate-split and component-anatomy figures
GATED_COL, ADDED_COL = "#4a6fa5", "#b0b7be"
INK, MUTED = "#1b232c", "#68727d"
RED, GREEN = "#c1121f", "#1b7837"
HG_BAND = (70.0, 150.0)
N_FREQ_DEFAULT = 129
FMAX_DEFAULT = 200.0          # cube covers 0..fmax over N_FREQ rows


def norm_contact(s):
    """Same normalisation the pipeline uses (lf_io_utils.normalize_label).

    Inlined rather than imported because lf_io_utils pulls in mne, which this script
    does not otherwise need. The ERSP filename carries the RAW channel name (A_L10)
    while labels.csv carries the normalised one (AL10); joining without this matched
    21 contacts out of 838. With it, all 2946 rows of the ungated cohort match.
    """
    return str(s).replace("_", "").replace("-", "").upper()


def spearman_brown(r):
    return 2.0 * r / (1.0 + r) if r > -1 else np.nan


def half_pairs():
    """[(patient, condition, contact, half1_path, half2_path)] found on disk."""
    out = []
    # ERSP_halves, not ERSP_matrix: the halves were moved out so that the
    # cube consumers, which glob ERSP_matrix/<cond>/*.npy, stop ingesting
    # them as if each half were a separate electrode.
    for p1 in sorted(ERSP.glob("*/LM/ERSP_halves/*/*_half1.npy")):
        p2 = Path(str(p1)[: -len("_half1.npy")] + "_half2.npy")
        if not p2.exists():
            continue
        pid = p1.parents[3].name
        cond = p1.parent.name
        # <pid>_<cond>_<reref>_ERSP_<contact>_TN_half1.npy
        m = re.match(rf"^{re.escape(pid)}_{re.escape(cond)}_.+?_ERSP_(.+?)_TN_half1\.npy$",
                     p1.name)
        contact = m.group(1) if m else p1.name
        out.append((pid, cond, contact, p1, p2))
    return out


def band_mask(n_freq, band, fmax=FMAX_DEFAULT):
    """Row mask for a frequency band, assuming rows span 0..fmax linearly.

    Stage 01 does not write the frequency vector beside the cube, so this is
    reconstructed from its length. If a cube ever ships with a real `f` axis, read
    that instead - the assumption is stated here rather than buried.
    """
    f = np.linspace(0.0, fmax, n_freq)
    return (f >= band[0]) & (f <= band[1])


def reliability(pairs, band=None, progress=True):
    rows = []
    for i, (pid, cond, contact, p1, p2) in enumerate(pairs):
        A, B = np.load(p1), np.load(p2)
        if A.shape != B.shape:
            continue
        if band is not None:
            m = band_mask(A.shape[0], band)
            A, B = A[m], B[m]
        a, b = A.ravel(), B.ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 20 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
            continue
        r = float(np.corrcoef(a[ok], b[ok])[0, 1])
        rows.append(dict(patient_id=pid, condition=cond, contact=contact,
                         split_half_r=r, spearman_brown=spearman_brown(r),
                         n_finite=int(ok.sum())))
        if progress and (i + 1) % 500 == 0:
            print(f"    {i + 1}/{len(pairs)}", end="\r")
    if progress:
        print(" " * 40, end="\r")
    return pd.DataFrame(rows)


def amplitude_gate():
    """(patient_id, contact, kept_by_amplitude) from the UNGATED run's labels.

    concat_hg_all is the only track carrying every contact with its gate flag, so it
    is the one place the two gates can be compared on the same rows.
    """
    idx = json.loads((CLUST / "index.json").read_text(encoding="utf-8"))
    runs = idx["runs"] if isinstance(idx, dict) else idx
    best = None
    for r in runs:
        if r["feature_set"] != "concat_hg_all":
            continue
        d = CLUST / r["method"] / r["feature_set"] / "runs" / r["run_id"]
        if (d / "labels.csv").exists() and (best is None or r["run_id"] > best[0]):
            best = (r["run_id"], d)
    if best is None:
        return None
    lab = pd.read_csv(best[1] / "labels.csv")
    col = "contact_norm" if "contact_norm" in lab.columns else "electrode"
    g = pd.to_numeric(lab["n_high_activity"], errors="coerce").fillna(0) > 0
    return pd.DataFrame(dict(patient_id=lab["patient_id"].astype(str),
                             ckey=lab[col].astype(str).map(norm_contact),
                             amp_gate=g.to_numpy()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r-min", type=float, default=0.2, dest="r_min",
                    help="Norman-Haignere threshold on the raw split-half r")
    ap.add_argument("--band", choices=["all", "hg"], default="all")
    a = ap.parse_args()

    pairs = half_pairs()
    print(f"  split-half cube pairs on disk: {len(pairs)}")
    if not pairs:
        print()
        print("  BLOCKED - stage 01 has not written the halves yet.")
        print("  lf_ersp.compute_ersp now returns avg_db_h1 / avg_db_h2 and 140/150")
        print("  save them as <stem>_half1.npy / _half2.npy, so this only needs")
        print("  140 (or 150) re-run. Nothing else in the pipeline changed:")
        print("  avg_db is bit-identical to before, verified against the old code.")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    print("  computing reliability over the whole cube ...")
    rel_all = reliability(pairs, band=None)
    print("  computing reliability over the high-gamma band ...")
    rel_hg = reliability(pairs, band=HG_BAND)
    rel_all["band"], rel_hg["band"] = "all", "hg"
    per_cond = pd.concat([rel_all, rel_hg], ignore_index=True)
    per_cond.to_csv(OUT / "reliability_per_contact_condition.csv", index=False)

    use = rel_hg if a.band == "hg" else rel_all
    # one number per contact: the best condition, matching the amplitude gate's
    # "responsive in at least one condition" logic
    use = use.copy()
    use["ckey"] = use["contact"].map(norm_contact)
    per_contact = (use.groupby(["patient_id", "ckey"])
                   .agg(r=("split_half_r", "max"),
                        contact=("contact", "first")).reset_index())
    per_contact["rel_gate"] = per_contact["r"] > a.r_min

    amp = amplitude_gate()
    if amp is not None:
        per_contact = per_contact.merge(amp, on=["patient_id", "ckey"], how="left")
    else:
        per_contact["amp_gate"] = np.nan
    # A left merge leaves this as OBJECT dtype wherever a contact did not match, and
    # `~` on an object column is bitwise-not over Python bools - it returns -1 and -2
    # rather than negating. The nullable boolean dtype negates correctly and keeps NA
    # as NA, which is what the unmatched rows are.
    per_contact["amp_gate"] = per_contact["amp_gate"].astype("boolean")
    per_contact.to_csv(OUT / "reliability_per_contact.csv", index=False)

    n = len(per_contact)
    matched = per_contact["amp_gate"].notna()
    print()
    print(f"  {n} contacts with a reliability estimate "
          f"({int(matched.sum())} matched to the amplitude gate)")
    print(f"  reliability gate (r > {a.r_min}, {a.band} band): "
          f"{int(per_contact['rel_gate'].sum())} kept")
    stats = dict(n_contacts=int(n), r_min=a.r_min, band=a.band,
                 n_pairs=len(pairs),
                 n_rel_keep=int(per_contact["rel_gate"].sum()))
    if matched.any():
        m = per_contact[matched].copy()
        m["amp_gate"] = m["amp_gate"].astype(bool)
        both = int((m["amp_gate"] & m["rel_gate"]).sum())
        amp_only = int((m["amp_gate"] & ~m["rel_gate"]).sum())
        rel_only = int((~m["amp_gate"] & m["rel_gate"]).sum())
        neither = int((~m["amp_gate"] & ~m["rel_gate"]).sum())
        stats.update(both=both, amp_only=amp_only, rel_only=rel_only,
                     neither=neither, n_matched=int(matched.sum()))
        print()
        print("             reliable   not reliable")
        print(f"  amp keep   {both:>8}   {amp_only:>12}")
        print(f"  amp drop   {rel_only:>8}   {neither:>12}")
        print()
        print(f"  the two gates disagree about {amp_only + rel_only} contacts:")
        print(f"    {amp_only} loud but not reproducible - kept today, should not be")
        print(f"    {rel_only} quiet but reproducible   - discarded today, should not be")
    (OUT / "reliability_gate_stats.json").write_text(json.dumps(stats, indent=2),
                                                     encoding="utf-8")
    draw(per_contact, per_cond, stats, a, OUT / "R1_reliability_gate.png")
    print(f"\n  -> {OUT / 'R1_reliability_gate.png'}")
    return 0


# ==============================================================================
def draw(pc, per_cond, stats, a, out_png):
    matched = pc["amp_gate"].notna()
    m = pc[matched].copy()
    m["amp_gate"] = m["amp_gate"].astype(bool)   # no NA left in this subset
    fig = plt.figure(figsize=(13.0, 8.8), dpi=200)
    gs = GridSpec(1, 3, width_ratios=[1.5, 1.0, 1.2], wspace=0.28,
                  left=0.052, right=0.985, top=0.655, bottom=0.090)

    # ---- A: the distribution, split by what the amplitude gate did with it
    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(-0.4, 1.0, 60)
    if matched.any():
        ax.hist([m.loc[m["amp_gate"], "r"], m.loc[~m["amp_gate"], "r"]], bins=bins,
                stacked=True, color=[GATED_COL, ADDED_COL], lw=0,
                label=["kept by the amplitude gate", "removed by it"])
    else:
        ax.hist(pc["r"], bins=bins, color=GATED_COL, lw=0)
    ax.axvline(a.r_min, color=RED, ls="--", lw=1.2)
    ax.text(a.r_min + 0.015, ax.get_ylim()[1] * 0.55,
            f"r > {a.r_min}\nNorman-Haignere", color=RED, fontsize=7.6,
            va="center")
    ax.set_xlabel("split-half reliability r", fontsize=9)
    ax.set_ylabel("contacts", fontsize=9)
    ax.legend(fontsize=7.6, frameon=False, loc="upper left")
    ax.set_title("A · what the amplitude gate keeps, against reliability",
                 fontsize=9.5, loc="left", color=INK, pad=6)

    # ---- B: the 2x2, which is the whole point
    axb = fig.add_subplot(gs[0, 1])
    if "both" in stats:
        M = np.array([[stats["both"], stats["amp_only"]],
                      [stats["rel_only"], stats["neither"]]], float)
        im = axb.imshow(M, cmap="Blues", vmin=0, vmax=M.max())
        for i in range(2):
            for j in range(2):
                agree = (i == j)
                axb.text(j, i, f"{int(M[i, j])}", ha="center", va="center",
                         fontsize=15 if not agree else 13,
                         fontweight="bold" if not agree else "normal",
                         color="white" if M[i, j] > 0.55 * M.max() else INK)
        axb.set_xticks([0, 1]); axb.set_xticklabels(["reliable", "not"], fontsize=8.5)
        axb.set_yticks([0, 1]); axb.set_yticklabels(["amp keeps", "amp drops"],
                                                    fontsize=8.5)
        axb.tick_params(length=0)
        for sp in axb.spines.values():
            sp.set_color("#ccd2d8")
        axb.set_title("B · where the two gates disagree", fontsize=9.5, loc="left",
                      color=INK, pad=6)
        axb.set_xlabel(f"off-diagonal = {stats['amp_only'] + stats['rel_only']} "
                       f"contacts decided differently", fontsize=7.8, color=MUTED,
                       labelpad=6)
    else:
        axb.axis("off")

    # ---- C: reliability by condition, so a single bad condition is visible
    axc = fig.add_subplot(gs[0, 2])
    sub = per_cond[per_cond["band"] == a.band]
    conds = sorted(sub["condition"].unique())
    data = [sub.loc[sub["condition"] == c, "split_half_r"].to_numpy() for c in conds]
    if data:
        bp = axc.boxplot(data, patch_artist=True, widths=0.55,
                         medianprops=dict(color=INK, lw=1.2), showfliers=False)
        axc.set_xticks(range(1, len(conds) + 1))
        axc.set_xticklabels(conds, fontsize=8)
        for patch in bp["boxes"]:
            patch.set_facecolor(GATED_COL); patch.set_alpha(0.55); patch.set_lw(0)
    axc.axhline(a.r_min, color=RED, ls="--", lw=1.2)
    axc.set_ylabel("split-half reliability r", fontsize=9)
    axc.tick_params(labelsize=8)
    axc.set_title("C · by condition", fontsize=9.5, loc="left", color=INK, pad=6)

    for x_ax in (ax, axc):
        x_ax.spines[["top", "right"]].set_visible(False)
        x_ax.tick_params(labelsize=8, colors=MUTED)

    fig.suptitle("Replacing the amplitude gate with a reliability gate",
                 x=0.052, y=0.972, ha="left", fontsize=15, color=INK)
    para = [
        f"Each contact's ERSP is averaged twice from disjoint halves of its trials "
        f"(odd trials vs even trials, so drift and fatigue are balanced across the "
        f"halves rather than confounded with them) and the two are correlated. "
        f"{stats['n_pairs']} cube pairs, {stats['n_contacts']} contacts.",
        f"This is the criterion both reference papers use. Norman-Haignere et al. "
        f"(2019) keep electrodes at split-half r > 0.2; Castellucci et al. (2026) use "
        f"significance against a shuffled null. Neither selects on amplitude, which is "
        f"what this project has been doing.",
        f"Gate applied on the raw r at {a.r_min} over the "
        f"{'high-gamma band' if a.band == 'hg' else 'whole time-frequency cube'}, "
        f"taking each contact's best condition - the same 'responsive in at least one "
        f"condition' logic the amplitude gate uses.",
    ]
    if "both" in stats:
        para.append(
            f"THE RESULT IS THE DISAGREEMENT, panel B: {stats['amp_only']} contacts are "
            f"loud enough to keep but do not reproduce, and {stats['rel_only']} are "
            f"quiet enough to discard but do. Those two numbers are what an amplitude "
            f"threshold cannot see.")
    fig.text(0.052, 0.935, "\n".join(textwrap.fill(t, width=158) for t in para),
             fontsize=8.4, color=MUTED, va="top", linespacing=1.58)

    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
