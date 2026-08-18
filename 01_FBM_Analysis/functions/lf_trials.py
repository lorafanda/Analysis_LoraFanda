from __future__ import annotations
import os, glob, hashlib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def collect_trials(
    prep_dir,
    fs_hz,
    p_low=-3.0, p_high=3.0,
    report_path=None,
    patient_id=None,
    *,
    outlier_method="IQR",
    iqr_k=1.5,
    min_stim_s=0.5,
    min_post_s=1.0,
    max_post_s=5.0,
    condition_aliases=None,
    z_low=None, z_high=None,
    pct_low=None, pct_high=None,
    keep_resp_accuracy=("correct","valid","1"),   # case-insensitive; None to keep all
):
    """
    Collect trial onset/offset/end indices from *.tsv files in `prep_dir`, normalize condition
    names, apply basic validity checks, and optionally trim outliers based on post-stimulus
    durations (trial_end - offset) per condition.

    Parameters
    ----------
    prep_dir : str
        Directory containing preprocessed *.tsv files.
    fs_hz : float
        Sampling rate in Hz.
    min_stim_s : float
        Minimum stimulus duration (offset - onset) in seconds.
    min_post_s : float
        Minimum post-stimulus duration (trial_end - offset) in seconds. A response
        shorter than this is not a real response. Defaulted here rather than routed
        through cfg because the notebooks do not pass min_stim_s either -- a cfg
        value they never read would look effective and do nothing.
    max_post_s : float
        Maximum post-stimulus duration (trial_end - offset) in seconds.
    outlier_method : {"IQR","zscore","percentile"}
        Method for outlier trimming of post-stimulus durations, applied after min/max filters.
    iqr_k : float
        Tukey factor for IQR fences (default 1.5).
    """

    # Language suffixes were enumerated one at a time, and the list fell behind the
    # data: auditory_naming_ENG was absent, so it never normalised to "audio" and
    # PAT_6704 acquired a fourth condition folder holding one stray trial, splitting
    # that patient's auditory trials across two incomplete sets. Enumerating is the
    # bug; the prefix fallback below means the next language code cannot repeat it.
    _PREFIX_RULES = (("auditory_naming", "audio"), ("picture_naming", "picture"),
                     ("reading_completion", "reading"))
    _unmapped_seen = set()

    def _norm_cond(x):
        s = str(x).strip().lower()
        if condition_aliases and s in condition_aliases:
            s = condition_aliases[s]
        mapping = {
            "picture_naming": "picture", "pict": "picture",
            "auditory": "audio", "auditory_naming": "audio", "auditory_naming_ger": "audio",
            "auditory_naming_fre": "audio", "auditory_naming_eng": "audio", "audi": "audio",
            "read": "reading", "reading": "reading", "reading_completion": "reading",
            "reading_completion_fre": "reading", "reading_completion_ger": "reading",
            "reading_completion_eng": "reading",
        }
        if s in mapping:
            return mapping[s]
        for pre, canon in _PREFIX_RULES:
            if s.startswith(pre):
                if s not in _unmapped_seen:
                    _unmapped_seen.add(s)
                    print(f"  [cond] {s!r} not in the table; matched prefix {pre!r} "
                         f"-> {canon!r}")
                return canon
        return s

    # DE-DUPLICATION. prep0 can hold the same trial table twice under two extraction
    # dates -- 4 of the 6 MicroEPI patients have a byte-identical
    # ..._2026-05-08.tsv / ..._2026-05-12.tsv pair for every condition. This glob
    # ingested both, so every trial was counted TWICE and every per-condition average
    # was silently double-weighted (audio n_in = 104 = 52 x 2).
    #
    # Only BYTE-IDENTICAL files are dropped. A patient with genuinely different files
    # for one condition has separate blocks that must still be concatenated, and
    # de-duplicating on filename or on condition would throw those away. Verified
    # across all 30 patients: 90 TSVs, 12 redundant copies, and no patient has two
    # DIFFERENT files for the same condition.
    _seen_hashes = {}
    _paths = []
    for path in sorted(glob.glob(os.path.join(prep_dir, "*.tsv"))):
        try:
            h = hashlib.md5(open(path, "rb").read()).hexdigest()
        except OSError:
            _paths.append(path)
            continue
        if h in _seen_hashes:
            print(f"  [trials] skipping {os.path.basename(path)} — byte-identical to "
                 f"{os.path.basename(_seen_hashes[h])}")
            continue
        _seen_hashes[h] = path
        _paths.append(path)

    rows = []
    for path in _paths:
        df = pd.read_csv(path, sep="\t")
        cols = {c.lower() for c in df.columns}
        need = {"sample", "sample_offsets", "trial_end"}
        if not need.issubset(cols):
            raise ValueError(f"[{os.path.basename(path)}] Missing {need}. Found: {sorted(cols)}")

        on   = df[[c for c in df.columns if c.lower()=="sample"][0]].to_numpy(dtype=np.int64)
        off  = df[[c for c in df.columns if c.lower()=="sample_offsets"][0]].to_numpy(dtype=np.int64)
        tend = df[[c for c in df.columns if c.lower()=="trial_end"][0]].to_numpy(dtype=np.int64)

        cond = None
        for c in ["condition_name","condition","trial_type","cond"]:
            if c in df.columns:
                cond = df[c].astype(str).map(_norm_cond).values; break
        if cond is None:
            base = os.path.basename(path).lower()
            if   "picture" in base: cond = np.array(["picture"]*len(on))
            elif "aud" in base:     cond = np.array(["audio"]*len(on))
            elif "reading" in base or "read" in base: cond = np.array(["reading"]*len(on))
            else: cond = np.array(["unknown"]*len(on))

        order = None
        for c in ["trial_idx","trial_id","index"]:
            if c in df.columns:
                order = df[c].to_numpy(); break
                
        # resp_accuracy column (written by LFfunctions_PDextract.save_onsets_offsets_by_condition).
        # Missing column or blank cells → treated as Unknown and dropped when filter is active.
        ra_col = None
        for c in ["resp_accuracy", "response_type", "responseaccuracy"]:
            if c in df.columns:
                ra_col = df[c].astype(str).str.strip().str.lower().values
                break
        if ra_col is None:
            ra_col = np.array([""] * len(on), dtype=object)
            
        for i in range(len(on)):
            rows.append(dict(
                sample=int(on[i]), sample_offsets=int(off[i]), trial_end=int(tend[i]),
                condition=str(cond[i]), order=(None if order is None else order[i]),
                resp_accuracy=str(ra_col[i]),
                source=os.path.basename(path)
            ))

    if not rows:
        return {}

    by_cond = {}
    for r in rows:
        by_cond.setdefault(r["condition"], []).append(r)
    for cond in by_cond:
        seq = by_cond[cond]
        seq.sort(key=lambda d: (float("inf") if d["order"] is None else d["order"], d["sample"]))

    groups, stats = {}, []
    qc_dir = os.path.dirname(report_path) if report_path else None
    if qc_dir: os.makedirs(qc_dir, exist_ok=True)

    method_norm = str(outlier_method).strip().lower()
    if method_norm in ("pct", "quant", "quantile"):
        method_norm = "percentile"

    for cond, seq in by_cond.items():
        on   = np.array([r["sample"] for r in seq], dtype=np.int64)
        off  = np.array([r["sample_offsets"] for r in seq], dtype=np.int64)
        tend = np.array([r["trial_end"] for r in seq], dtype=np.int64)

        bad1 = off <= on
        bad2 = tend < off
        if bad1.any() or bad2.any():
            raise ValueError(f"[{patient_id or 'patient'} | {cond}] annotation error: "
                             f"{bad1.sum()} rows offset<=onset; {bad2.sum()} rows trial_end<offset")
        # Drop trials whose resp_accuracy isn't in the allowed set
        # (default keeps only 'correct'; "wrong" and blank/no-response are dropped).
        ra = np.array([str(r.get("resp_accuracy","")).strip().lower() for r in seq], dtype=object)
        if keep_resp_accuracy is not None:
            allowed = {str(s).strip().lower() for s in keep_resp_accuracy}
            ra_keep = np.array([v in allowed for v in ra], dtype=bool)
        else:
            ra_keep = np.ones(len(ra), dtype=bool)
            
        stim_s = (off - on) / float(fs_hz)
        post_s = (tend - off) / float(fs_hz)

        # 1) min stimulus + min/max post-stimulus duration filters
        keep = (stim_s >= float(min_stim_s)) & (post_s >= float(min_post_s)) \
             & (post_s <= float(max_post_s)) & ra_keep
        
        # 2) outlier filter on post durations (applied after hard limits)
        lo = hi = np.nan
        z_low_used = z_high_used = np.nan
        pct_low_used = pct_high_used = np.nan
        legacy_p_low = float(p_low)
        legacy_p_high = float(p_high)

        if keep.any():
            pdur = post_s[keep]

            if method_norm == "iqr":
                q1, q3 = np.percentile(pdur, [25, 75])
                iqr = q3 - q1
                lo, hi = q1 - float(iqr_k) * iqr, q3 + float(iqr_k) * iqr
                keep &= (post_s >= lo) & (post_s <= hi)

            elif method_norm == "zscore":
                if z_low is None or z_high is None:
                    if (p_low != -3.0) or (p_high != 3.0):
                        warnings.warn(
                            "[collect_trials] `p_low/p_high` are deprecated; "
                            "prefer `z_low/z_high` for outlier_method='zscore'.",
                            DeprecationWarning
                        )
                    zL = float(z_low) if z_low is not None else float(p_low)
                    zH = float(z_high) if z_high is not None else float(p_high)
                else:
                    zL, zH = float(z_low), float(z_high)

                mu = float(np.nanmean(pdur))
                sd = float(np.nanstd(pdur) + 1e-12)
                z = (post_s - mu) / sd
                keep &= (z >= zL) & (z <= zH)
                lo, hi = mu + zL * sd, mu + zH * sd
                z_low_used, z_high_used = zL, zH

            elif method_norm == "percentile":
                if pct_low is None or pct_high is None:
                    if (p_low != -3.0) or (p_high != 3.0):
                        warnings.warn(
                            "[collect_trials] `p_low/p_high` are deprecated; "
                            "prefer `pct_low/pct_high` for outlier_method='percentile'.",
                            DeprecationWarning
                        )
                    pl = float(pct_low) if pct_low is not None else float(p_low)
                    ph = float(pct_high) if pct_high is not None else float(p_high)
                else:
                    pl, ph = float(pct_low), float(pct_high)

                if not (0.0 <= pl < ph <= 100.0):
                    raise ValueError(f"[{patient_id or 'patient'} | {cond}] "
                                     f"percentile bounds must satisfy 0 ≤ low < high ≤ 100; got {pl}, {ph}")
                lo, hi = np.percentile(pdur, [pl, ph])
                keep &= (post_s >= lo) & (post_s <= hi)
                pct_low_used, pct_high_used = pl, ph

        on_kept, off_kept, tend_kept = on[keep], off[keep], tend[keep]
        groups[cond] = (on_kept, off_kept, tend_kept)

        if qc_dir:
            import matplotlib.pyplot as _plt
            _plt.figure(figsize=(8,4))
            bins = np.linspace(0, max(1e-9, float(np.nanmax(post_s))) * 1.1, 40)
            _plt.hist(post_s[keep], bins=bins, color="green", alpha=.6, label=f"Kept (n={keep.sum()})")
            if (~keep).any():
                _plt.hist(post_s[~keep], bins=bins, color="red", alpha=.5, label=f"Dropped (n={(~keep).sum()})")
            if np.isfinite(lo): _plt.axvline(lo, color="blue", ls="--", label=f"low={lo:.2f}s")
            if np.isfinite(hi): _plt.axvline(hi, color="blue", ls="--", label=f"high={hi:.2f}s")
            _plt.axvline(float(max_post_s), color="orange", ls="--", label=f"max_post={max_post_s:.1f}s")
            if float(min_post_s) > 0:
                _plt.axvline(float(min_post_s), color="orange", ls="--", label=f"min_post={min_post_s:.1f}s")
            _plt.title(f"{patient_id} | {cond} | stim≥{min_stim_s:.2f}s | "
                       f"{min_post_s:.1f}s≤post≤{max_post_s:.1f}s | {method_norm}")
            _plt.xlabel("Post duration (s)"); _plt.ylabel("Count"); _plt.legend(); _plt.tight_layout()
            out_png = os.path.join(qc_dir, f"{patient_id}_{cond}_{method_norm}_postDur_QC.png")
            _plt.savefig(out_png, dpi=160); _plt.close()

        stats.append(dict(
            patient_id=patient_id, condition=cond,
            n_in=len(on), n_kept=int(keep.sum()),
            stim_min=float(stim_s.min()), stim_med=float(np.median(stim_s)), stim_max=float(stim_s.max()),
            post_min=float(post_s.min()), post_med=float(np.median(post_s)), post_max=float(post_s.max()),
            outlier_method=str(outlier_method),
            min_post_s=float(min_post_s),
            max_post_s=float(max_post_s),
            z_low=z_low_used, z_high=z_high_used,
            pct_low=pct_low_used, pct_high=pct_high_used,
            legacy_p_low=legacy_p_low, legacy_p_high=legacy_p_high,
            iqr_k=float(iqr_k),
            post_lo=float(lo) if np.isfinite(lo) else np.nan,
            post_hi=float(hi) if np.isfinite(hi) else np.nan,
            n_dropped_accuracy=int((~ra_keep).sum()),
        ))

    if report_path and stats:
        pd.DataFrame(stats).to_csv(report_path, sep="\t", index=False)

    return groups


def plot_montage_overview(
    signals, fs, names, *,
    cond_groups=None,
    save_dir=None,
    patient_id="",
    figsize_w=200,
    dpi=150,
    fmt="tif",              # "tif" | "png" — default keeps 140's output format
):
    """
    Save a stacked montage trace of all channels over full recording duration.
    Optionally overlays trial onsets/offsets from cond_groups as vertical lines.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    fs = float(fs)
    n_samples = signals.shape[0]
    time = np.arange(n_samples) / fs

    # Normalize each channel
    centered = signals - np.mean(signals, axis=0, keepdims=True)
    peak = np.max(np.abs(centered), axis=0, keepdims=True)
    peak[peak == 0] = 1.0
    normalized = centered / peak

    fig_height = max(4.0, len(names) * 0.4)
    fig, ax = plt.subplots(figsize=(figsize_w, fig_height), facecolor="none")

    for i, sig in enumerate(normalized.T):
        ax.plot(time, sig + i, lw=0.3, color="k", alpha=0.6)

    # Overlay onsets/offsets from all conditions
    if cond_groups:
        colors = plt.cm.tab10.colors
        for ci, (cond, (onsets, offsets, _)) in enumerate(cond_groups.items()):
            color = colors[ci % len(colors)]
            for on in onsets:
                ax.axvline(on / fs, color=color, lw=0.5, alpha=0.5, ls="-")
            for off in offsets:
                ax.axvline(off / fs, color=color, lw=0.5, alpha=0.5, ls="--")

        # Legend per condition
        handles = [plt.Line2D([0], [0], color=colors[ci % len(colors)], lw=1.0, label=cond)
                   for ci, cond in enumerate(cond_groups.keys())]
        ax.legend(handles=handles, loc="upper right", fontsize=8)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channels")
    ax.set_title(f"{patient_id} — Montage overview (post-notch)")
    ax.set_xlim(time[0], time[-1])
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.grid(axis="x", ls="--", alpha=0.3)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # A 200-inch figure at 150 dpi is 30000 px wide; as TIFF that is ~1 GB per
        # patient, and 9 of them were 73% of the whole real-time output tree. PNG of
        # the same figure is a fraction of that.
        _ext = "png" if str(fmt).lower() == "png" else "tif"
        out = os.path.join(save_dir, f"{patient_id}_montage_overview.{_ext}")
        fig.savefig(out, dpi=dpi, format=("png" if _ext == "png" else "tiff"))
        plt.close(fig)
        return out
    else:
        plt.show()
        plt.close(fig)
        return None
    
    
    
    