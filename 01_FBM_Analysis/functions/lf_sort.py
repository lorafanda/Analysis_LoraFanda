"""
lf_sort.py
==========

Micro-electrode spike detection / pre-sorting for tetrode (4-wire bundle)
recordings, with a configurable **detection unit**:

    detection_unit="tetrode"  -> one event per neuron, deduplicated across the 4
                                 wires of a tetrode (the largest-deflection wire is
                                 recorded as `peak_channel`). Best when the 4 wires
                                 sample the *same* neurons (true tetrode sorting).

    detection_unit="wire"     -> spikes detected on **each contact independently**,
                                 with NO cross-wire merge. A spike that crosses
                                 threshold on several wires yields one event per
                                 wire. Best when you want per-contact spike trains.

This module is a clean, standalone reconstruction of the spike helpers that the
notebook used (`lf_spikes`), keeping the validated algorithms (Quian-Quiroga MAD
noise estimate, zero-phase spike-band bandpass, intra-tetrode CAR, greedy
refractory merge, edge-safe snippet extraction) and adding the `detection_unit`
switch + a per-contact data model.

Reuse / provenance
------------------
* The MAD detector, intra-tetrode CAR and snippet logic mirror the original
  `lf_spikes` (recovered, faithful). `detection_unit="tetrode"` reproduces the
  original merge exactly.
* The CAR idea is the tetrode analogue of `lf_ersp.apply_grid_car` (per-array
  common-average reference); here the "array" is the 4-wire bundle.
* Style and standalone conventions follow `lf_notch.py` (numpy-only at import
  time; scipy imported lazily inside functions).

Public API
----------
bandpass_spike_band(signals, fs, *, low=300, high=6000, order=4)
intra_tetrode_car(signals_tetrode)
detect_threshold_crossings(signals_filt, fs, *, k_mad, polarity, refractory_ms,
                           detection_unit, verbose)
extract_snippets(signals_filt, spike_idx, fs, *, pre_ms=0.5, post_ms=1.5)
group_micros_into_tetrodes(names, *, exclude_prefixes=("X","x"))
sort_session(signals_micro, names_micro, fs, *, detection_unit, ...)  # high-level driver
"""
from __future__ import annotations

import re
import numpy as np

__all__ = [
    "bandpass_spike_band",
    "intra_tetrode_car",
    "detect_threshold_crossings",
    "extract_snippets",
    "group_micros_into_tetrodes",
    "sort_session",
    "plot_spike_rate_over_time",
    "plot_snippet_overlay",
    "plot_psth",
    "plot_activity_raster",
    "cluster_and_qc",
]


# -----------------------------------------------------------------------------
# Conditioning
# -----------------------------------------------------------------------------
def bandpass_spike_band(signals, fs, *, low=300.0, high=6000.0, order=4):
    """Zero-phase Butterworth bandpass for the spike band.

    `signals` : (n_samples,) or (n_samples, n_channels). Returns same shape, float32.
    The high edge is clamped to 0.95 * Nyquist for filter stability.
    """
    from scipy.signal import butter, filtfilt

    X = np.asarray(signals, dtype=np.float32)
    nyq = 0.5 * fs
    high_eff = min(high, 0.95 * nyq)
    b, a = butter(order, [low / nyq, high_eff / nyq], btype="band")
    if X.ndim == 1:
        return filtfilt(b, a, X).astype(np.float32)
    return filtfilt(b, a, X, axis=0).astype(np.float32)


def intra_tetrode_car(signals_tetrode):
    """Common-average reference within a tetrode.

    Subtracts the across-wire mean from each wire, removing shared (common-mode)
    noise without cancelling single-unit signals (which are not common-mode across
    the bundle). Input must be (n_samples, n_wires). Returns float32, same shape.
    """
    X = np.asarray(signals_tetrode, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected (n_samples, n_wires), got shape {X.shape}")
    car = np.mean(X, axis=1, keepdims=True)
    return (X - car).astype(np.float32)


def _mad_sigma(x):
    """Quian-Quiroga robust noise estimate: sigma = median(|x|) / 0.6745."""
    return float(np.median(np.abs(x)) / 0.6745)


# -----------------------------------------------------------------------------
# Detection
# -----------------------------------------------------------------------------
def _wire_crossings(x, thr, polarity):
    """Rising-edge sample indices where channel `x` first exceeds its threshold."""
    if polarity == "neg":
        hit = x < -thr
    elif polarity == "pos":
        hit = x > thr
    elif polarity == "both":
        hit = np.abs(x) > thr
    else:
        raise ValueError(f"Unknown polarity {polarity!r}")
    # first sample of each contiguous super-threshold run
    return np.where(np.diff(hit.astype(np.int8)) == 1)[0] + 1


def _merge_refractory(candidates, X, refr_samp):
    """
    Greedy refractory merge over (sample, channel) candidates sorted by sample.

    Maintains a 'pending' best (largest |amplitude|) event; when a new candidate
    arrives more than `refr_samp` after the pending one, the pending is committed
    and the new one becomes pending; otherwise the larger-amplitude of the two is
    kept. Returns (idx_list, ch_list). This is the exact original tetrode merge.
    """
    merged_idx, merged_ch = [], []
    pending_idx, pending_amp, pending_ch = None, -np.inf, -1
    for s, c in candidates:
        amp = abs(float(X[s, c]))
        if pending_idx is None or (s - pending_idx) > refr_samp:
            if pending_idx is not None:
                merged_idx.append(pending_idx)
                merged_ch.append(pending_ch)
            pending_idx, pending_amp, pending_ch = s, amp, c
        else:  # within refractory window: keep the larger deflection
            if amp > pending_amp:
                pending_idx, pending_amp, pending_ch = s, amp, c
    if pending_idx is not None:
        merged_idx.append(pending_idx)
        merged_ch.append(pending_ch)
    return merged_idx, merged_ch


def detect_threshold_crossings(signals_filt, fs, *, k_mad=6.0, polarity="neg",
                               refractory_ms=1.5, detection_unit="tetrode",
                               verbose=True):
    """
    MAD-based spike detection, configurable per tetrode or per contact.

    For each channel, threshold = k_mad * sigma_MAD (Quian-Quiroga). A spike is a
    rising threshold crossing (below -thr for 'neg', above +thr for 'pos', either
    for 'both').

    detection_unit
    --------------
    "tetrode" : crossings from all wires are merged with one shared refractory
                window; coincident multi-wire firing collapses to a single event
                whose `peak_channel` is the max-deflection wire. (Original behaviour.)
    "wire"    : each wire is detected and refractory-merged **independently**; a
                spike crossing several wires yields one event per wire. `peak_channel`
                is simply the wire the event came from.

    Returns
    -------
    spike_idx    : (n_spikes,) int64, sorted by sample.
    peak_channel : (n_spikes,) int8, the wire index for each event.
    thresholds   : (n_channels,) float, per-wire detection threshold.
    """
    X = np.asarray(signals_filt, dtype=np.float32)
    if X.ndim == 1:
        X = X[:, None]
    n_samples, n_ch = X.shape

    sigmas = np.array([_mad_sigma(X[:, c]) for c in range(n_ch)], dtype=np.float32)
    thresholds = k_mad * sigmas
    refr_samp = int(round(refractory_ms * 1e-3 * fs))

    # per-wire candidate crossings
    cand_by_wire = {c: _wire_crossings(X[:, c], thresholds[c], polarity)
                    for c in range(n_ch)}

    if detection_unit == "tetrode":
        candidates = sorted(
            ((int(s), c) for c in range(n_ch) for s in cand_by_wire[c]),
            key=lambda t: t[0],
        )
        midx, mch = _merge_refractory(candidates, X, refr_samp)
        spike_idx = np.array(midx, dtype=np.int64)
        peak_channel = np.array(mch, dtype=np.int8)

    elif detection_unit == "wire":
        idx_parts, ch_parts = [], []
        for c in range(n_ch):
            cand = [(int(s), c) for s in cand_by_wire[c]]  # already sorted per wire
            midx, mch = _merge_refractory(cand, X, refr_samp)
            idx_parts.extend(midx)
            ch_parts.extend(mch)
        spike_idx = np.array(idx_parts, dtype=np.int64)
        peak_channel = np.array(ch_parts, dtype=np.int8)
        order = np.argsort(spike_idx, kind="stable")
        spike_idx = spike_idx[order]
        peak_channel = peak_channel[order]

    else:
        raise ValueError(f"detection_unit must be 'tetrode' or 'wire', got {detection_unit!r}")

    if verbose:
        rate = spike_idx.size / (n_samples / fs)
        print(f"  [detect:{detection_unit}] {spike_idx.size} events ({rate:.1f} Hz overall) "
              f"thresholds={np.round(thresholds, 1).tolist()}")
    return spike_idx, peak_channel, thresholds


# -----------------------------------------------------------------------------
# Snippets
# -----------------------------------------------------------------------------
def extract_snippets(signals_filt, spike_idx, fs, *, pre_ms=0.5, post_ms=1.5):
    """Extract waveform snippets around each spike index, across all wires.

    Returns
    -------
    snippets  : (n_kept, width, n_channels) float32, width = pre+post samples;
                the spike sits at sample `pre` within each snippet.
    keep_mask : (n_spikes,) bool — False for spikes too close to a recording edge.
    """
    X = np.asarray(signals_filt, dtype=np.float32)
    if X.ndim == 1:
        X = X[:, None]
    n_samples, n_ch = X.shape
    spike_idx = np.asarray(spike_idx, dtype=np.int64)

    pre = int(round(pre_ms * 1e-3 * fs))
    post = int(round(post_ms * 1e-3 * fs))
    width = pre + post

    keep = (spike_idx >= pre) & (spike_idx + post <= n_samples)
    valid_idx = spike_idx[keep]

    snippets = np.empty((valid_idx.size, width, n_ch), dtype=np.float32)
    for i, s in enumerate(valid_idx):
        snippets[i] = X[s - pre: s + post, :]
    return snippets, keep


# -----------------------------------------------------------------------------
# Tetrode grouping (notebook cell-5 logic, with the X-exclusion fixed)
# -----------------------------------------------------------------------------
def group_micros_into_tetrodes(names, *, exclude_prefixes=("X", "x")):
    """
    Group an ordered list of micro channel names into tetrodes of 4 wires sharing
    the same alphabetic prefix. Prefixes starting with any string in
    `exclude_prefixes` are skipped (defense-in-depth against unused 'X' channels).

    Returns dict {tetrode_label: [column indices into `names`]}.
    """
    groups: dict[str, list[int]] = {}
    for ci, nm in enumerate(names):
        m = re.match(r"([A-Za-z]+)(\d+)", str(nm))
        if not m:
            continue
        prefix = m.group(1)
        if any(prefix.startswith(p) for p in exclude_prefixes):
            continue
        groups.setdefault(prefix, []).append(ci)

    tetrodes: dict[str, list[int]] = {}
    for prefix, idxs in groups.items():
        idxs.sort(key=lambda c: int(re.match(r"[A-Za-z]+(\d+)", str(names[c])).group(1)))
        for ti, start in enumerate(range(0, len(idxs), 4)):
            chunk = idxs[start:start + 4]
            if len(chunk) == 4:
                lbl = prefix if ti == 0 else f"{prefix}_{ti + 1}"
                tetrodes[lbl] = chunk
    return tetrodes


# -----------------------------------------------------------------------------
# Fragment-boundary guard (notebook cell-5 logic)
# -----------------------------------------------------------------------------
def _fragment_keep_mask(spike_idx, fs, fragment_boundaries_s, guard_ms):
    if guard_ms <= 0 or fragment_boundaries_s is None or len(fragment_boundaries_s) <= 2:
        return np.ones(len(spike_idx), dtype=bool)
    guard = int(round(guard_ms * 1e-3 * fs))
    keep = np.ones(len(spike_idx), dtype=bool)
    for b_s in fragment_boundaries_s[1:-1]:  # skip session start (0) and end
        b = int(round(b_s * fs))
        keep &= ~((spike_idx >= b - guard) & (spike_idx <= b + guard))
    return keep


# -----------------------------------------------------------------------------
# High-level driver
# -----------------------------------------------------------------------------
def sort_session(signals_micro, names_micro, fs, *,
                 detection_unit="tetrode", exclude_prefixes=("X", "x"),
                 spike_low_hz=300.0, spike_high_hz=6000.0, bp_order=4,
                 k_mad=6.0, polarity="neg", refractory_ms=1.5,
                 do_car=True, snippet_pre_ms=0.5, snippet_post_ms=1.5,
                 min_amplitude_uv=None, max_amplitude_uv=300.0,
                 fragment_boundaries_s=None,
                 fragment_guard_ms=100.0, verbose=True):
    """
    Run the full per-tetrode detection pipeline and return a results dict.

    Pipeline per tetrode: bandpass -> (intra-tetrode CAR) -> detect (tetrode|wire)
    -> fragment-boundary guard -> snippets -> amplitude window.

    Amplitude window (measured as the largest |deflection| on the spike's peak wire,
    from its snippet): a spike is kept only if
    ``min_amplitude_uv <= peak_amp <= max_amplitude_uv``.
    ``min_amplitude_uv`` is the amplitude FLOOR (e.g. 100 keeps only spikes whose
    trough reaches below -100 µV, isolating the larger, cleaner units);
    ``max_amplitude_uv`` is the saturation/artifact ceiling. Either may be None.

    Returns
    -------
    dict keyed by:
      - tetrode label                  (detection_unit="tetrode"), or
      - "<tetrode label>::<wire name>" (detection_unit="wire", one entry per contact)
    Each value has: spike_times_s, snippets (N,time,wires), peak_channel,
    thresholds, wire_indices, wire_names, and counts (n_raw, n_final).
    """
    signals_micro = np.asarray(signals_micro)
    tetrodes = group_micros_into_tetrodes(names_micro, exclude_prefixes=exclude_prefixes)
    if not tetrodes:
        raise ValueError("No tetrodes formed — check naming / exclude_prefixes.")

    out = {}
    for tet_lbl, ch_idx in tetrodes.items():
        wire_names = [names_micro[c] for c in ch_idx]
        if verbose:
            print(f"\n--- Tetrode {tet_lbl} (wires {wire_names}) ---")

        filt = bandpass_spike_band(signals_micro[:, ch_idx], fs,
                                   low=spike_low_hz, high=spike_high_hz, order=bp_order)
        car = intra_tetrode_car(filt) if do_car else filt

        spike_idx, peak_ch, thrs = detect_threshold_crossings(
            car, fs, k_mad=k_mad, polarity=polarity, refractory_ms=refractory_ms,
            detection_unit=detection_unit, verbose=verbose,
        )
        n_raw = spike_idx.size

        # fragment-boundary guard
        keep = _fragment_keep_mask(spike_idx, fs, fragment_boundaries_s, fragment_guard_ms)
        spike_idx, peak_ch = spike_idx[keep], peak_ch[keep]

        # snippets (edge-trimmed)
        snippets, keep_edge = extract_snippets(car, spike_idx, fs,
                                               pre_ms=snippet_pre_ms, post_ms=snippet_post_ms)
        spike_idx, peak_ch = spike_idx[keep_edge], peak_ch[keep_edge]

        # amplitude window: largest |deflection| on each spike's peak wire (from its snippet)
        if snippets.shape[0] and (min_amplitude_uv is not None or max_amplitude_uv is not None):
            peak_amp = np.abs(snippets[np.arange(snippets.shape[0]), :, peak_ch]).max(axis=1)
            keep_amp = np.ones(snippets.shape[0], dtype=bool)
            if min_amplitude_uv is not None:
                keep_amp &= peak_amp >= float(min_amplitude_uv)
            if max_amplitude_uv is not None:
                keep_amp &= peak_amp <= float(max_amplitude_uv)
            n_drop = int((~keep_amp).sum())
            snippets = snippets[keep_amp]
            spike_idx = spike_idx[keep_amp]
            peak_ch = peak_ch[keep_amp]
            if verbose and n_drop:
                lo = "" if min_amplitude_uv is None else f"{min_amplitude_uv:g}"
                hi = "inf" if max_amplitude_uv is None else f"{max_amplitude_uv:g}"
                print(f"  amplitude window [{lo or '0'}, {hi}] uV: dropped {n_drop}")

        base_entry = lambda sel: dict(
            spike_times_s=spike_idx[sel].astype(np.float64) / fs,
            snippets=snippets[sel],
            peak_channel=peak_ch[sel],
            thresholds=thrs,
            wire_indices=list(ch_idx),
            wire_names=wire_names,
            n_raw=int(n_raw),
            n_final=int(np.count_nonzero(sel)),
        )

        if detection_unit == "tetrode":
            out[tet_lbl] = base_entry(np.ones(spike_idx.size, dtype=bool))
            if verbose:
                print(f"  -> {out[tet_lbl]['n_final']} events kept (raw={n_raw})")
        else:  # per-contact: split by wire
            for w in range(len(ch_idx)):
                sel = (peak_ch == w)
                key = f"{tet_lbl}::{wire_names[w]}"
                out[key] = base_entry(sel)
                if verbose:
                    print(f"  -> contact {wire_names[w]}: {out[key]['n_final']} events")
    return out


# -----------------------------------------------------------------------------
# Diagnostic plot helpers (readable replacements for the old lf_spikes.plot_*)
# matplotlib is imported lazily so importing lf_sort never requires it.
# -----------------------------------------------------------------------------
def plot_spike_rate_over_time(spike_times_s, total_duration_s, *, bin_s=10.0,
                              ax=None, title=""):
    """Firing rate (Hz) vs time, binned at bin_s seconds."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 2))
    st = np.asarray(spike_times_s, dtype=float)
    edges = np.arange(0.0, float(total_duration_s) + bin_s, bin_s)
    counts, _ = np.histogram(st, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rate = counts / bin_s
    ax.plot(centers, rate, color="steelblue", lw=0.8)
    ax.fill_between(centers, 0, rate, color="steelblue", alpha=0.25, step="mid")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rate (Hz)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_snippet_overlay(snippets, fs, *, peak_channel=None, max_show=500,
                         ax=None, title=""):
    """Overlay spike waveforms on the dominant wire + mean template.

    snippets : (N, time, wires)  — the layout returned by extract_snippets.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 2))
    S = np.asarray(snippets, dtype=float)
    if S.ndim != 3 or S.shape[0] == 0:
        ax.text(0.5, 0.5, "no snippets", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return ax
    N, T, W = S.shape
    pc = np.asarray(peak_channel, dtype=int) if peak_channel is not None else np.zeros(N, dtype=int)
    dom = int(np.bincount(pc, minlength=W).argmax())
    t_ms = np.arange(T) / fs * 1000.0
    show = np.random.choice(N, min(max_show, N), replace=False)
    for j in show:
        ax.plot(t_ms, S[j, :, pc[j]], color="0.6", alpha=0.10, lw=0.4)
    on_dom = S[pc == dom]
    if on_dom.shape[0]:
        ax.plot(t_ms, on_dom[:, :, dom].mean(axis=0), "r-", lw=1.5, label="mean")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (uV)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_psth(spike_times_s, event_times_s, *, window_s=(-0.5, 1.5), bin_s=0.025,
              ax=None, title="", color="k"):
    """Peri-stimulus time histogram (Hz), spikes aligned to each event."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 2))
    st = np.asarray(spike_times_s, dtype=float)
    ev = np.asarray(event_times_s, dtype=float)
    t0, t1 = window_s
    edges = np.arange(t0, t1 + bin_s, bin_s)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if ev.size == 0 or st.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return ax
    rel = []
    for e in ev:
        d = st - e
        rel.append(d[(d >= t0) & (d < t1)])
    rel = np.concatenate(rel) if rel else np.array([])
    counts, _ = np.histogram(rel, bins=edges)
    rate = counts / (bin_s * max(1, ev.size))
    ax.bar(centers, rate, width=bin_s, color=color, alpha=0.8, align="center")
    ax.axvline(0, color="r", ls="--", lw=0.8)
    ax.set_xlabel("Time from event (s)")
    ax.set_ylabel("Rate (Hz)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_activity_raster(spikes_per_tetrode, duration_s=None, *, t_range=None, ax=None,
                         max_ticks_per_unit=20000, title="Spike raster"):
    """Dot raster of spiking activity: one row per unit, a tick at each spike time.

    spikes_per_tetrode : dict returned by sort_session (keys = units / contacts).
    duration_s : full session length (sets x-limit if t_range is None).
    t_range : optional (t0, t1) seconds to zoom in on a window.
    max_ticks_per_unit : cap ticks drawn per row (uniformly subsampled) so very active
        units stay responsive; the row label still shows the true spike count.
    """
    import matplotlib.pyplot as plt
    items = list(spikes_per_tetrode.items())
    if ax is None:
        _, ax = plt.subplots(figsize=(14, max(1.5, 0.35 * len(items))))
    labels = []
    for row, (lbl, d) in enumerate(items):
        st_full = np.asarray(d["spike_times_s"], dtype=float)
        st = st_full
        if t_range is not None:
            st = st[(st >= t_range[0]) & (st <= t_range[1])]
        if st.size > max_ticks_per_unit:
            st = st[np.linspace(0, st.size - 1, max_ticks_per_unit).astype(int)]
        ax.plot(st, np.full(st.size, row), "|", color="k", markersize=4, markeredgewidth=0.5)
        labels.append(f"{lbl}  (n={st_full.size})")
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(-0.5, len(items) - 0.5)
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    if t_range is not None:
        ax.set_xlim(*t_range)
    elif duration_s:
        ax.set_xlim(0, duration_s)
    ax.grid(True, axis="x", alpha=0.3)
    return ax


def cluster_and_qc(spikes_per_tetrode, fs, *, snippet_pre_ms=0.5, max_k=4,
                   isi_refractory_ms=2.0, min_spikes=40, units=None,
                   plot=True, random_state=0):
    """Per-contact spike clustering (PCA -> GMM, k chosen by BIC) + quality metrics.

    For each unit/contact in `spikes_per_tetrode` (the dict from sort_session, snippets
    laid out (N, time, wires)), the peak-wire waveforms are PCA-reduced and clustered
    with a Gaussian mixture whose component count is picked by BIC (1..max_k). For each
    resulting cluster it computes:
      n, rate_hz, isi_viol_pct (<isi_refractory_ms), snr (median|peak| / baseline std),
      amp_uv (median peak), t2p_ms (trough->peak of the mean template).

    Returns a list of per-cluster metric dicts. If plot=True, draws per unit:
      [PCA scatter coloured by cluster] [mean templates] [ISI of the biggest cluster].

    A "clean single unit" is roughly: isi_viol_pct < ~1-2%, snr > ~5, tight template.
    """
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    import matplotlib.pyplot as plt

    keys = list(spikes_per_tetrode.keys()) if units is None else [u for u in units if u in spikes_per_tetrode]
    pre = int(round(snippet_pre_ms * 1e-3 * fs))
    rows = []

    if plot:
        fig, axes = plt.subplots(max(1, len(keys)), 3, figsize=(13, 3.0 * max(1, len(keys))))
        axes = np.atleast_2d(axes)

    for ri, key in enumerate(keys):
        d = spikes_per_tetrode[key]
        snip = np.asarray(d["snippets"]); st = np.asarray(d["spike_times_s"])
        pc = np.asarray(d["peak_channel"], dtype=int)
        n = snip.shape[0]
        if n == 0:
            if plot:
                for j in range(3):
                    axes[ri, j].text(0.5, 0.5, "no spikes", ha="center", va="center", transform=axes[ri, j].transAxes)
            continue

        wf = snip[np.arange(n), :, pc]            # (N, time) on each spike's peak wire

        if n >= min_spikes:
            Z = PCA(n_components=min(3, wf.shape[1]), random_state=random_state).fit_transform(wf)
            best = None
            kmax = max(1, min(max_k, n // 20))
            for k in range(1, kmax + 1):
                gm = GaussianMixture(k, covariance_type="full", random_state=random_state, n_init=2).fit(Z)
                bic = gm.bic(Z)
                if best is None or bic < best[0]:
                    best = (bic, k, gm)
            labels = best[2].predict(Z); k_best = best[1]
        else:
            Z = None; labels = np.zeros(n, dtype=int); k_best = 1

        for cl in range(k_best):
            m = labels == cl
            if not m.any():
                continue
            stc = np.sort(st[m]); wfc = wf[m]
            amp = np.abs(wfc).max(axis=1)
            tmpl = wfc.mean(axis=0)
            tr = int(np.argmin(tmpl)); pk = tr + int(np.argmax(tmpl[tr:]))
            t2p_ms = (pk - tr) * 1000.0 / fs
            base_std = float(wfc[:, :max(1, pre)].std(axis=1).mean())
            snr = float(np.median(amp) / (base_std + 1e-9))
            isi = np.diff(stc) * 1000.0 if stc.size > 1 else np.array([])
            viol = float((isi < isi_refractory_ms).mean() * 100) if isi.size else float("nan")
            span = float(stc[-1] - stc[0]) if stc.size > 1 else 0.0
            rows.append(dict(unit=key, cluster=int(cl), n=int(m.sum()),
                             rate_hz=(m.sum() / span) if span > 0 else float("nan"),
                             isi_viol_pct=viol, snr=snr,
                             amp_uv=float(np.median(amp)), t2p_ms=float(t2p_ms)))

        if plot:
            t_ms = (np.arange(wf.shape[1]) - pre) / fs * 1000.0
            if Z is not None:
                axes[ri, 0].scatter(Z[:, 0], Z[:, 1], c=labels, s=5, cmap="tab10", alpha=0.5)
            axes[ri, 0].set_title(f"{key}  (k={k_best})"); axes[ri, 0].set_xlabel("PC1"); axes[ri, 0].set_ylabel("PC2")
            for cl in range(k_best):
                mm = labels == cl
                if mm.any():
                    axes[ri, 1].plot(t_ms, wf[mm].mean(axis=0), lw=1.5, label=f"c{cl} (n={int(mm.sum())})")
            axes[ri, 1].axvline(0, color="k", ls="--", lw=0.5); axes[ri, 1].set_title("templates")
            axes[ri, 1].set_xlabel("ms"); axes[ri, 1].set_ylabel("uV"); axes[ri, 1].legend(fontsize=7)
            cl_big = int(np.bincount(labels).argmax())
            stc = np.sort(st[labels == cl_big]); isi = np.diff(stc) * 1000.0 if stc.size > 1 else np.array([])
            if isi.size:
                axes[ri, 2].hist(isi[isi < 50], bins=50, color="steelblue", alpha=0.8)
                axes[ri, 2].axvline(isi_refractory_ms, color="r", ls="--", lw=0.8)
            axes[ri, 2].set_title(f"ISI cluster {cl_big} (<50 ms)"); axes[ri, 2].set_xlabel("ms")

    if plot:
        plt.tight_layout(); plt.show()
    return rows
