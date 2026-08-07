#!/usr/bin/env python3
"""
lf_decompose.py — graded decomposition of high-gamma responses, and the statistics
needed to argue it is real.

WHY THIS EXISTS. The K=5 partition of the concatenated cohort does not survive its own
assumptions: silhouette cannot separate K=4/5/6 (0.166 / 0.176 / 0.171, a spread smaller
than seed noise), 85% of the silhouette mass sits in one cluster, five defensible
preprocessing pipelines produce five near-orthogonal partitions (ARI 0.17 between raw and
unit-norm), and 19% of sampled cortex has no cluster above 50% of the local mixture.
Rather than force a hard label, every electrode gets a graded weight on every component.

METHOD. Convex NMF (Ding, Li & Jordan 2010): X ~= X W G', with W, G >= 0. Components are
convex combinations of observed electrodes, so each one IS a real response profile rather
than an abstract direction, and — unlike ordinary NMF — the DATA may be signed. That
matters here: high-gamma is baseline-relative dB and goes negative. This is the method
used on this exact data type by Hamilton, Edwards & Chang (2018, Curr Biol
10.1016/j.cub.2018.04.033), Hamilton et al. (2021, Cell 10.1016/j.cell.2021.07.019, on
signed z-scored high-gamma), Kurteff et al. (2024, J Neurosci
10.1523/JNEUROSCI.1109-24.2024, on three concatenated conditions) and Norman-Haignere
et al. (2022, Curr Biol 10.1016/j.cub.2022.01.069).

VALIDATION. Rank is chosen by held-out reconstruction, following Norman-Haignere 2022
("cross-validated prediction accuracy to determine the number of components"), never by an
internal index. Stability is reported against a SIZE-MATCHED PSEUDO-GROUP NULL, because
that null already overturned a finding in this project: a leave-one-patient-out minimum of
0.448 looked like patient fragility until random groups of the same sizes scored
0.459 +/- 0.019. Consensus across preprocessing pipelines follows Lowe & Schall (2018,
eNeuro 10.1523/ENEURO.0131-18.2018), who clustered the consensus of 48 pipelines rather
than defending one.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "unit_norm", "per_patient_z", "PIPELINES",
    "convex_nmf", "reconstruct", "cv_rank_curve",
    "pipeline_consensus", "consensus_labels",
    "pseudo_group_null", "lopo_stability", "spatial_coherence",
    "soft_rgb", "mixture_summary",
]


# ── preprocessing ────────────────────────────────────────────────────────────
def unit_norm(X: np.ndarray) -> np.ndarray:
    """Scale each electrode to unit length: cluster on SHAPE, not magnitude.

    On this cohort it raises anatomical coherence from 1.36x to 1.70x chance (nearest-
    neighbour label agreement in fsaverage space, divided by a label shuffle). It also
    gives the quietest ~25% of electrodes arbitrary assignments, which is an argument for
    graded weights rather than for a gate: a low-confidence electrode should get flat
    loadings, not a coin-flip label.
    """
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def per_patient_z(X: np.ndarray, patient: np.ndarray) -> np.ndarray:
    """Centre and scale within each patient. Removes patient identity from the features
    (AMI with patient 0.18 -> 0.02) but does NOT improve generalisation, because patients
    differ in where they were implanted and this removes real anatomy with the nuisance."""
    Y = X.copy()
    for p in np.unique(patient):
        m = patient == p
        Y[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-9)
    return Y


#: named pipelines for the consensus, in the Lowe & Schall (2018) sense
PIPELINES: Dict[str, str] = {
    "raw": "as shipped",
    "unit_norm": "shape only (unit length per electrode)",
    "patient_z": "per-patient z-score",
    "shape_patient_z": "per-patient z, then unit length",
}


def apply_pipeline(X: np.ndarray, name: str, patient: Optional[np.ndarray] = None) -> np.ndarray:
    if name == "raw":
        return X
    if name == "unit_norm":
        return unit_norm(X)
    if name == "patient_z":
        return per_patient_z(X, patient)
    if name == "shape_patient_z":
        return unit_norm(per_patient_z(X, patient))
    raise ValueError(f"unknown pipeline {name!r}; known: {sorted(PIPELINES)}")


# ── convex NMF ───────────────────────────────────────────────────────────────
def convex_nmf(X: np.ndarray, k: int, *, n_iter: int = 300, tol: float = 1e-6,
               random_state: int = 0, verbose: bool = False):
    """Convex NMF of a SIGNED matrix.  X (n_samples x n_features) ~= G @ (W.T @ X).

    Returns (W, G, components) with
        W  (n_samples x k)  non-negative; component j = W[:, j] . X, a convex-ish
                            combination of real electrodes, so it is itself a response
                            profile you can plot
        G  (n_samples x k)  non-negative loadings — how much of each component each
                            electrode expresses. THIS is the graded membership.
        components (k x n_features) = (W.T @ X) / colsum(W), the profiles themselves

    Multiplicative updates from Ding, Li & Jordan (2010), IEEE TPAMI 32(1):45-55,
    "Convex and Semi-Nonnegative Matrix Factorizations". The A+/A- split of the Gram
    matrix is what lets the data carry negative values.
    """
    rng = np.random.default_rng(random_state)
    Xa = np.asarray(X, dtype=np.float64)
    n = Xa.shape[0]

    A = Xa @ Xa.T                                  # n x n Gram
    Ap = (np.abs(A) + A) / 2.0
    An = (np.abs(A) - A) / 2.0

    # k-means style init is what the paper recommends; a random positive start also works
    from sklearn.cluster import KMeans
    lab = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(Xa)
    G = np.zeros((n, k)) + 0.2
    G[np.arange(n), lab] = 1.0
    W = G / G.sum(0, keepdims=True)
    G = G + 0.2 * rng.random((n, k))

    prev = np.inf
    for it in range(n_iter):
        GtG = G.T @ G
        W *= np.sqrt((Ap @ G + An @ W @ GtG) / np.maximum(An @ G + Ap @ W @ GtG, 1e-12))
        WtAp, WtAn = W.T @ Ap, W.T @ An
        G *= np.sqrt((Ap @ W + G @ (W.T @ An @ W)) /
                     np.maximum(An @ W + G @ (W.T @ Ap @ W), 1e-12))
        if it % 10 == 0 or it == n_iter - 1:
            err = np.linalg.norm(Xa - G @ (W.T @ Xa), "fro")
            if verbose:
                print(f"    iter {it:4d}  ||X - GW'X||_F = {err:.4f}")
            if abs(prev - err) / max(prev, 1e-12) < tol:
                break
            prev = err

    # Normalise each column of W to sum to 1 so the component really is a convex
    # combination of observed electrodes. comp is then a weighted AVERAGE of real
    # responses and keeps its dB units, which is what makes it plottable as a profile.
    Wn = W / np.maximum(W.sum(0, keepdims=True), 1e-12)
    comp = Wn.T @ Xa
    return Wn, G, comp


def reconstruct(X: np.ndarray, W: np.ndarray, G: np.ndarray) -> np.ndarray:
    return G @ (W.T @ X)


def cv_rank_curve(X: np.ndarray, ks: Sequence[int], *, n_folds: int = 5,
                  random_state: int = 0, n_iter: int = 200) -> pd.DataFrame:
    """Held-out reconstruction as a function of component count.

    Electrodes are split into folds; the decomposition is fit WITHOUT a fold, then that
    fold's electrodes are projected onto the fitted components by non-negative least
    squares and their reconstruction error measured. This is the criterion
    Norman-Haignere et al. (2022) used, and unlike silhouette it can be wrong: a rank that
    overfits gets a worse score rather than a better one.
    """
    from scipy.optimize import nnls
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(len(X))
    folds = np.array_split(idx, n_folds)
    rows = []
    for k in ks:
        for fi, te in enumerate(folds):
            tr = np.setdiff1d(idx, te)
            W, G, comp = convex_nmf(X[tr], k, random_state=random_state, n_iter=n_iter)
            B = comp.T                                     # n_features x k
            err = num = 0.0
            for i in te:
                g, _ = nnls(B, X[i])
                err += float(((X[i] - B @ g) ** 2).sum())
                num += float((X[i] ** 2).sum())
            rows.append(dict(k=int(k), fold=fi,
                             held_out_err=err / max(num, 1e-12),
                             var_explained=1.0 - err / max(num, 1e-12)))
    return pd.DataFrame(rows)


# ── consensus across preprocessing pipelines ─────────────────────────────────
def pipeline_consensus(X: np.ndarray, k: int, *, patient: Optional[np.ndarray] = None,
                       pipelines: Optional[Iterable[str]] = None,
                       seeds: Sequence[int] = (0, 1, 2, 3, 4)) -> np.ndarray:
    """Co-association matrix over electrodes, pooled across preprocessing pipelines.

    C[i, j] = fraction of (pipeline, seed) fits in which electrodes i and j land in the
    same cluster. Pairs that co-cluster only under one pipeline wash out; pairs that
    survive every defensible choice are the ones worth naming. Lowe & Schall (2018) did
    this over 48 pipelines to avoid defending any single one.
    """
    from sklearn.cluster import KMeans
    names = list(pipelines) if pipelines is not None else list(PIPELINES)
    n = len(X)
    C = np.zeros((n, n), dtype=np.float32)
    runs = 0
    for nm in names:
        Xp = apply_pipeline(X, nm, patient)
        for s in seeds:
            y = KMeans(n_clusters=k, n_init=10, random_state=s).fit_predict(Xp)
            C += (y[:, None] == y[None, :])
            runs += 1
    return C / max(runs, 1)


def consensus_labels(C: np.ndarray, k: int) -> np.ndarray:
    """Cluster the co-association matrix itself (average-linkage on 1 - C)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    return fcluster(Z, t=k, criterion="maxclust") - 1


# ── stability, always against a matched null ─────────────────────────────────
def lopo_stability(X: np.ndarray, groups: np.ndarray, k: int, *,
                   random_state: int = 42, n_init: int = 20) -> pd.DataFrame:
    """Leave-one-group-out agreement with the full-cohort solution, per group."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    base = KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit(X)
    rows = []
    for g in np.unique(groups):
        keep = groups != g
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit(X[keep])
        held = base.predict(X[keep])
        # ARI saturates when a held-out set is homogeneous: both-constant returns 1.0 and
        # one-constant returns 0.0 regardless of agreement. Flag it rather than average it.
        degenerate = len(np.unique(held)) < 2 or len(np.unique(km.labels_)) < 2
        rows.append(dict(group=str(g), n=int((~keep).sum()),
                         ari=float(adjusted_rand_score(held, km.labels_)),
                         degenerate=bool(degenerate)))
    return pd.DataFrame(rows)


def pseudo_group_null(X: np.ndarray, groups: np.ndarray, k: int, *, n_rep: int = 10,
                      random_state: int = 0, **kw) -> pd.DataFrame:
    """The same test with FAKE groups of identical sizes.

    Without this a leave-one-out result cannot be read at all: on this cohort the real
    minimum of 0.448 sits inside a null of 0.459 +/- 0.019, so what looked like patient
    fragility was just the effect of removing that many electrodes.
    """
    rng = np.random.default_rng(random_state)
    sizes = pd.Series(groups).value_counts().to_numpy()
    out = []
    for rep in range(n_rep):
        idx = rng.permutation(len(X))
        fake = np.empty(len(X), dtype=object)
        s = 0
        for i, sz in enumerate(sizes):
            fake[idx[s:s + sz]] = f"F{i}"
            s += sz
        d = lopo_stability(X, fake, k, **kw)
        out.append(dict(rep=rep, mean=float(d["ari"].mean()), min=float(d["ari"].min())))
    return pd.DataFrame(out)


def spatial_coherence(labels: np.ndarray, xyz: np.ndarray, *, k_nn: int = 10,
                      n_shuffle: int = 20, random_state: int = 0) -> tuple:
    """Of an electrode's k nearest neighbours in fsaverage space, how many share its
    label — divided by the same quantity under a label shuffle. 1.0 means no spatial
    structure. Chance-correcting matters because a solution with one dominant cluster
    scores high on the raw version for free."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(random_state)
    ok = ~np.isnan(xyz).any(1)
    P, y = xyz[ok], np.asarray(labels)[ok]
    nb = cKDTree(P).query(P, k=k_nn + 1)[1][:, 1:]
    obs = float((y[nb] == y[:, None]).mean())
    null = float(np.mean([(lambda s: (s[nb] == s[:, None]).mean())(rng.permutation(y))
                          for _ in range(n_shuffle)]))
    return obs, obs / max(null, 1e-12)


# ── graded display ───────────────────────────────────────────────────────────
def soft_rgb(Gn: np.ndarray, palette: Sequence[Sequence[float]], *,
             grey=(0.55, 0.55, 0.55), gamma: float = 1.0) -> np.ndarray:
    """Colour by graded membership: hue = leading component, saturation = how much it
    leads. Contested territory desaturates toward grey instead of being handed to an
    arbitrary winner, which is the honest rendering when 19% of sampled cortex has no
    component above half the local mixture."""
    Gn = np.asarray(Gn, dtype=float)
    s = Gn.sum(1, keepdims=True)
    Gn = Gn / np.where(s > 0, s, 1)
    lead = Gn.argmax(1)
    top = Gn.max(1)
    second = np.sort(Gn, axis=1)[:, -2] if Gn.shape[1] > 1 else np.zeros(len(Gn))
    conf = np.clip((top - second) / np.maximum(top, 1e-9), 0, 1) ** gamma
    base = np.asarray(palette, dtype=float)[lead]
    return (1 - conf)[:, None] * np.asarray(grey, dtype=float) + conf[:, None] * base


def mixture_summary(Gn: np.ndarray, *, hi: float = 0.8, lo: float = 0.5) -> dict:
    """How graded is this decomposition? Reports the split between territory one
    component owns, contested territory, and territory with no majority at all."""
    Gn = np.asarray(Gn, dtype=float)
    s = Gn.sum(1, keepdims=True)
    P = Gn / np.where(s > 0, s, 1)
    top = P.max(1)
    return {
        "n": int(len(P)),
        "frac_dominant": float((top >= hi).mean()),
        "frac_contested": float(((top < hi) & (top >= lo)).mean()),
        "frac_no_majority": float((top < lo).mean()),
        "median_top_weight": float(np.median(top)),
    }
