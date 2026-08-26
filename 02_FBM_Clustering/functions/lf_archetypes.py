#!/usr/bin/env python3
"""
lf_archetypes.py - archetypal analysis (Cutler & Breiman 1994, Technometrics 36(4)).

WHY THIS AND NOT ANOTHER DECOMPOSITION
Convex NMF already gives graded membership, and its components are convex combinations
of real electrodes. Archetypal analysis adds ONE constraint on top of that, and the
constraint is the whole point:

    convex NMF     X ~= G (W'X)       W >= 0, G >= 0
    archetypes     X ~= A (B X)       A >= 0 AND rows sum to 1
                                      B >= 0 AND rows sum to 1

Forcing BOTH sides onto the probability simplex moves the components from the INTERIOR
of the data to its CONVEX HULL. A convex-NMF component is a weighted average, and an
average of a mixed population looks like the population - so its components tend to be
watered-down versions of the typical response. An archetype is an EXTREME: the purest
auditory response in the cohort, the purest reading response, and every electrode is
then written as a mixture of those extremes with weights that sum to one.

That makes each electrode's loading vector read directly as proportions - "60% auditory
archetype, 40% reading archetype" - which is a sentence about the electrode rather than
about the fit. Convex NMF's loadings have to be renormalised before they mean that, and
the renormalisation is a step where the meaning can quietly change.

WHAT IT IS NOT. Archetypes are not cluster centres and they are not more "correct" than
centroids. They sit at the edge of the data, so no electrode need be near one, and an
archetype can be driven by a handful of extreme electrodes. That is a real risk and
`archetype_support()` below is here to measure it rather than to hope.

METHOD. Alternating projected gradient with exact Euclidean projection onto the simplex
(Duchi et al. 2008), initialised by FurthestSum (Morup & Hansen 2012). Both steps use a
Lipschitz step size with backtracking, so the RSS is monotone by construction and a run
that does not decrease is a bug rather than a tuning problem.
"""
from __future__ import annotations

import numpy as np

__all__ = ["archetypal_analysis", "project_simplex", "furthest_sum",
           "archetype_support", "explained_variance"]


def project_simplex(V: np.ndarray) -> np.ndarray:
    """Row-wise Euclidean projection onto {x >= 0, sum(x) = 1}.

    Duchi, Shalev-Shwartz, Singer & Chandra (2008), exact and O(d log d) per row.
    Clipping at zero and renormalising is NOT this and gives a different point.
    """
    V = np.atleast_2d(np.asarray(V, dtype=np.float64))
    n, d = V.shape
    U = np.sort(V, axis=1)[:, ::-1]
    css = np.cumsum(U, axis=1) - 1.0
    ind = np.arange(1, d + 1)
    cond = (U - css / ind) > 0
    # index of the last True in each row
    rho = d - 1 - np.argmax(cond[:, ::-1], axis=1)
    theta = css[np.arange(n), rho] / (rho + 1.0)
    return np.maximum(V - theta[:, None], 0.0)


def furthest_sum(X: np.ndarray, k: int, rng) -> np.ndarray:
    """k row indices spread to the edges of the cloud (Morup & Hansen 2012).

    Random initialisation lands archetypes in the middle, where the gradient is flat
    and they stay. Starting at the extremes is most of what makes this converge.
    """
    n = X.shape[0]
    start = int(rng.integers(n))
    chosen = [start]
    d = np.linalg.norm(X - X[start], axis=1)
    while len(chosen) < k:
        j = int(np.argmax(d))
        if j in chosen:                       # degenerate cloud: take anything unused
            left = np.setdiff1d(np.arange(n), chosen)
            if left.size == 0:
                break
            j = int(left[0])
        chosen.append(j)
        d = d + np.linalg.norm(X - X[j], axis=1)
        d[chosen] = -np.inf
    return np.array(chosen[:k], dtype=int)


def archetypal_analysis(X: np.ndarray, k: int, *, n_iter: int = 300,
                        tol: float = 1e-7, random_state: int = 0,
                        verbose: bool = False):
    """X (n x p) ~= A @ Z with Z = B @ X, both A and B row-stochastic.

    Returns (A, B, Z, info) with

        A  (n x k)  each row sums to 1 - the electrode as a MIXTURE OF ARCHETYPES.
                    This is the graded membership, and it needs no renormalising.
        B  (k x n)  each row sums to 1 - the archetype as a MIXTURE OF ELECTRODES,
                    so an archetype is a real response profile you can plot and can
                    trace back to the electrodes that built it.
        Z  (k x p)  = B @ X, the archetypes themselves.
        info        dict with rss, explained variance, per-iteration rss, whether it
                    converged, and how concentrated each archetype's support is.
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    if not (1 <= k <= n):
        raise ValueError(f"k must be in [1, {n}], got {k}")
    rng = np.random.default_rng(random_state)

    # ── init ────────────────────────────────────────────────────────────────
    seed_idx = furthest_sum(X, k, rng)
    B = np.zeros((k, n))
    B[np.arange(k), seed_idx] = 1.0
    Z = B @ X
    A = project_simplex(rng.random((n, k)) + 1e-3)

    tss = float(((X - X.mean(0)) ** 2).sum())
    hist = []
    prev = np.inf
    converged = False

    # SPECTRAL norms, not Frobenius. The Lipschitz constant of the gradient is the
    # largest SINGULAR VALUE squared; ||.||_F over-estimates it by up to a factor of
    # rank, which makes 1/L a step small enough that the first version crawled and
    # left a triangle vertex 4.2 units from where it belonged. ||X||_2 is computed
    # once because X never changes.
    nX2 = float(np.linalg.norm(X, 2)) ** 2
    stepA = stepB = None

    for it in range(n_iter):
        # ── A step: each electrode's mixture, archetypes held ───────────────
        Z = B @ X
        R = X - A @ Z
        gA = -(R @ Z.T)
        cur = float((R * R).sum())
        L = max(float(np.linalg.norm(Z, 2)) ** 2, 1e-12)
        # start from twice the step that worked last time - halving alone can only
        # ever shrink, so a single hard iteration would throttle every one after it
        step = 2.0 * stepA if stepA is not None else 1.0 / L
        for _ in range(40):
            A_new = project_simplex(A - step * gA)
            r = X - A_new @ Z
            e = float((r * r).sum())
            if e <= cur + 1e-12:
                A, R, cur, stepA = A_new, r, e, step
                break
            step *= 0.5

        # ── B step: what each archetype is made of, mixtures held ───────────
        gB = -((A.T @ R) @ X.T)
        L = max(float(np.linalg.norm(A, 2)) ** 2 * nX2, 1e-12)
        step = 2.0 * stepB if stepB is not None else 1.0 / L
        for _ in range(40):
            B_new = project_simplex(B - step * gB)
            r = X - A @ (B_new @ X)
            e = float((r * r).sum())
            if e <= cur + 1e-12:
                B, cur, stepB = B_new, e, step
                break
            step *= 0.5

        hist.append(cur)
        if verbose and (it % 25 == 0 or it == n_iter - 1):
            print(f"    it {it:>4}  rss {cur:.6g}  var {1 - cur / tss:.4f}")
        if prev < np.inf and abs(prev - cur) <= tol * max(prev, 1e-12):
            converged = True
            prev = cur
            break
        prev = cur

    Z = B @ X
    # MONOTONE BY CONSTRUCTION - both steps backtrack until the RSS does not rise. If
    # this ever trips, the projection or a gradient is wrong, not the step size.
    h = np.asarray(hist)
    if h.size > 1 and np.any(np.diff(h) > 1e-6 * max(abs(h[0]), 1.0)):
        raise RuntimeError("archetypal_analysis: RSS increased - this cannot happen "
                           "with backtracking and means a gradient is wrong")

    info = dict(rss=float(prev), var_explained=float(1.0 - prev / tss) if tss > 0 else 0.0,
                n_iter=len(hist), converged=bool(converged), rss_history=h.tolist(),
                seed_rows=seed_idx.tolist(), k=int(k), n=int(n), p=int(p))
    info.update(archetype_support(B))
    return A, B, Z, info


def archetype_support(B: np.ndarray, *, top: int = 10) -> dict:
    """How few electrodes each archetype actually rests on.

    An archetype sitting on the hull can be one outlier wearing a hat. The effective
    support - exp of the entropy of its weights - says how many electrodes it really
    averages: 1.0 means a single electrode IS the archetype, and a large number means
    it is a broad edge of the cloud rather than a point on it.
    """
    B = np.asarray(B, dtype=np.float64)
    W = B / np.maximum(B.sum(1, keepdims=True), 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(W > 0, W * np.log(W), 0.0), axis=1)
    eff = np.exp(ent)
    return dict(effective_support=eff.tolist(),
                min_effective_support=float(eff.min()),
                top_weight=W.max(1).tolist(),
                mass_in_top=[float(np.sort(w)[::-1][:top].sum()) for w in W])


def explained_variance(X: np.ndarray, A: np.ndarray, Z: np.ndarray) -> float:
    """Fraction of total variance the reconstruction accounts for."""
    X = np.asarray(X, dtype=np.float64)
    rss = float(((X - A @ Z) ** 2).sum())
    tss = float(((X - X.mean(0)) ** 2).sum())
    return float(1.0 - rss / tss) if tss > 0 else 0.0
