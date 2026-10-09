"""Acquisition scoring and candidate selection for Bayesian optimisation."""

import logging

import numpy as np
import pandas as pd
from ase import Atoms
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..._numeric_defaults import (
    ACQUISITION_SIGMA_FLOOR,
    ACQUISITION_XI_DEFAULT,
    DEFAULT_SEED,
)
from ...config import (
    BO_INITIAL_SAMPLING_OPTIONS,
    AdsorptionConfig,
)
from ...models import PlacementDescriptor, PlacementSpec
from ...placement import generators as placement_generators
from ...placement.site_context import SiteContext
from ..features import FEATURE_ABS_COLUMNS, extract_features
from ..schema import PlacementRecord
from .surrogate import _median_nn_lengthscale, predict_with_uncertainty
from .types import AcquisitionType, InitialSamplingType

logger = logging.getLogger(__name__)


def lcb_scores(
    mu: np.ndarray,
    sigma: np.ndarray,
    kappa: float = 1.96,
) -> np.ndarray:
    """Lower confidence bound for minimisation: mu - kappa * sigma.

    Lower scores are better (more promising candidates).

    Parameters
    ----------
    mu
        Predicted mean values.
    sigma
        Predicted standard deviations.
    kappa
        Exploration-exploitation trade-off parameter.
    """
    return mu - kappa * sigma


def ei_scores(
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
    xi: float = ACQUISITION_XI_DEFAULT,
) -> np.ndarray:
    """Compute expected improvement scores for minimisation.

    EI = E[max(0, f_best - Y)] under Gaussian Y ~ N(mu, sigma^2). Higher EI is better.
    When *every* ``sigma`` is (near) zero, ranks by ``-mu`` so the pool does not
    collapse to an arbitrary tied ordering of zeros. In a mixed-``sigma`` pool,
    zero-``sigma`` rows use the analytic limit ``max(f_best - mu - xi, 0)`` so
    they stay on the same scale as finite-``sigma`` EI.

    Parameters
    ----------
    mu
        Predicted mean values.
    sigma
        Predicted standard deviations.
    f_best
        Best observed function value so far.
    xi
        Small jitter to encourage exploration.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    imp = f_best - mu - xi
    finite = sigma > ACQUISITION_SIGMA_FLOOR
    if not np.any(finite):
        return -mu
    z = np.divide(
        imp,
        sigma,
        out=np.zeros_like(imp, dtype=float),
        where=finite,
    )
    ei = imp * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
    return np.where(finite, ei, np.maximum(imp, 0.0))


def pi_scores(
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
    xi: float = ACQUISITION_XI_DEFAULT,
) -> np.ndarray:
    """Probability of Improvement for minimisation: P(Y < f_best - xi).

    Higher PI is better. When every ``sigma`` is zero, ranks by ``-mu`` (same
    rationale as EI). In a mixed-``sigma`` pool, zero-``sigma`` rows use the
    analytic step ``1`` if ``mu < f_best - xi`` else ``0``.

    Parameters
    ----------
    mu
        Predicted mean values.
    sigma
        Predicted standard deviations.
    f_best
        Best observed function value so far.
    xi
        Small jitter to encourage exploration.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    finite = sigma > ACQUISITION_SIGMA_FLOOR
    if not np.any(finite):
        return -mu
    z = np.divide(
        f_best - xi - mu,
        sigma,
        out=np.zeros_like(mu, dtype=float),
        where=finite,
    )
    pi = stats.norm.cdf(z)
    degenerate = np.where(mu < f_best - xi, 1.0, 0.0)
    return np.clip(np.where(finite, pi, degenerate), 0.0, 1.0)


def _farthest_point_indices(
    features: pd.DataFrame | np.ndarray,
    n_pick: int,
    rng: np.random.RandomState,
) -> list[int]:
    """Greedy farthest-point sampling in standardized feature space."""
    matrix = (
        features.to_numpy(dtype=float)
        if isinstance(features, pd.DataFrame)
        else np.asarray(features, dtype=float)
    )
    n_pool = matrix.shape[0]
    if n_pick >= n_pool:
        return list(range(n_pool))
    scaled = StandardScaler().fit_transform(matrix)
    first = int(rng.randint(n_pool))
    chosen = [first]
    min_dists = np.linalg.norm(scaled - scaled[first], axis=1)
    for _ in range(n_pick - 1):
        min_dists[chosen] = -1.0
        nxt = int(np.argmax(min_dists))
        chosen.append(nxt)
        min_dists = np.minimum(min_dists, np.linalg.norm(scaled - scaled[nxt], axis=1))
    return chosen


def _stratified_conformer_indices(
    features: pd.DataFrame,
    n_pick: int,
    rng: np.random.RandomState,
) -> list[int]:
    """Round-robin across conformer_index groups, then spread-fill remainder."""
    if "conformer_index" not in features.columns:
        return _farthest_point_indices(features, n_pick, rng)

    groups: dict[int, list[int]] = {}
    for i, value in enumerate(features["conformer_index"].astype(int)):
        groups.setdefault(int(value), []).append(i)
    for members in groups.values():
        rng.shuffle(members)
    keys = list(groups.keys())
    rng.shuffle(keys)

    chosen: list[int] = []
    while len(chosen) < n_pick:
        progressed = False
        for key in keys:
            if groups[key]:
                chosen.append(groups[key].pop())
                progressed = True
                if len(chosen) >= n_pick:
                    break
        if not progressed:
            break

    if len(chosen) < n_pick:
        chosen_set = set(chosen)
        remaining = [i for i in range(len(features)) if i not in chosen_set]
        if remaining:
            local = _farthest_point_indices(
                features.iloc[remaining],
                min(n_pick - len(chosen), len(remaining)),
                rng,
            )
            chosen.extend(remaining[i] for i in local)
    return chosen


def select_initial_bo_indices(
    candidate_features: pd.DataFrame,
    n_initial: int,
    *,
    sampling: InitialSamplingType = "spread_xyz",
    random_state: int = DEFAULT_SEED,
) -> list[int]:
    """Pick initial BO pool positions before any energy evaluations.

    Strategies:
    - ``random``: uniform without replacement
    - ``spread``: farthest-point on all geometry-aware features
    - ``spread_xyz``: farthest-point on Cartesian COM columns (``x``, ``y``,
      ``z`` from ``x_abs``/``y_abs``/``z_abs``) only
    - ``stratified``: round-robin across conformer_index, then spread-fill

    Parameters
    ----------
    candidate_features
        DataFrame of candidate placement features (FEATURE_NAMES columns).
    n_initial
        Number of initial samples to select.
    sampling
        Sampling strategy name.
    random_state
        Random seed for reproducibility.
    """
    if sampling not in BO_INITIAL_SAMPLING_OPTIONS:
        allowed = ", ".join(repr(item) for item in BO_INITIAL_SAMPLING_OPTIONS)
        raise ValueError(f"sampling must be one of {allowed}, got {sampling!r}")
    n_pool = len(candidate_features)
    n_pick = min(int(n_initial), n_pool)
    if n_pick <= 0:
        return []
    if n_pick >= n_pool:
        return list(range(n_pool))
    rng = np.random.RandomState(random_state)
    if sampling == "random":
        return rng.choice(n_pool, size=n_pick, replace=False).tolist()
    if sampling == "spread":
        return _farthest_point_indices(candidate_features, n_pick, rng)
    if sampling == "spread_xyz":
        position_cols = [
            c for c in FEATURE_ABS_COLUMNS if c in candidate_features.columns
        ]
        subset = (
            candidate_features[position_cols] if position_cols else candidate_features
        )
        return _farthest_point_indices(subset, n_pick, rng)
    return _stratified_conformer_indices(candidate_features, n_pick, rng)


def select_candidates(
    scores: np.ndarray,
    batch_size: int,
    evaluated_indices: set[int] | None = None,
    *,
    higher_is_better: bool = False,
) -> list[int]:
    """Return up to *batch_size* best candidate indices by rank order.

    For minimisation objectives (default), ranks by ascending *scores*.
    For *higher_is_better* (e.g. EI, PI), ranks by descending *scores*.
    Indices in *evaluated_indices* are excluded.

    Parameters
    ----------
    scores
        Acquisition scores for each candidate.
    batch_size
        Number of candidates to select.
    evaluated_indices
        Set of already-evaluated indices to exclude.
    higher_is_better
        If True, rank by descending scores.
    """
    s = np.asarray(scores, dtype=float).ravel()
    order = np.argsort(-s) if higher_is_better else np.argsort(s)
    selected: list[int] = []
    for idx in order:
        if evaluated_indices is not None and int(idx) in evaluated_indices:
            continue
        selected.append(int(idx))
        if len(selected) >= batch_size:
            break
    return selected


def select_candidates_batch_diverse(
    scores: np.ndarray,
    features: pd.DataFrame | np.ndarray,
    batch_size: int,
    evaluated_indices: set[int] | None = None,
    *,
    higher_is_better: bool = False,
    scaled_features: np.ndarray | None = None,
) -> list[int]:
    """Greedy batch selection with soft local penalization in feature space.

    Picks the best remaining score, then down-weights (or up-penalizes for
    minimisation) candidates near the chosen point so a single batch does not
    collapse onto a tight cluster of near-duplicates.

    Parameters
    ----------
    scores
        Acquisition scores for each candidate.
    features
        Feature matrix for diversity computation.
    batch_size
        Number of candidates to select.
    evaluated_indices
        Set of already-evaluated indices to exclude.
    higher_is_better
        If True, rank by descending scores.
    scaled_features
        Optional pre-standardized feature matrix (same row order as *features*).
        When provided, skips re-fitting ``StandardScaler`` every acquisition round.
    """
    s = np.asarray(scores, dtype=float).copy().ravel()
    matrix = (
        features.to_numpy(dtype=float)
        if isinstance(features, pd.DataFrame)
        else np.asarray(features, dtype=float)
    )
    n = len(s)
    if matrix.shape[0] != n:
        raise ValueError(
            f"features rows ({matrix.shape[0]}) must match scores length ({n})"
        )
    blocked: set[int] = set(evaluated_indices or ())
    available = [i for i in range(n) if i not in blocked]
    if not available or batch_size <= 0:
        return []
    if batch_size == 1 or len(available) == 1:
        return select_candidates(
            s,
            batch_size,
            evaluated_indices=blocked,
            higher_is_better=higher_is_better,
        )

    if scaled_features is not None:
        scaled = np.asarray(scaled_features, dtype=float)
        if scaled.shape[0] != n:
            raise ValueError(
                f"scaled_features rows ({scaled.shape[0]}) must match scores length ({n})"
            )
    else:
        scaled = StandardScaler().fit_transform(matrix)
    lengthscale = _median_nn_lengthscale(scaled[available], use_kdtree=True)
    finite = s[np.isfinite(s)]
    strength = float(np.std(finite)) if finite.size > 1 else 1.0
    strength = max(strength, 1e-3)

    chosen: list[int] = []
    remaining = set(available)
    working = s.copy()
    for _ in range(min(batch_size, len(available))):
        if not remaining:
            break
        cand = np.array(sorted(remaining), dtype=int)
        vals = working[cand]
        pick_local = int(np.argmax(vals) if higher_is_better else np.argmin(vals))
        pick = int(cand[pick_local])
        chosen.append(pick)
        remaining.remove(pick)
        if not remaining:
            break
        rem = np.array(sorted(remaining), dtype=int)
        dists = cdist(scaled[pick : pick + 1], scaled[rem])[0]
        near = np.exp(-0.5 * np.square(dists / lengthscale))
        if higher_is_better:
            working[rem] = working[rem] - strength * near
        else:
            working[rem] = working[rem] + strength * near
    return chosen


def build_spec_features_geometry_aware(
    specs: list[PlacementSpec],
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    *,
    smiles: str | None = None,
    molecule: str = "",
    surface_id: str = "",
    site_context: SiteContext | None = None,
    slab_for_sites: Atoms | None = None,
    materialization_cache: dict[int, tuple[Atoms, PlacementDescriptor]] | None = None,
) -> tuple[pd.DataFrame, list[int]]:
    """Extract geometry-aware features from specs via resolved deterministic poses.

    Returns ``(features_df, valid_indices)`` where *valid_indices* maps each
    row in the DataFrame back to the position of the corresponding spec in
    *specs*.  Specs that cannot produce a valid placement are skipped; a single
    INFO line summarizes how many were skipped when that count is positive.

    When *materialization_cache* is provided, successful
    ``(adsorbate, descriptor)`` pairs are stored under ``placement_index`` for
    reuse by evaluate paths.

    Parameters
    ----------
    specs
        List of placement specifications.
    conformers
        List of conformer structures.
    slab
        Surface slab atoms.
    config
        Adsorption configuration.
    smiles
        Optional SMILES string for the molecule.
    molecule
        Molecule name.
    surface_id
        Surface identifier.
    site_context
        Optional site context for placement.
    slab_for_sites
        Optional alternate slab for site detection.
    materialization_cache
        Optional cache for materialized placements.
    """
    rows: list[dict[str, float]] = []
    valid_indices: list[int] = []
    if not conformers:
        raise ValueError(
            "build_spec_features_geometry_aware requires at least one conformer"
        )

    generated = placement_generators.generate_placements_from_specs(
        specs,
        conformers,
        slab,
        config,
        smiles=smiles,
        site_context=site_context,
        slab_for_sites=slab_for_sites,
        materialization_cache=materialization_cache,
    )
    for i, (spec, (result, _fail_reason)) in enumerate(
        zip(specs, generated, strict=True)
    ):
        if result is None:
            logger.debug(
                "Skipping spec placement_index=%d: no valid placement",
                spec.placement_index,
            )
            continue
        adsorbate, descriptor = result
        if materialization_cache is not None:
            materialization_cache[int(descriptor.placement_index)] = (
                adsorbate.copy(),
                descriptor,
            )
        record = PlacementRecord.from_descriptor(
            descriptor,
            molecule=molecule,
            smiles=smiles or "",
            surface_id=surface_id,
            config=config,
        )
        rows.append(extract_features(record))
        valid_indices.append(i)

    n_skip = len(specs) - len(rows)
    if n_skip > 0:
        logger.info(
            "Build_spec_features_geometry_aware: skipped %d/%d specs (no valid placement)",
            n_skip,
            len(specs),
        )
    return pd.DataFrame(rows), valid_indices


def score_and_select(
    model: Pipeline,
    candidate_features: pd.DataFrame,
    batch_size: int,
    kappa: float = 1.96,
    evaluated_indices: set[int] | None = None,
    acquisition: AcquisitionType = "lcb",
    f_best: float | None = None,
    scaled_features: np.ndarray | None = None,
    n_jobs: int | None = None,
    sigma_scale: np.ndarray | None = None,
) -> list[int]:
    """Score candidates with the given acquisition and select the top batch.

    For ``acquisition="lcb"`` uses LCB (mu - kappa * sigma); lower is better.
    For ``acquisition="ei"`` and ``acquisition="pi"`` uses EI or PI; higher is better.
    ``f_best`` is required for EI and PI (current best observed value for minimisation).
    With near-zero ``sigma`` (unfitted linear models), EI/PI fall back to
    ranking by ``-mu`` so the pool does not collapse to an arbitrary tie.
    Batches use soft local penalization in feature space so picks are diverse.

    Parameters
    ----------
    model
        Fitted surrogate pipeline.
    candidate_features
        DataFrame of candidate placement features.
    batch_size
        Number of candidates to select.
    kappa
        Exploration parameter for LCB acquisition.
    evaluated_indices
        Set of already-evaluated indices to exclude.
    acquisition
        Acquisition function type ("lcb", "ei", or "pi").
    f_best
        Best observed value (required for EI and PI).
    scaled_features
        Optional pre-standardized pool matrix for diversity (avoids re-scaling
        every acquisition round).
    n_jobs
        CPU workers for per-tree uncertainty prediction (joblib convention);
        forwarded to :func:`predict_with_uncertainty`.
    sigma_scale
        Optional per-candidate multiplier on predictive ``sigma`` (e.g.
        occupancy inflation near committed adsorbates). Must match the number
        of candidate rows when set.
    """
    mu, sigma = predict_with_uncertainty(model, candidate_features, n_jobs=n_jobs)
    if sigma_scale is not None:
        scale = np.asarray(sigma_scale, dtype=float).ravel()
        if scale.shape[0] != sigma.shape[0]:
            raise ValueError(
                f"sigma_scale length ({scale.shape[0]}) must match "
                f"candidates ({sigma.shape[0]})"
            )
        sigma = sigma * scale
    if acquisition == "lcb":
        scores = lcb_scores(mu, sigma, kappa=kappa)
        return select_candidates_batch_diverse(
            scores,
            candidate_features,
            batch_size,
            evaluated_indices=evaluated_indices,
            higher_is_better=False,
            scaled_features=scaled_features,
        )
    if acquisition in ("ei", "pi"):
        if f_best is None:
            raise ValueError("f_best is required for EI and PI acquisition")
        if acquisition == "ei":
            scores = ei_scores(mu, sigma, f_best=f_best)
        else:
            scores = pi_scores(mu, sigma, f_best=f_best)
        return select_candidates_batch_diverse(
            scores,
            candidate_features,
            batch_size,
            evaluated_indices=evaluated_indices,
            higher_is_better=True,
            scaled_features=scaled_features,
        )
    raise ValueError(f"Unknown acquisition: {acquisition!r}")


def splice_exploration_picks(
    rng: np.random.RandomState,
    chosen: list[int],
    *,
    pool_size: int,
    evaluated_indices: set[int] | None = None,
    exploration_fraction: float = 0.0,
) -> list[int]:
    """Replace the tail of an acquisition batch with random unevaluated picks.

    Single source of truth for the exploration splice used by both the
    production BO loop and offline replay/benchmarks: ``ceil(len(chosen) *
    fraction)`` random draws from pool positions that are neither evaluated nor
    already picked, appended after (at least one) acquisition pick so a batch
    of size 1 never goes fully random.

    Returns a new list; *chosen* is not modified.
    """
    frac = float(exploration_fraction)
    if frac <= 0.0 or not chosen:
        return list(chosen)
    blocked = set(evaluated_indices or ())
    chosen_set = set(chosen)
    explore_n = int(np.ceil(len(chosen) * frac))
    available = [
        i for i in range(int(pool_size)) if i not in blocked and i not in chosen_set
    ]
    if not available:
        return list(chosen)
    # Leave >=1 acquisition pick.
    explore_n = min(explore_n, len(available), max(0, len(chosen) - 1))
    if explore_n <= 0:
        return list(chosen)
    random_picks = rng.choice(available, size=explore_n, replace=False).tolist()
    return chosen[: len(chosen) - explore_n] + random_picks
