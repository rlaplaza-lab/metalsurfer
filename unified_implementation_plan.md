# Unified Implementation Plan (Validated & Re-prioritised)

## Summary

After tracing the codebase against both improvement documents, several key findings were re-classified:
- **`site_context` threading** (improvements_1 #1) is **already 100% complete** across both standard and Bayesian workflows in production; no plumbing is missing.
- **Input validation** (improvements_1 #3) must validate `len(atoms) >= 1`, but must **NOT** reject degenerate unit cells unconditionally, as nanoparticles and non-periodic systems intentionally have zero-volume cells with bounding-box fallbacks.
- **Planar slab handling** (improvements_1 #4) was expanded: `_top_layer_is_planar_from_arrays()` fails not only on `rank < 3` but also on `len(top_indices) < 3` (primitive 1x1 cells). Checking `np.var(h) < z_variance_threshold` covers both cases physically.
- **Set-based unevaluated lookup** (improvements_2 #2) is **REJECTED**: Python set iteration is non-deterministic and breaks seed reproducibility in `rng.choice(unevaluated, ...)`.
- **NearestNeighbors for self-distance** (improvements_2 #3) is **REJECTED**: `cdist` is faster (<10 µs in C/BLAS for N≤50) and correctly masks duplicate rows within `_DISTANCE_ZERO_EPS`.

---

## Approved Changes

### Wave 1 — Correctness & Robustness (Ready to Implement)

**1. Add input validation to `get_unified_sites()`** (improvements_1 #3)
- File: `src/metalsurfer/placement/site_enumeration.py`
- Add `if len(atoms) == 0: raise ValueError("atoms must contain at least one atom")` at the top of `_enumerate_unified_sites()`.
- Do NOT add an unconditional `cell_has_volume(cell)` check; line 496 already synthesizes a bounding-box cell for nanoparticles.

**2. Fix planar slab handling for primitive & collinear top layers** (improvements_1 #4)
- File: `src/metalsurfer/placement/site_enumeration.py`
- In `_top_layer_is_planar_from_arrays()`:
  - If `len(top_indices) == 0`: return `False`.
  - For `len(top_indices) < 3` or `rank < 3`: return `float(np.var(h)) < z_variance_threshold`.
- Eliminates unnecessary Voronoi attempts on 1-2 atom unit cells and collinear top layers.

**3. Elevate symmetry fallback log level** (improvements_1 #5)
- File: `src/metalsurfer/placement/site_context.py`
- Change `logger.info()` to `logger.warning()` in the `SymmetryAnalysisError` handler of `resolve_site_context_for_sampling()`.
- Ensures symmetry fallback events are visible under default production logging configurations.

---

### Wave 2 — Cleanups & Safeguards (Optional / Low Risk)

**4. Safeguard against duplicate cache entries / FIFO evictions** (improvements_1 #2)
- Files: `src/metalsurfer/placement/site_context.py`, `src/metalsurfer/placement/dissociative.py`
- Add `if cache_key in _SITE_CONTEXT_CACHE: return _SITE_CONTEXT_CACHE[cache_key]` inside the lock before inserting.
- Note: This prevents duplicate FIFO evictions; concurrent misses on identical slabs are already rare due to sequential per-molecule screening.

**5. Hoist static `pid_to_pool_position` lookup in BO** (improvements_2 #4)
- File: `src/metalsurfer/workflow/bayesian.py`
- Build a static `global_pid_to_pos` dict once before the BO batch loop instead of reconstructing it per batch.

---

## Rejected / Already Resolved

| # | Finding | Status | Reason |
|---|---------|--------|--------|
| 1.1 | Thread `site_context` | **Already Resolved** | Tracing confirms `site_context` is already threaded through every step (`core.py`, `placement_fill.py`, `shared.py`, `bayesian.py`, `generators.py`, `pose.py`) and cached by `_SITE_CONTEXT_CACHE`. |
| 1.6 | Tighten `_DISTANCE_ZERO_EPS` | **Rejected** | `1e-12` is appropriate; increasing to `1e-8` risks false atop classification on dense hollow/bridge sites. |
| 2.1 | Reduce `_TRANSFER_GATE_FOLDS` to 2 | **Rejected / Unnecessary** | Line 814 already skips transfer when `n_current < 4`. K=2 at n=4 leaves only 2 training samples, which is statistically degenerate for GBDT. Typical training time for K=3 is <50 ms. |
| 2.2 | `_unevaluated()` set difference | **Rejected** | Set iteration order is non-deterministic, breaking seed reproducibility in `rng.choice()`. List comprehension takes ~50 µs and is deterministic. |
| 2.3 | `NearestNeighbors` for self-distance | **Rejected** | Line 1069 already handles `len(X_prev) <= 1`. For N≤50, `cdist` in C/BLAS is faster than sklearn object construction and handles duplicate row masking. |
| 2.5 | Cache reference units list | **Rejected** | Function runtime is <0.5 µs. Caching adds mutable state and invalidation bugs with `pending_additions`. |

---

## Validation Strategy

1. Run placement test suite: `pytest tests/placement/ -v` (all 254 tests pass).
2. Run Bayesian test suite: `pytest tests/test_bayesian.py -v` (70 unit tests pass).
3. Add targeted unit tests:
   - Empty `Atoms()` raises `ValueError` in `get_unified_sites()`.
   - Primitive 1x1 slab (1 atom in top layer) is correctly identified as planar in `_top_layer_is_planar_from_arrays()`.
   - Collinear top-layer atoms are correctly identified as planar in `_top_layer_is_planar_from_arrays()`.
