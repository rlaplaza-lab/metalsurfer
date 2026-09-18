"""Internal constants for placement numerics and physical heuristics."""

from ase.data import atomic_numbers
from ase.data import covalent_radii as ase_covalent_radii

from .. import _numeric_defaults

# Re-export shared defaults under placement-private names (used by geometry/sites).
_CONTACT_DISTANCE_THRESHOLD_DEFAULT_ANGSTROM = (
    _numeric_defaults.CONTACT_DISTANCE_THRESHOLD_DEFAULT_ANGSTROM
)
_CONTACT_MAX_CLOSEST_APPROACH_ANGSTROM = (
    _numeric_defaults.CONTACT_MAX_CLOSEST_APPROACH_ANGSTROM
)
_DEFAULT_HOLLOW_SITE_DEDUP_TOLERANCE = (
    _numeric_defaults.DEFAULT_HOLLOW_SITE_DEDUP_TOLERANCE
)
_DEFAULT_PLANAR_Z_VARIANCE_THRESHOLD = (
    _numeric_defaults.DEFAULT_PLANAR_Z_VARIANCE_THRESHOLD
)
_DEFAULT_SITE_EQUIVALENCE_TOLERANCE = (
    _numeric_defaults.DEFAULT_SITE_EQUIVALENCE_TOLERANCE
)
_DEFAULT_SYMMETRY_TOLERANCE = _numeric_defaults.DEFAULT_SYMMETRY_TOLERANCE
_MIN_CONTACT_RATIO_DEFAULT = _numeric_defaults.MIN_CONTACT_RATIO_DEFAULT
_MIN_INITIAL_DISTANCE_DEFAULT_ANGSTROM = (
    _numeric_defaults.MIN_INITIAL_DISTANCE_DEFAULT_ANGSTROM
)
_SURFACE_NORMAL_FALLBACK_NORM_EPS = _numeric_defaults.SURFACE_NORMAL_FALLBACK_NORM_EPS

RECOVERABLE_DISTANCE_REASONS = frozenset(
    {"adsorbate_overlap", "too_close", "too_far", "vdw_overlap"}
)


# Mean covalent radius of element sets from ASE data.
def _mean_tabulated_covalent_radius(elements: list[str]) -> float:
    """Mean positive covalent radius over *elements* present in ASE tables."""
    radii = []
    for elem in elements:
        z = atomic_numbers.get(elem)
        if z is not None and z < len(ase_covalent_radii):
            r = float(ase_covalent_radii[z])
            if r > 0.0:
                radii.append(r)
    return float(sum(radii) / len(radii))


def _compute_mean_adsorbate_covalent_radius() -> float:
    """Mean covalent radius of common adsorbate elements (C, H, O, N, S, P)."""
    return _mean_tabulated_covalent_radius(["C", "H", "O", "N", "S", "P"])


def _compute_mean_framework_covalent_radius() -> float:
    """Mean covalent radius of common framework (metal) elements.

    Used as the surface-radius fallback when a slab/framework exposes no
    recognised covalent radii. This is physically distinct from the adsorbate
    fallback and must not reuse it.
    """
    return _mean_tabulated_covalent_radius(
        ["Cu", "Pt", "Pd", "Ag", "Au", "Ni", "Fe", "Al", "Co"]
    )


# ---------------------------------------------------------------------------
# Voronoi site detection
# ---------------------------------------------------------------------------

# Minimum separation (Å) used to deduplicate Voronoi vertices as the same site.
_VORONOI_DEDUP_TOLERANCE: float = 0.1
_DISTANCE_ZERO_EPS: float = 1e-12
_DISTANCE_RATIO_FLOOR_EPS: float = 1e-8

# Number of nearest framework atoms used to estimate local geometry.
_NORMAL_K_NEIGHBOURS: int = 4
_SITE_CLASSIFICATION_NEIGHBOURS: int = 6

# Site-type classification based on d_i / d_1 distance ratios, where d_1 is
# the distance to the nearest framework atom and d_i is the i-th closest.
_ATOP_RATIO: float = 1.3  # d_2 / d_1 > this → atop site
_BRIDGE_EQ_TOL: float = 0.15  # |d_2 - d_1| / d_1 < this → bridge candidate
_BRIDGE_FAR_RATIO: float = 1.2  # d_3 / d_1 > this → confirmed bridge (not hollow)
_HOLLOW_EQ_TOL: float = 0.15  # |d_2,3 - d_1| / d_1 < this → hollow site

# Voronoi vertex classified as a pore site when nearest-atom distance exceeds
# gamma * mean(top-layer covalent radius), with a hard floor for sparse systems.
_PORE_THRESHOLD_COVALENT_SCALE: float = 2.5
_PORE_THRESHOLD_MIN_ANGSTROM: float = 2.0

# Radius-derived Voronoi accessibility window:
# probe_radius = alpha * mean(top-layer covalent radius)
# max_distance = beta * mean(top-layer covalent radius)
_VORONOI_PROBE_RADIUS_COVALENT_SCALE: float = 1.25
_VORONOI_MAX_DISTANCE_COVALENT_SCALE: float = 4.25
# Adsorbate vs surface fallbacks are physically distinct; do not alias them.
_ADSORBATE_COVALENT_RADIUS_FALLBACK: float = _compute_mean_adsorbate_covalent_radius()
_SURFACE_COVALENT_RADIUS_FALLBACK: float = _compute_mean_framework_covalent_radius()

# Ridge-based geodesic enrichment
# Subdivide Voronoi edges longer than _ENRICHMENT_SPACING_BETA × median(nn_distance).
_ENRICHMENT_SPACING_BETA: float = 1.2
# Hard cap on subdivisions per edge to prevent runaway on very long ridges.
_ENRICHMENT_MAX_SUBDIVISIONS: int = 6

# Top-layer depth for slab filtering based on local covalent radii.
_TOP_LAYER_DEPTH_COVALENT_SCALE: float = 1.8
_TOP_LAYER_DEPTH_MIN_ANGSTROM: float = 0.5
# Cap derived depth so FCC-like interlayers (~2.1 A) stay a single primary layer.
_TOP_LAYER_DEPTH_MAX_ANGSTROM: float = 1.2
# Default flatness tolerance (z-band half-width) for top-layer planarity checks.
# Distinct from _TOP_LAYER_DEPTH_MIN_ANGSTROM, which floors the derived depth.
_PLANAR_TOP_LAYER_TOLERANCE_ANGSTROM: float = 0.5
# Include one terrace below the primary band only when this close to h_max.
_STEP_TERRACE_MAX_GAP_ANGSTROM: float = 1.0

# Delaunay reference classification.
_DELAUNAY_BRIDGE_THRESHOLD_FRACTION: float = 0.3

# ---------------------------------------------------------------------------
# Placement geometry (z-offsets and parallel placement)
# ---------------------------------------------------------------------------

# Per-site-type z-offset as a fraction of mean local surface covalent radius.
# Hollow / pore / bridge sites sit slightly lower than atop; envelope is intermediate.
_SITE_Z_OFFSET_FROM_SURFACE_RADIUS: dict[str, float] = {
    "atop": 0.0,
    "bridge": -0.09,
    "hollow": -0.18,
    "pore": -0.18,
    "envelope": -0.135,
}

# Flat-aromatic parallel placement floor as a scale on (r_surface + r_mol).
_PARALLEL_Z_FLOOR_RADIUS_SUM_SCALE: float = 1.2

# Radius-derived adjustments to z_base_lo / z_base_hi for flat-aromatic
# parallel placements. These shrink the z-range so the ring sits closer to
# (but not inside) the surface.
_PARALLEL_Z_LO_SHRINK_RADIUS_SUM_SCALE: float = 0.2
_PARALLEL_Z_HI_SHRINK_RADIUS_SUM_SCALE: float = 0.3
_PARALLEL_Z_MIN_HI_MARGIN: float = 0.3  # ensure z_base_hi >= z_base_lo + this (Å)

# ---------------------------------------------------------------------------
# Dissociative placement (e.g. H₂ → 2 H on hollow sites)
# ---------------------------------------------------------------------------
# Adaptive min/max fragment separation from atomic radii and hollow NN geometry;
# see dissociative placement implementation for the constraint formulas.
_DISSOCIATIVE_MIN_FRAGMENT_SEP_RADIUS_SCALE: float = 0.7
_DISSOCIATIVE_MIN_FRAGMENT_SEP_FLOOR_ANGSTROM: float = 1.0
_DISSOCIATIVE_MAX_ADJACENT_SEP_NN_SCALE: float = 1.2
_DISSOCIATIVE_MAX_ADJACENT_SEP_FLOOR_ANGSTROM: float = 1.5
_DISSOCIATIVE_MAX_ADJACENT_SEP_CAP_ANGSTROM: float = 3.2

# ---------------------------------------------------------------------------
# Atop site injection
# ---------------------------------------------------------------------------

# Height factor for injected atop sites: site_z = atom_z + factor × median(nn_distance).
_ATOP_INJECTION_HEIGHT_FACTOR: float = 0.8

# ---------------------------------------------------------------------------
# Nanoparticle surface topology (hull + NN graph)
# ---------------------------------------------------------------------------

# Surface-atom nearest-neighbour edges accepted as bridges (scale on metal-metal nn).
_NP_NN_BOND_MIN_SCALE: float = 0.8
_NP_NN_BOND_MAX_SCALE: float = 1.2
# Atom counts as on the convex hull when max facet signed distance exceeds -eps.
_NP_HULL_SURFACE_EPS: float = 1e-6
# Site is outside the hull when max facet signed distance exceeds this eps.
_NP_HULL_OUTSIDE_EPS: float = 1e-6
# 4-fold hollows must be planar within this fraction of the metal-metal nn.
_NP_PLANAR_4RING_TOL_SCALE: float = 0.15

# ---------------------------------------------------------------------------
# Site clustering and symmetry
# ---------------------------------------------------------------------------
_BOUNDING_BOX_CELL_PAD_ANGSTROM: float = 5.0
_SLAB_Z_ABS_TOLERANCE_DEFAULT_ANGSTROM: float = 0.5
_KD_RADIUS_SEARCH_PADDING: float = 1.5

# ---------------------------------------------------------------------------
# Geometry numerics and shape/orientation heuristics
# ---------------------------------------------------------------------------
_QUATERNION_NORM_EPS: float = 1e-12
_FRAME_PROJECTION_TIE_EPS: float = 1e-10
_VECTOR_NORM_EPS: float = 1e-12
_FRAME_REF_ALIGNMENT_DOT_THRESHOLD: float = 0.95
_ROTATION_ALIGN_DOT_PARALLEL: float = 0.9999
_ROTATION_ALIGN_DOT_ANTIPARALLEL: float = -0.9999
_ROTATION_ALIGN_AXIS_SWITCH_DOT: float = 0.9
_INERTIA_EPS: float = 1e-8
_LINEAR_SHAPE_RATIO_MAX: float = 0.02
_FLAT_SHAPE_I1_I3_MAX: float = 0.55
_FLAT_SHAPE_I2_I3_MIN: float = 0.45
_BINDER_VECTOR_MIN_NORM: float = 0.1
_BINDER_ALIGNMENT_TARGET_DOT: float = 0.95
_PRINCIPAL_AXIS_SHORT_ALIGN_MAX_DOT: float = 0.7
_PRINCIPAL_AXIS_LONG_ALIGN_MIN_DOT: float = 0.3
_PRINCIPAL_AXIS_ROT_AXIS_MIN_NORM: float = 1e-6
_PRINCIPAL_AXIS_ROTATION_STEPS: int = 36
_PRINCIPAL_AXIS_ROTATION_STEP_DEG: float = 10.0

# Radius-derived fallback and contact-quality thresholds.
_VDW_RADIUS_FROM_COVALENT_SCALE: float = 1.2
_CONTACT_QUALITY_COVALENT_SUM_SCALE: float = 1.35
# Max variance of contact distances when requiring multi-atom contact (Å²).
_CONTACT_ATOM_VARIANCE_MAX: float = 0.5
_MIN_DISTANCE_HARD_FALLBACK_ANGSTROM: float = 2.0
_ADSORBATE_SEPARATION_COVALENT_SUM_SCALE: float = 1.0

# ---------------------------------------------------------------------------
# Policy and generator grids
# ---------------------------------------------------------------------------
_PARALLEL_FRACTION_NO_BINDERS: float = 0.8
_PARALLEL_FRACTION_SINGLE_BINDER: float = 0.3
_PARALLEL_FRACTION_LOW_BINDER_RATIO: float = 0.8
_PARALLEL_FRACTION_NO_RING: float = 0.5
_PARALLEL_FRACTION_HIGH_BINDER_RATIO: float = 0.3
_PARALLEL_FRACTION_MEDIUM_BINDER_RATIO: float = 0.5
_PARALLEL_FRACTION_HIGH_RATIO_CUTOFF: float = 0.5
_PARALLEL_FRACTION_MEDIUM_RATIO_CUTOFF: float = 0.2
_PLACEMENT_GRID_COUNT_SEED: int = 0
_GRID_BUILD_CAP: int = 10**9
# Working-set multiplier for early-cap policy paths (dissociative / heavy filters).
_EARLY_CAP_WORKING_SET_MULTIPLIER: int = 8
_TILT_FULL: tuple[float, ...] = (0.0, 15.0, 30.0, 45.0, 60.0, 90.0)
_TILT_PARALLEL: tuple[float, ...] = (0.0, 15.0, 30.0)
_AZIMUTH: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
_AZIMUTH_IN_PLANE: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
_Z_FRACTIONS: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)

# Soft priors when stratifying the placement grid down to n_desired: prefer
# milder tilts and mid-window heights (still seeded / deterministic).
_POLICY_PRIOR_TILT_WEIGHT_PER_DEG: float = 0.02
_POLICY_PRIOR_Z_FRACTION_TARGET: float = 0.5
_POLICY_PRIOR_Z_FRACTION_WEIGHT: float = 2.0

# Porous site sampling: prefer open pores near the front of the stratified draw.
_POROUS_SITE_INDEX_WEIGHT: float = 1e-3
# Working-set size for pore-only site lists before stratified sampling.
_PORE_SITE_CAP_NUM_PLACEMENTS_DEFAULT: int = 20
_PORE_SITE_CAP_MULTIPLIER: int = 20
_PORE_SITE_CAP_FLOOR: int = 80

# Discrete XY jitter attempts when clash descent is disabled.
_DISTANCE_RECOVERY_XY_ATTEMPTS: int = 4
# Deterministic mixers for XY recovery RNG (seed ^ placement ^ site).
_XY_RECOVERY_SEED_MIXER: int = 1_000_003
_XY_RECOVERY_PLACEMENT_MIXER: int = 97
_XY_RECOVERY_SITE_MIXER: int = 1_009

# Lateral-offset in-plane basis: switch Cartesian ref when nearly parallel to
# the site normal. Intentionally looser than
# ``_FRAME_REF_ALIGNMENT_DOT_THRESHOLD`` (0.95) used by
# ``compute_surface_site_frame`` — recovery only needs a non-degenerate basis,
# not a high-quality frame.
_LATERAL_OFFSET_REF_SWITCH_DOT: float = 0.9

# Round Cartesian positions to this many decimals before hashing (Å).
# Matches the precision used for dissociative / occupancy position digests.
_XYZ_HASH_DECIMALS: int = 6

# Packmol-style rigid-body clash descent (recovery + n-tuplet).
_CLASH_DESCENT_MAXITER: int = 40
_CLASH_DESCENT_AZIMUTH_BOUND_DEG: float = 20.0
_CLASH_DESCENT_SUCCESS_F: float = 1e-12
_CLASH_DESCENT_SUCCESS_VIOLATION_ANGSTROM: float = 1e-3
# Lateral salvage travel as a scale on the incoming in-plane footprint
# (or mean covalent radius for single atoms).
_CLASH_LATERAL_FOOTPRINT_SCALE: float = 1.0
# Treat azimuth deltas smaller than this as zero when composing quaternions.
_CLASH_AZIMUTH_DELTA_EPS_DEG: float = 1e-15
# n-tuplet near-miss rescue skips when min distance is below this fraction of
# the smallest covalent-pair sum (stacked nuclei).
_TUPLET_CLASH_RESCUE_COVALENT_SCALE: float = 0.5
# One-shot Voronoi accessibility widen when the first window finds no sites.
_VORONOI_AUTO_WIDEN_PROBE_SCALE: float = 0.8
_VORONOI_AUTO_WIDEN_MAX_SCALE: float = 1.25

# ---------------------------------------------------------------------------
# Adaptive-grid site generator (internal A/B plugin)
# ---------------------------------------------------------------------------
# Coarse spacing h0 = clip(c_h * L, h_min, h_max); refine while h > c_fine * L.
_ADAPTIVE_GRID_SPACING_SCALE: float = 0.45
_ADAPTIVE_GRID_FINE_SCALE: float = 0.25
_ADAPTIVE_GRID_H_MIN: float = 0.35
_ADAPTIVE_GRID_H_MAX: float = 1.5
_ADAPTIVE_GRID_MAX_LEVELS: int = 2
# Near-zero molecular extents (flat thickness, single-atom footprint) skip this.
_ADAPTIVE_GRID_EXTENT_EPS: float = 0.05
# NMS merge radius: max(nms_scale * h_fine, nms_length_scale * L).
# Within a coordination class this sets same-type spacing; cross-type peaks in
# the same ball are kept (same-class-only suppression).
_ADAPTIVE_GRID_NMS_SCALE: float = 1.5
_ADAPTIVE_GRID_NMS_LENGTH_SCALE: float = 0.5
# Soft per-chunk work budget: each shell/refine chunk keeps
# n_seeds × n_offsets ≤ this (all atoms are still seeded).
_ADAPTIVE_GRID_WORK_BUDGET: int = 250_000
# Floor merge radius as a fraction of framework median NN.
_ADAPTIVE_GRID_NMS_FRAMEWORK_SCALE: float = 0.35
# Same-class NMS merge as a fraction of framework median NN.
_ADAPTIVE_GRID_NMS_ATOP_NN_SCALE: float = 0.55
_ADAPTIVE_GRID_NMS_BRIDGE_NN_SCALE: float = 0.35
_ADAPTIVE_GRID_NMS_HOLLOW_NN_SCALE: float = 0.50
# Cross-class absolute floor (Å). 0 keeps bridge/hollow coexistence; typed
# post-classify dedup removes same-label near-duplicates.
_ADAPTIVE_GRID_NMS_HARD_FLOOR: float = 0.0
# Floor characteristic length before deriving h0 / refine depth.
_ADAPTIVE_GRID_LENGTH_FRAMEWORK_SCALE: float = 0.25
# Neighbours within this factor of nn count toward provisional coordination.
_ADAPTIVE_GRID_COORD_NN_FACTOR: float = 1.15
# Bin-prethin before NMS when the coarse cloud exceeds this many points.
_ADAPTIVE_GRID_BIN_PRETHIN: int = 8_000
# Exposure probe step along the outward unit vector (Å).
_ADAPTIVE_GRID_EXPOSURE_STEP: float = 0.25
