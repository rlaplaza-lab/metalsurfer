"""Internal constants for placement numerics and physical heuristics."""


# Compute mean covalent radius of common adsorbate elements from ASE data.
# This replaces hardcoded fallback values with dynamically computed values.
def _compute_mean_adsorbate_covalent_radius() -> float:
    """Mean covalent radius of common adsorbate elements (C, H, O, N, S, P)."""
    from ase.data import atomic_numbers
    from ase.data import covalent_radii as ase_covalent_radii

    common_elements = ["C", "H", "O", "N", "S", "P"]
    radii = []
    for elem in common_elements:
        z = atomic_numbers.get(elem)
        if z is not None and z < len(ase_covalent_radii):
            r = float(ase_covalent_radii[z])
            if r > 0.0:
                radii.append(r)
    if radii:
        return float(sum(radii) / len(radii))
    # Ultimate fallback if ASE data is unavailable (should never happen)
    return 0.77


# ---------------------------------------------------------------------------
# Material detection
# ---------------------------------------------------------------------------

# Fraction of unit-cell z-extent occupied by framework atoms above which a
# 3D-periodic structure is treated as porous instead of slab-like.
_SLAB_MAX_OCCUPIED_FRACTION: float = 0.70

# ---------------------------------------------------------------------------
# Voronoi site detection
# ---------------------------------------------------------------------------

# Minimum separation (Å) used to deduplicate Voronoi vertices as the same site.
_VORONOI_DEDUP_TOLERANCE: float = 0.1
_VORONOI_FRACTIONAL_CELL_MARGIN: float = 0.01
_SURFACE_NORMAL_FALLBACK_NORM_EPS: float = 1e-8
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
# Unified fallback: mean covalent radius of common adsorbate elements (C,H,O,N,S,P)
# Computed dynamically from ASE data to avoid magic numbers.
_MEAN_COVALENT_RADIUS_FALLBACK: float = _compute_mean_adsorbate_covalent_radius()
# Backwards compatibility alias
_VORONOI_RADIUS_FALLBACK_ANGSTROM: float = _MEAN_COVALENT_RADIUS_FALLBACK

# Ridge-based geodesic enrichment
# Subdivide Voronoi edges longer than _ENRICHMENT_SPACING_BETA × median(nn_distance).
_ENRICHMENT_SPACING_BETA: float = 1.2
# Hard cap on subdivisions per edge to prevent runaway on very long ridges.
_ENRICHMENT_MAX_SUBDIVISIONS: int = 6

# Top-layer depth for slab filtering based on local covalent radii.
_TOP_LAYER_DEPTH_COVALENT_SCALE: float = 1.8
_TOP_LAYER_DEPTH_MIN_ANGSTROM: float = 0.5

# Delaunay reference classification.
_DELAUNAY_BRIDGE_THRESHOLD_FRACTION: float = 0.3
# Use unified fallback for Delaunay characteristic length
_DELAUNAY_CHAR_LENGTH_FALLBACK_ANGSTROM: float = _MEAN_COVALENT_RADIUS_FALLBACK

# ---------------------------------------------------------------------------
# Placement geometry (z-offsets and parallel placement)
# ---------------------------------------------------------------------------

# Per-site-type z-offset as a fraction of mean local surface covalent radius.
# Hollow / bridge sites sit slightly lower than atop; envelope is intermediate.
_SITE_Z_OFFSET_FROM_SURFACE_RADIUS: dict[str, float] = {
    "atop": 0.0,
    "bridge": -0.09,
    "hollow": -0.18,
    "envelope": -0.135,
}

# Hard floor for flat-aromatic parallel placement above the surface (max of
# min_floor and scale times the sum of surface and molecule radii).
_PARALLEL_Z_FLOOR_MIN_ANGSTROM: float = 2.2
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

# Minimum and maximum separation between fragment landing sites (adaptive
# min/max from atomic radii and surface geometry; see implementation).
#
# Note: The algorithm now uses an adaptive approach that considers both atomic
# properties (covalent radii) and surface geometry (hollow site distances).
# This makes it automatically compatible with both close-packed and open surfaces.
#
# The adaptive approach uses:
# - atomic_constraint = scale * (2 * mean_surface_atom_covalent_radius)
# - surface_constraint = 0.8 * mean_hollow_site_NN_distance
# - min_sep = max(floor, min(atomic_constraint, surface_constraint))
#
# This ensures that:
# 1. For close-packed surfaces: surface_constraint is smaller, so it dominates
# 2. For open surfaces: atomic_constraint is smaller, so it dominates
# 3. Always respects the absolute minimum floor for physical reasonableness
_DISSOCIATIVE_MIN_FRAGMENT_SEP_RADIUS_SCALE: float = (
    0.7  # More conservative atomic constraint
)
_DISSOCIATIVE_MIN_FRAGMENT_SEP_FLOOR_ANGSTROM: float = 1.0  # Absolute minimum: 1.0 Å
_DISSOCIATIVE_MAX_ADJACENT_SEP_NN_SCALE: float = 1.2  # Conservative surface scaling
_DISSOCIATIVE_MAX_ADJACENT_SEP_FLOOR_ANGSTROM: float = (
    1.5  # Minimum max separation: 1.5 Å
)
_DISSOCIATIVE_MAX_ADJACENT_SEP_CAP_ANGSTROM: float = 3.2  # Maximum reasonable: 3.2 Å

# ---------------------------------------------------------------------------
# Atop site injection
# ---------------------------------------------------------------------------

# Height factor for injected atop sites: site_z = atom_z + factor × median(nn_distance).
# 0.8 places the site slightly below the median Voronoi vertex height, closer to
# the binding geometry expected for atop adsorbates (CO, H₂O, NH₃).
_ATOP_INJECTION_HEIGHT_FACTOR: float = 0.8

# ---------------------------------------------------------------------------
# Site clustering and symmetry
# ---------------------------------------------------------------------------
_DEFAULT_SYMMETRY_TOLERANCE: float = 0.1
_DEFAULT_SITE_EQUIVALENCE_TOLERANCE: float = 0.05
_DEFAULT_HOLLOW_SITE_DEDUP_TOLERANCE: float = 0.1
_DEFAULT_PLANAR_Z_VARIANCE_THRESHOLD: float = 0.01
_BOUNDING_BOX_CELL_PAD_ANGSTROM: float = 5.0
_SLAB_Z_ABS_TOLERANCE_DEFAULT_ANGSTROM: float = 0.5
_KD_RADIUS_SEARCH_PADDING: float = 1.5

# Use unified fallback for molecular covalent radius
_MOL_COVALENT_RADIUS_FALLBACK: float = _MEAN_COVALENT_RADIUS_FALLBACK

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
# Soft upper bound on closest mol–slab approach for "strict" initial contact (Å).
_CONTACT_MAX_CLOSEST_APPROACH_ANGSTROM: float = 0.8
# Distance cutoff counting an adsorbate atom as contacting the surface (Å).
_CONTACT_DISTANCE_THRESHOLD_DEFAULT_ANGSTROM: float = 2.5
# Max variance of contact distances when requiring multi-atom contact (Å²).
_CONTACT_ATOM_VARIANCE_MAX: float = 0.5
_MIN_DISTANCE_COVALENT_FALLBACK_SCALE: float = 1.0
_MIN_DISTANCE_HARD_FALLBACK_ANGSTROM: float = 2.0
_MIN_INITIAL_DISTANCE_DEFAULT_ANGSTROM: float = 1.5
_ADSORBATE_SEPARATION_COVALENT_SUM_SCALE: float = 1.0

# ---------------------------------------------------------------------------
# Policy and generator grids
# ---------------------------------------------------------------------------
_ORIENTATION_CLASSIFICATION_PARALLEL_DOT_THRESHOLD: float = 0.7
_PARALLEL_FRACTION_NO_BINDERS: float = 0.8
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
