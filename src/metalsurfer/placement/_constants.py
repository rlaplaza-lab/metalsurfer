"""Numeric constants for the placement submodule.

All physics thresholds, geometry offsets, and sampling parameters used
across generators.py, sites.py, and _material.py are centralised here.

Values are chosen from empirical benchmarks on a range of slabs,
nanoparticles, and porous materials; see CORE_SYSTEM_EXPLANATION.md §Placement.
"""

# ---------------------------------------------------------------------------
# Material detection
# ---------------------------------------------------------------------------

# Fraction of unit-cell z-extent that must be vacuum for slab classification.
# A slab with 70 % or more of its z-axis *empty* is classified as a slab
# rather than a 3D-periodic porous material.
_SLAB_VACUUM_FRACTION: float = 0.70

# ---------------------------------------------------------------------------
# Voronoi site detection
# ---------------------------------------------------------------------------

# Minimum separation (Å) used to deduplicate Voronoi vertices as the same site.
_VORONOI_DEDUP_TOLERANCE: float = 0.1

# Number of nearest framework atoms used to estimate the local surface normal.
_NORMAL_K_NEIGHBOURS: int = 4

# Site-type classification based on d_i / d_1 distance ratios, where d_1 is
# the distance to the nearest framework atom and d_i is the i-th closest.
_ATOP_RATIO: float = 1.3  # d_2 / d_1 > this → atop site
_BRIDGE_EQ_TOL: float = 0.15  # |d_2 - d_1| / d_1 < this → bridge candidate
_BRIDGE_FAR_RATIO: float = 1.2  # d_3 / d_1 > this → confirmed bridge (not hollow)
_HOLLOW_EQ_TOL: float = 0.15  # |d_2,3 - d_1| / d_1 < this → hollow site

# Voronoi vertex classified as a pore site when its nearest-atom distance
# exceeds this threshold (Å).
_PORE_THRESHOLD_ANGSTROM: float = 2.5

# Ridge-based geodesic enrichment
# Subdivide Voronoi edges longer than _ENRICHMENT_SPACING_BETA × median(nn_distance).
_ENRICHMENT_SPACING_BETA: float = 1.2
# Hard cap on subdivisions per edge to prevent runaway on very long ridges.
_ENRICHMENT_MAX_SUBDIVISIONS: int = 6

# ---------------------------------------------------------------------------
# Placement geometry (z-offsets and parallel placement)
# ---------------------------------------------------------------------------

# Per-site-type z-offset (Å) applied on top of the base z_range.
# Hollow / bridge sites sit slightly lower than atop; envelope is intermediate.
_SITE_Z_OFFSETS: dict[str, float] = {
    "atop": 0.0,
    "bridge": -0.1,
    "hollow": -0.2,
    "envelope": -0.15,
}

# Hard floor for flat-aromatic parallel placement above the surface (Å).
# Prevents the ring from being placed inside the surface for low z_range values.
_PARALLEL_Z_FLOOR_ANGSTROM: float = 2.4

# Adjustments to z_base_lo / z_base_hi for flat-aromatic parallel placements.
# These shrink the z-range so the ring sits closer to (but not inside) the surface.
_PARALLEL_Z_LO_SHRINK: float = 0.4  # lower z_base_lo by this amount (Å)
_PARALLEL_Z_HI_SHRINK: float = 0.6  # lower z_base_hi by this amount (Å)
_PARALLEL_Z_MIN_HI_MARGIN: float = 0.3  # ensure z_base_hi >= z_base_lo + this (Å)

# Interleaving ratio for parallel vs EN-down placements within the spec cycle.
# Reserved for future use in multi-stage placement sampling.
_PARALLEL_EN_CYCLE_SIZE: int = 10

# Linear molecules: intended fraction of specs using vertical (binding-atom-down)
# orientation. Reserved for future use.
_LINEAR_VERTICAL_FRACTION: float = 0.75

# Rotation cycle divisor for linear molecules.
# (cycle % N) == 0 → flat; (cycle % N) != 0 → vertical.  Reserved for future use.
_LINEAR_VERTICAL_CYCLE: int = 4

# ---------------------------------------------------------------------------
# Dissociative placement (e.g. H₂ → 2 H on hollow sites)
# ---------------------------------------------------------------------------

# Minimum and maximum separation (Å) between the two fragment landing sites.
# Site pairs closer than _min or farther than _max are rejected.
_DISSOCIATIVE_MIN_FRAGMENT_SEP: float = 1.0
_DISSOCIATIVE_MAX_ADJACENT_SEP: float = 2.8

# ---------------------------------------------------------------------------
# Atop site injection
# ---------------------------------------------------------------------------

# Height factor for injected atop sites: site_z = atom_z + factor × median(nn_distance).
# 0.8 places the site slightly below the median Voronoi vertex height, closer to
# the binding geometry expected for atop adsorbates (CO, H₂O, NH₃).
_ATOP_INJECTION_HEIGHT_FACTOR: float = 0.8
