"""Internal shared numeric defaults for AdsorptionConfig and placement.

Private module (leading underscore): prefer ``AdsorptionConfig`` fields as the
user-facing API. Kept outside ``placement`` so ``config`` can import without
loading ``placement.generators`` (circular import).
"""

DEFAULT_SYMMETRY_TOLERANCE: float = 0.1
DEFAULT_SITE_EQUIVALENCE_TOLERANCE: float = 0.05
DEFAULT_HOLLOW_SITE_DEDUP_TOLERANCE: float = 0.1
DEFAULT_PLANAR_Z_VARIANCE_THRESHOLD: float = 0.01
DEFAULT_SEED: int = 42
DEFAULT_FMAX: float = 0.05
MIN_INITIAL_DISTANCE_DEFAULT_ANGSTROM: float = 1.5
MIN_ADSORBATE_SEPARATION_DEFAULT_ANGSTROM: float = 1.5
MIN_CONTACT_RATIO_DEFAULT: float = 0.8
# Absolute distance in Angstrom (NOT a ratio): the closest adsorbate-substrate
# contact must be no further than this for a placement to count as "in contact".
# It is compared against ``calculate_contact_quality()["contact_distance"]``,
# which is a real interatomic distance, and it must therefore sit above the
# lower bound enforced by ``check_initial_placement_distance``
# (max(MIN_INITIAL_DISTANCE_DEFAULT_ANGSTROM, MIN_CONTACT_RATIO_DEFAULT * sum of
# covalent radii) ~= 1.7 A for typical adsorbate/metal pairs). A sub-Angstrom
# value here makes the admissible window empty and rejects every physically
# reasonable placement.
CONTACT_MAX_CLOSEST_APPROACH_ANGSTROM: float = 3.0
CONTACT_DISTANCE_THRESHOLD_DEFAULT_ANGSTROM: float = 2.5
MIN_CALCULATOR_CELL_C_ANG: float = 18.0
# Degeneracy guard for cell-frame geometry: below this norm the a x b cross
# product (or an in-plane basis vector) is treated as degenerate and a
# Cartesian fallback axis is used instead.
SURFACE_NORMAL_FALLBACK_NORM_EPS: float = 1e-8
