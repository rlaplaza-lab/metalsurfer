"""Internal shared numeric defaults for AdsorptionConfig and placement.

Private module (leading underscore): prefer ``AdsorptionConfig`` fields as the
user-facing API. Kept outside ``placement`` so ``config`` can import without
loading ``placement.generators`` (circular import).
"""

DEFAULT_SYMMETRY_TOLERANCE: float = 0.1
DEFAULT_SITE_EQUIVALENCE_TOLERANCE: float = 0.05
DEFAULT_HOLLOW_SITE_DEDUP_TOLERANCE: float = 0.1
DEFAULT_PLANAR_Z_VARIANCE_THRESHOLD: float = 0.01
MIN_INITIAL_DISTANCE_DEFAULT_ANGSTROM: float = 1.5
MIN_CONTACT_RATIO_DEFAULT: float = 0.8
CONTACT_MAX_CLOSEST_APPROACH_ANGSTROM: float = 0.8
CONTACT_DISTANCE_THRESHOLD_DEFAULT_ANGSTROM: float = 2.5
