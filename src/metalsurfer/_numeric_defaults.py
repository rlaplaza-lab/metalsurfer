"""Internal shared numeric defaults for AdsorptionConfig, placement, and BO.

Private module (leading underscore): prefer ``AdsorptionConfig`` fields as the
user-facing API. Kept outside ``placement`` so ``config`` can import without
loading ``placement.generators`` (circular import).
"""

from typing import Literal

DEFAULT_SYMMETRY_TOLERANCE: float = 0.1
DEFAULT_SITE_EQUIVALENCE_TOLERANCE: float = 0.05
DEFAULT_PLANAR_Z_VARIANCE_THRESHOLD: float = 0.01
DEFAULT_TOP_LAYER_TOLERANCE: float = 0.5
DEFAULT_SEED: int = 42
DEFAULT_FMAX: float = 0.05
DEFAULT_MODEL_NAME: str = "uma-s-1p2"
DEFAULT_TASK_NAME: str = "oc25"
DEFAULT_DISPERSION: Literal["none", "d3"] = "none"
DEFAULT_DEVICE: str = "cuda"
DEFAULT_STAGE1_STEPS: int = 50
DEFAULT_STAGE2_STEPS: int = 150
DEFAULT_PLACEMENT_Z_RANGE: tuple[float, float] = (0.7, 1.25)
# Bottom anchor (Å): skip a no-op translate, then reject a slab still off zero.
DEFAULT_BOTTOM_ANCHOR_TRANSLATE_EPS: float = 1e-6
DEFAULT_BOTTOM_ANCHOR_MAX_OFFSET_ANG: float = 0.05
DEFAULT_FROZEN_SUBSTRATE_DISPLACEMENT_TOL_ANG: float = 0.01
# Joblib-style CPU workers: -2 = all but one CPU (shared by config, sites, BO).
DEFAULT_N_JOBS: int = -2

MIN_INITIAL_DISTANCE_DEFAULT_ANGSTROM: float = 1.5
MIN_ADSORBATE_SEPARATION_DEFAULT_ANGSTROM: float = 1.5
MIN_CONTACT_RATIO_DEFAULT: float = 0.8
# Scale on the incoming in-plane footprint radius used under coverage occupancy.
OCCUPANCY_FOOTPRINT_SCALE_DEFAULT: float = 0.85
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
# Scale on (r_i + r_j) for counting an adsorbate atom as in contact.
CONTACT_COVALENT_SCALE_DEFAULT: float = 1.35
MIN_CALCULATOR_CELL_C_ANG: float = 18.0
# Vacuum (Å) ensured above the top surface layer by ensure_slab_z_alignment.
DEFAULT_SLAB_TOP_VACUUM_ANG: float = 15.0
# Farther than this (Å) from the nearest substrate atom counts as desorption.
BINDING_DISTANCE_THRESHOLD_DEFAULT_ANGSTROM: float = 4.0
# Default in-plane image separation (Å) enforced during substrate preparation.
MIN_PBC_IMAGE_SEPARATION_DEFAULT_ANGSTROM: float = 8.0
# BO transfer: prior rows below this feature-space similarity are dropped.
BO_TRANSFER_MIN_SIMILARITY_DEFAULT: float = 0.05
# Campaign ``bo.transfer.weight_cap``. Cumulative refit keeps its own cap.
BO_TRANSFER_WEIGHT_CAP_DEFAULT: float = 0.35
BO_CUMULATIVE_REFIT_WEIGHT_CAP_DEFAULT: float = 0.5
BO_CUMULATIVE_REFIT_PROXIMITY_FLOOR_DEFAULT: float = 0.05
BO_TRANSFER_TRUST_PATIENCE_DEFAULT: int = 2
ADAPTIVE_GRID_DEFAULT_SPACING: float = 0.70
ADAPTIVE_GRID_DEFAULT_REFINE_LEVELS: int = 0
ADAPTIVE_GRID_NMS_FRAMEWORK_SCALE: float = 0.25
AUTOBATCHER_MAX_MEMORY_PADDING_DEFAULT: float = 0.5
SATURATION_AUTOBATCHER_REUSE_GROWTH_ATOMS: int = 32
SATURATION_AUTOBATCHER_REUSE_GROWTH_FRACTION: float = 0.1
# Gas-phase conformer dedup: centred RMSD (Å) and energy difference (eV).
RMSD_DEDUP_THRESHOLD_DEFAULT_ANGSTROM: float = 0.35
ENERGY_DEDUP_THRESHOLD_DEFAULT_EV: float = 0.05
# Degeneracy guard for cell-frame geometry: below this norm the a x b cross
# product (or an in-plane basis vector) is treated as degenerate and a
# Cartesian fallback axis is used instead.
SURFACE_NORMAL_FALLBACK_NORM_EPS: float = 1e-8
# Min vacuum gap (Å) by which -a×b must exceed +a×b to select the lower face.
SLAB_VACUUM_FACE_MARGIN_ANG: float = 1.0
ACQUISITION_XI_DEFAULT: float = 1e-6
ACQUISITION_SIGMA_FLOOR: float = 1e-9
RESIDUAL_SIGMA_DISTANCE_TEMPER: float = 0.25
DEFAULT_TRANSFER_EXPLORATION_FRACTION: float = 0.2
DEFAULT_BO_SIMILARITY_LENGTHSCALE: float = 4.0
DEFAULT_UCB_KAPPA: float = 1.96  # two-sided 95% normal quantile
# Boltzmann (eV/K) and IUPAC SATP reservoir defaults for saturation Ω ranking.
K_B_EV_PER_K: float = 8.617e-5
STANDARD_TEMPERATURE_K: float = 298.15
STANDARD_PRESSURE_BAR: float = 1.0  # p° in Ω = E_ads − k_B T ln(a · p / p°)
# Conformer prior temperature (K). Distinct from STANDARD_TEMPERATURE_K (SATP).
DEFAULT_BOLTZMANN_TEMPERATURE_K: float = 300.0
