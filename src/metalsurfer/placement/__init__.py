"""Placement package: site detection, generators, geometry, and policy."""

from __future__ import annotations

from ase import Atoms

from ..symmetry import SymmetryAnalyzer
from ._material import (
    material_aware_pbc as material_aware_pbc,
)
from .generators import (
    classify_adsorbate_orientation as classify_adsorbate_orientation,
)
from .generators import (
    distribute_placement_budget as distribute_placement_budget,
)
from .generators import (
    enumerate_placement_specs as enumerate_placement_specs,
)
from .generators import (
    estimate_placement_spec_capacity as estimate_placement_spec_capacity,
)
from .generators import (
    generate_placement_from_descriptor as generate_placement_from_descriptor,
)
from .generators import (
    generate_placement_from_pose as generate_placement_from_pose,
)
from .generators import (
    generate_placement_from_spec as generate_placement_from_spec,
)
from .generators import (
    generate_placement_from_spec_with_reason as generate_placement_from_spec_with_reason,
)
from .geometry import (
    calculate_min_distance as calculate_min_distance,
)
from .geometry import (
    check_initial_placement_distance as check_initial_placement_distance,
)
from .sites import (
    DEFAULT_SYMMETRY_TOLERANCE,
)
from .sites import (
    get_hollow_sites_for_adatoms as get_hollow_sites_for_adatoms,
)
from .sites import (
    get_symmetry_aware_sites as get_symmetry_aware_sites,
)
from .sites import (
    get_unified_sites as get_unified_sites,
)


def get_symmetry_info(
    slab: Atoms,
    symmetry_tolerance: float = DEFAULT_SYMMETRY_TOLERANCE,
) -> dict[str, object]:
    """Symmetry metadata including spglib space group."""
    symmetry_analyzer = SymmetryAnalyzer(slab, symmetry_tolerance=symmetry_tolerance)
    return symmetry_analyzer.get_symmetry_info()
