"""Adsorbate placement generation on slab surfaces.

The spec-based workflow (:func:`enumerate_placement_specs` →
:func:`generate_placement_from_spec`) provides deterministic, reproducible
placement of adsorbates on surface sites.
"""

from . import generators, geometry, sites

# Public API
__all__ = [
    "calculate_min_distance",
    "check_initial_placement_distance",
    "classify_adsorbate_orientation",
    "detect_material_type",
    "distribute_placement_budget",
    "enumerate_placement_specs",
    "estimate_molecule_complexity",
    "estimate_placement_spec_capacity",
    "generate_placement_from_pose",
    "generate_placement_from_descriptor",
    "generate_placement_from_spec",
    "generate_placement_from_spec_with_reason",
    "get_hollow_sites_for_adatoms",
    "get_symmetry_aware_sites",
    "get_symmetry_info",
    "get_unified_sites",
]

# Re-export public symbols
calculate_min_distance = geometry.calculate_min_distance
check_initial_placement_distance = geometry.check_initial_placement_distance
classify_adsorbate_orientation = generators.classify_adsorbate_orientation
detect_material_type = sites.detect_material_type
distribute_placement_budget = generators.distribute_placement_budget
enumerate_placement_specs = generators.enumerate_placement_specs
estimate_molecule_complexity = generators.estimate_molecule_complexity
estimate_placement_spec_capacity = generators.estimate_placement_spec_capacity
generate_placement_from_pose = generators.generate_placement_from_pose
generate_placement_from_descriptor = generators.generate_placement_from_descriptor
generate_placement_from_spec = generators.generate_placement_from_spec
generate_placement_from_spec_with_reason = (
    generators.generate_placement_from_spec_with_reason
)
get_hollow_sites_for_adatoms = sites.get_hollow_sites_for_adatoms
get_symmetry_aware_sites = sites.get_symmetry_aware_sites
get_symmetry_info = sites.get_symmetry_info
get_unified_sites = sites.get_unified_sites

# Internal symbols used by tests
_classify_molecule_shape = geometry._classify_molecule_shape
_cluster_equivalent_sites = sites._cluster_equivalent_sites
_compute_site_z_base = sites._compute_site_z_base
_get_site_surface_radii = sites._get_site_surface_radii
_is_flat_aromatic = generators._is_flat_aromatic
_is_flat_aromatic_with_en = generators._is_flat_aromatic_with_en
_is_dissociable_diatomic = generators._is_dissociable_diatomic
_get_hollow_site_pairs = generators._get_hollow_site_pairs
_random_rotation_matrix = geometry._random_rotation_matrix
material_aware_pbc = geometry.material_aware_pbc
