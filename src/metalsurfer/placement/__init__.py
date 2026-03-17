"""Adsorbate placement generation on slab surfaces.

Each call to :func:`generate_conformer_placement` produces a single
adsorbate placement for a given *placement_id*.  Successive IDs yield
diverse (x, y, z, rotation) combinations that span the periodic cell
and the full SO(3) orientation space.
"""

from . import generators, geometry, sites

# Public API
__all__ = [
    "calculate_min_distance",
    "check_initial_placement_distance",
    "classify_adsorbate_orientation",
    "enumerate_placement_specs",
    "generate_conformer_placement",
    "generate_placement_from_descriptor",
    "generate_placement_from_spec",
    "get_adsorption_sites",
    "get_envelope_placement_sites",
    "get_hollow_sites_for_adatoms",
    "is_surface_planar",
]

# Re-export public symbols
calculate_min_distance = geometry.calculate_min_distance
check_initial_placement_distance = geometry.check_initial_placement_distance
classify_adsorbate_orientation = generators.classify_adsorbate_orientation
enumerate_placement_specs = generators.enumerate_placement_specs
generate_conformer_placement = generators.generate_conformer_placement
generate_placement_from_descriptor = generators.generate_placement_from_descriptor
generate_placement_from_spec = generators.generate_placement_from_spec
get_adsorption_sites = sites.get_adsorption_sites
get_envelope_placement_sites = sites.get_envelope_placement_sites
get_hollow_sites_for_adatoms = sites.get_hollow_sites_for_adatoms
is_surface_planar = sites.is_surface_planar

# Internal symbols used by tests
_classify_molecule_shape = geometry._classify_molecule_shape
_cluster_equivalent_sites = sites._cluster_equivalent_sites
_compute_site_z_base = sites._compute_site_z_base
_get_site_surface_radii = sites._get_site_surface_radii
_is_flat_aromatic = generators._is_flat_aromatic
_is_flat_aromatic_with_en = generators._is_flat_aromatic_with_en
_random_rotation_matrix = geometry._random_rotation_matrix
_sample_xy_in_cell = geometry._sample_xy_in_cell
