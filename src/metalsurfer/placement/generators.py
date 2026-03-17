"""Placement generation logic for adsorbate placement on slab surfaces."""

import itertools
import logging
import random
from collections.abc import Callable

import numpy as np
from ase import Atoms
from ase.geometry import find_mic

from ..config import AdsorptionConfig
from ..conformers import select_conformer_for_placement
from ..exceptions import DependencyMissingError
from ..models import PlacementDescriptor, PlacementSpec
from . import geometry as geom
from . import sites as sts

logger = logging.getLogger(__name__)

# Site-type z offsets (A) relative to base placement_z_range for physics-based placement
_SITE_Z_OFFSETS: dict[str, float] = {
    "atop": 0.0,
    "bridge": -0.1,
    "hollow": -0.2,
    "envelope": -0.15,
}
# Buffers for flat-aromatic parallel placement (Å)
_PARALLEL_Z_FLOOR_ANGSTROM = 2.4
# Cycle size for interleaving parallel vs EN-down placements
_PARALLEL_EN_CYCLE_SIZE = 10
# Linear molecules: fraction of placements with vertical (binding-atom-down) orientation
_LINEAR_VERTICAL_FRACTION = 0.75
# Rotation cycle divisor: (cycle % N) == 0 uses flat; (cycle % N) != 0 uses vertical
_LINEAR_VERTICAL_CYCLE = 4


def _is_flat_aromatic(
    shape: str,
    smiles: str | None,
    symbols: list[str],
) -> bool:
    """True if the adsorbate is flat with aromatic EN atoms (parallel-placement candidate)."""
    if shape != "flat":
        return False
    binders = geom._binding_atom_candidates(symbols)
    if smiles is not None:
        return _is_flat_aromatic_with_en(smiles)
    return bool(binders)


def _is_flat_aromatic_with_en(smiles: str) -> bool:
    """True if molecule has aromatic rings and electronegative (binding) atoms."""
    try:
        from rdkit import Chem
    except (ImportError, AttributeError) as exc:
        raise DependencyMissingError(
            "rdkit",
            "flat aromatic detection for placement",
            "pip install rdkit",
        ) from exc
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)
    aromatic = any(a.GetIsAromatic() for a in mol.GetAtoms())
    binders = {"O", "N", "S", "F", "Cl", "Br", "I"}
    has_en = any(a.GetSymbol() in binders for a in mol.GetAtoms())
    return bool(aromatic and has_en)


def classify_adsorbate_orientation(
    atoms: Atoms, slab_size: int, threshold: float = 0.7
) -> str:
    """Classify adsorbate as 'parallel' or 'EN-down' from inertia (plane normal vs surface).

    For flat molecules, the plane normal is the axis of largest inertia (eigenvecs[:, 2]),
    per the perpendicular axis theorem. Parallel = ring approximately horizontal;
    EN-down = ring tilted, electronegative atom toward surface.
    """
    pos = atoms.get_positions()[slab_size:]
    if len(pos) < 3:
        return "unknown"
    masses = atoms.get_masses()[slab_size:]
    _, eigenvecs = geom._compute_inertia_tensor(pos, masses)
    plane_normal = eigenvecs[:, 2]
    if plane_normal[2] < 0:
        plane_normal = -plane_normal
    dot = abs(float(np.dot(plane_normal, np.array([0.0, 0.0, 1.0]))))
    return "parallel" if dot > threshold else "EN-down"


def _get_xy_for_placement(
    placement_id: int,
    config: AdsorptionConfig,
    slab: Atoms,
    cell: np.ndarray,
    rng: random.Random,
) -> tuple[float, float, str | None, list[dict[str, object]] | None]:
    """Return (x, y, site_type, unique_sites) for placement. site_type/unique_sites None for random."""
    unique_sites, use_sites = _get_unique_sites_for_specs(slab, config)
    if use_sites and unique_sites:
        idx = placement_id % len(unique_sites)
        site = unique_sites[idx]
        xy = np.asarray(site["xy"])
        return float(xy[0]), float(xy[1]), str(site["site_type"]), unique_sites
    x, y = geom._sample_xy_in_cell(cell, rng)
    return x, y, None, None


def _get_unique_sites_for_specs(
    slab: Atoms,
    config: AdsorptionConfig,
) -> tuple[list[dict[str, object]], bool]:
    """Get unique sites for placement. Returns (sites, use_sites)."""
    pbc = slab.get_pbc()
    slab_ok = pbc is not None and len(pbc) >= 2 and pbc[0] and pbc[1] and len(slab) >= 6
    use_sites = config.placement_mode == "sites" or (
        config.placement_mode == "auto" and slab_ok
    )
    use_envelope = config.placement_mode == "envelope" or (
        config.placement_mode == "auto" and slab_ok
    )

    if not use_sites and config.placement_mode != "envelope":
        return [], False
    if not slab_ok and config.placement_mode != "random":
        return [], False

    if config.placement_mode == "envelope" and slab_ok:
        raw_sites = sts.get_envelope_placement_sites(
            slab, config.top_layer_tolerance, cell=np.array(slab.get_cell())
        )
    elif use_sites:
        if config.placement_mode == "auto" and use_envelope:
            planar = sts.is_surface_planar(
                slab,
                config.top_layer_tolerance,
                config.planar_z_variance_threshold,
            )
            if planar:
                raw_sites = sts.get_adsorption_sites(slab, config.top_layer_tolerance)
            else:
                raw_sites = sts.get_envelope_placement_sites(
                    slab,
                    config.top_layer_tolerance,
                    cell=np.array(slab.get_cell()),
                )
        else:
            raw_sites = sts.get_adsorption_sites(slab, config.top_layer_tolerance)
    else:
        raw_sites = None

    if not raw_sites:
        return [], False
    cell = np.array(slab.get_cell())
    unique_sites = sts._cluster_equivalent_sites(raw_sites, cell, tolerance=0.05)
    return unique_sites, bool(unique_sites)


def _apply_placement_core(
    adsorbate: Atoms,
    spec: PlacementSpec,
    slab: Atoms,
    config: AdsorptionConfig,
    smiles: str | None,
    z_fraction: float | None = None,
    xy_override: tuple[float, float] | None = None,
) -> tuple[Atoms, PlacementDescriptor] | None:
    """Single placement attempt: orient, position, distance check. Returns (adsorbate, descriptor) or None."""
    ads_pos = adsorbate.get_positions().copy()
    ads_pos -= np.mean(ads_pos, axis=0)
    surface_z = float(np.max(slab.get_positions()[:, 2]))
    cell = np.array(slab.get_cell())
    normal = np.array([0.0, 0.0, 1.0])
    shape, _, _ = geom._classify_molecule_shape(ads_pos)
    symbols = adsorbate.get_chemical_symbols()

    unique_sites, use_sites = _get_unique_sites_for_specs(slab, config)
    if xy_override is not None:
        x, y = xy_override[0], xy_override[1]
    elif use_sites and spec.site_index >= 0 and spec.site_index < len(unique_sites):
        site = unique_sites[spec.site_index]
        xy = np.asarray(site["xy"])
        x, y = float(xy[0]), float(xy[1])
    else:
        rng = random.Random(config.seed + spec.placement_index)
        x, y = geom._sample_xy_in_cell(cell, rng)

    site = (
        unique_sites[spec.site_index]
        if use_sites and 0 <= spec.site_index < len(unique_sites)
        else None
    )
    z_base_lo, z_base_hi = sts._compute_site_z_base(config, slab, site, symbols)
    if site and spec.site_type and spec.site_type in _SITE_Z_OFFSETS:
        offset = _SITE_Z_OFFSETS[spec.site_type]
        z_base_lo = z_base_lo + offset
        z_base_hi = z_base_hi + offset

    flat_aromatic = _is_flat_aromatic(shape, smiles, symbols)
    if flat_aromatic and spec.orientation_type == "parallel":
        z_base_lo = max(_PARALLEL_Z_FLOOR_ANGSTROM, z_base_lo - 0.4)
        z_base_hi = max(z_base_lo + 0.3, z_base_hi - 0.6)

    zf = spec.z_fraction if z_fraction is None else z_fraction
    z = z_base_lo + zf * (z_base_hi - z_base_lo)

    if spec.orientation_type == "parallel":
        base_pos = geom._flat_orientation_from_principal_axis(
            ads_pos,
            normal,
            azimuth_in_plane_deg=spec.azimuth_in_plane_deg,
            face_flip=spec.face_flip,
        )
    else:
        base_pos = geom._surface_aligned_rotation(
            ads_pos,
            normal,
            0,
            symbols,
            en_atom_index=spec.en_atom_index,
        )
        if base_pos is None:
            base_pos = ads_pos.copy()

    rotated_pos = geom._rotation_with_tilt(
        base_pos, normal, spec.tilt_deg, spec.azimuth_deg
    )

    # For envelope sites, use site-local z; for planar use global surface_z
    surface_ref = (
        float(site["z"])
        if site and "z" in site and spec.site_type == "envelope"
        else surface_z
    )
    test = rotated_pos.copy()
    test[:, 0] += x
    test[:, 1] += y
    test[:, 2] += surface_ref + z

    adsorbate.set_positions(test)
    ok, _ = geom.check_initial_placement_distance(
        adsorbate,
        slab,
        min_distance=config.min_initial_distance,
        min_contact_ratio=config.min_contact_ratio,
        max_initial_distance=config.max_initial_distance,
    )
    if not ok:
        return None

    slab_indices = None
    if site is not None and "slab_indices" in site:
        slab_indices = tuple(site["slab_indices"])
    descriptor = PlacementDescriptor(
        conformer_index=spec.conformer_index,
        orientation_type=spec.orientation_type,
        face_flip=spec.face_flip,
        en_atom_index=spec.en_atom_index,
        site_index=spec.site_index,
        site_type=spec.site_type,
        tilt_deg=spec.tilt_deg,
        azimuth_deg=spec.azimuth_deg,
        azimuth_in_plane_deg=spec.azimuth_in_plane_deg,
        z_fraction=zf,
        placement_index=spec.placement_index,
        x=x,
        y=y,
        z=z,
        shape=shape,
        slab_indices=slab_indices,
    )
    return adsorbate, descriptor


def enumerate_placement_specs(
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    smiles: str | None,
    n_desired: int,
    filter_spec: Callable[[PlacementSpec], bool] | None = None,
) -> list[PlacementSpec]:
    """Enumerate placement specs for diverse sampling.

    Builds a stratified set of specs covering conformers, orientation types,
    face flip, electronegative atoms, sites, tilt, and azimuth.
    """
    if not conformers:
        return []

    unique_sites, use_sites = _get_unique_sites_for_specs(slab, config)
    site_indices = (
        list(range(len(unique_sites))) if use_sites and unique_sites else [-1]
    )
    n_conformers = len(conformers)

    ads_pos = conformers[0].get_positions().copy()
    ads_pos -= np.mean(ads_pos, axis=0)
    shape, _, _ = geom._classify_molecule_shape(ads_pos)
    symbols = conformers[0].get_chemical_symbols()
    binders = geom._binding_atom_candidates(symbols)
    n_binders = max(len(binders), 1)
    flat_aromatic = _is_flat_aromatic(shape, smiles, symbols)

    tilt_full = [0.0, 15.0, 30.0, 45.0, 60.0, 90.0]
    tilt_parallel = [0.0, 15.0, 30.0]
    azimuth = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    az_ip = [0.0, 90.0, 180.0, 270.0]
    zf = [0.1, 0.3, 0.5, 0.7, 0.9]
    parallel_fraction = config.flat_aromatic_parallel_fraction

    def site_type_for(site_idx: int) -> str | None:
        if not use_sites or site_idx < 0 or site_idx >= len(unique_sites):
            return None
        return str(unique_sites[site_idx]["site_type"])

    def make_spec(
        placement_index: int,
        conf_idx: int,
        orient: str,
        face_flip: bool,
        en_idx: int | None,
        site_idx: int,
        tilt: float,
        az: float,
        az_ip_val: float,
        zf_val: float,
    ) -> PlacementSpec:
        return PlacementSpec(
            conformer_index=conf_idx,
            orientation_type=orient,
            face_flip=face_flip,
            en_atom_index=en_idx,
            site_index=site_idx,
            site_type=site_type_for(site_idx),
            tilt_deg=tilt,
            azimuth_deg=az,
            azimuth_in_plane_deg=az_ip_val,
            z_fraction=zf_val,
            placement_index=placement_index,
        )

    specs: list[PlacementSpec] = []
    pid = 0

    if flat_aromatic:
        parallel_items = list(
            itertools.product(
                range(n_conformers),
                [False, True],
                site_indices,
                tilt_parallel,
                azimuth,
                az_ip,
                zf,
            )
        )
        en_down_items = list(
            itertools.product(
                range(n_conformers),
                range(n_binders),
                site_indices,
                tilt_full,
                azimuth,
                zf,
            )
        )
        n_par = max(1, int(n_desired * parallel_fraction))
        n_en = max(1, n_desired - n_par)
        for _i, (ci, ff, si, tl, azv, aip, zfv) in enumerate(parallel_items[:n_par]):
            if len(specs) >= n_desired:
                break
            spec = make_spec(pid, ci, "parallel", ff, None, si, tl, azv, aip, zfv)
            pid += 1
            if filter_spec is None or filter_spec(spec):
                specs.append(spec)
        for _i, (ci, ei, si, tl, azv, zfv) in enumerate(en_down_items[:n_en]):
            if len(specs) >= n_desired:
                break
            spec = make_spec(
                pid,
                ci,
                "EN-down",
                False,
                ei if n_binders > 1 else None,
                si,
                tl,
                azv,
                0.0,
                zfv,
            )
            pid += 1
            if filter_spec is None or filter_spec(spec):
                specs.append(spec)
    else:
        orient = "vertical" if shape == "linear" else "round"
        items = list(
            itertools.product(
                range(n_conformers),
                site_indices,
                tilt_full,
                azimuth,
                zf,
            )
        )
        for ci, si, tl, azv, zfv in items:
            if len(specs) >= n_desired:
                break
            spec = make_spec(pid, ci, orient, False, None, si, tl, azv, 0.0, zfv)
            pid += 1
            if filter_spec is None or filter_spec(spec):
                specs.append(spec)

    return specs[:n_desired]


def generate_placement_from_spec(
    spec: PlacementSpec,
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    smiles: str | None = None,
) -> tuple[Atoms, PlacementDescriptor] | None:
    """Generate adsorbate placement from spec. Returns (adsorbate, descriptor) or None."""
    if not conformers:
        return None
    adsorbate = conformers[spec.conformer_index % len(conformers)].copy()
    result = _apply_placement_core(adsorbate, spec, slab, config, smiles)
    if result is not None:
        return result
    # Retry with fixed z_fractions
    for zf in [0.2, 0.4, 0.6, 0.8]:
        adsorbate = conformers[spec.conformer_index % len(conformers)].copy()
        result = _apply_placement_core(
            adsorbate, spec, slab, config, smiles, z_fraction=zf
        )
        if result is not None:
            return result
    return None


def generate_placement_from_descriptor(
    descriptor: PlacementDescriptor,
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    smiles: str | None = None,
) -> Atoms | None:
    """Reproduce placement deterministically from descriptor."""
    if not conformers:
        return None
    adsorbate = conformers[descriptor.conformer_index % len(conformers)].copy()
    ads_pos = adsorbate.get_positions().copy()
    ads_pos -= np.mean(ads_pos, axis=0)

    surface_z = float(np.max(slab.get_positions()[:, 2]))
    # For envelope sites, descriptor.z is offset from site-local z, not surface_z
    if descriptor.site_type == "envelope" and descriptor.site_index >= 0:
        unique_sites, _ = _get_unique_sites_for_specs(slab, config)
        if (
            unique_sites
            and descriptor.site_index < len(unique_sites)
            and "z" in unique_sites[descriptor.site_index]
        ):
            surface_z = float(unique_sites[descriptor.site_index]["z"])
    normal = np.array([0.0, 0.0, 1.0])
    symbols = adsorbate.get_chemical_symbols()

    if descriptor.orientation_type == "parallel":
        base_pos = geom._flat_orientation_from_principal_axis(
            ads_pos,
            normal,
            azimuth_in_plane_deg=descriptor.azimuth_in_plane_deg,
            face_flip=descriptor.face_flip,
        )
    else:
        base_pos = geom._surface_aligned_rotation(
            ads_pos,
            normal,
            0,
            symbols,
            en_atom_index=descriptor.en_atom_index,
        )
        if base_pos is None:
            base_pos = ads_pos.copy()

    rotated_pos = geom._rotation_with_tilt(
        base_pos, normal, descriptor.tilt_deg, descriptor.azimuth_deg
    )

    test = rotated_pos.copy()
    test[:, 0] += descriptor.x
    test[:, 1] += descriptor.y
    test[:, 2] += surface_z + descriptor.z

    adsorbate.set_positions(test)
    adsorbate.set_cell(slab.get_cell())
    adsorbate.set_pbc(slab.get_pbc())
    return adsorbate


def _is_dissociable_diatomic(adsorbate: Atoms) -> bool:
    """True if molecule is a homonuclear diatomic (e.g. H2, O2, N2)."""
    syms = adsorbate.get_chemical_symbols()
    return len(syms) == 2 and syms[0] == syms[1]


def _generate_dissociative_placement(
    adsorbate: Atoms,
    slab: Atoms,
    placement_id: int,
    config: AdsorptionConfig,
    rng: random.Random,
    slab_for_sites: Atoms | None = None,
) -> Atoms | None:
    """Place fragments at different hollow sites for dissociative adsorption.

    When slab_for_sites is provided (e.g. base metal slab in saturation), use it
    for hollow-site detection; otherwise use slab. Always use slab for distance
    checks so existing adsorbates are respected.
    """
    if not _is_dissociable_diatomic(adsorbate):
        return None

    sites_slab = slab_for_sites if slab_for_sites is not None else slab
    syms = adsorbate.get_chemical_symbols()
    surface_z = float(np.max(sites_slab.get_positions()[:, 2]))
    cell = slab.get_cell()
    z_lo, z_hi = config.placement_z_range
    z_offset = rng.uniform(z_lo, z_hi)

    hollow_sites = sts.get_hollow_sites_for_adatoms(
        sites_slab,
        top_layer_tolerance=config.top_layer_tolerance,
        dedup_tolerance=0.2,
    )
    if len(hollow_sites) < 2:
        return None

    # Exclude sites occupied by existing adsorbates (saturation step 2+)
    if slab_for_sites is not None and len(slab) > len(slab_for_sites):
        adsorbate_positions = slab.get_positions()[len(slab_for_sites) :]
        pbc_slab = [True, True, False]
        available_sites: list[np.ndarray] = []
        for h_xy in hollow_sites:
            site_pos = np.append(h_xy, surface_z)
            d = geom.calculate_min_distance(
                site_pos.reshape(1, 3),
                adsorbate_positions,
                cell=cell,
                pbc=pbc_slab,
            )
            if d >= config.min_initial_distance:
                available_sites.append(h_xy)
        hollow_sites = available_sites
        if len(hollow_sites) < 2:
            return None

    pbc = [True, True, False]
    min_fragment_sep = 1.0
    max_adjacent_sep = 2.8
    pairs: list[tuple[int, int]] = []
    for i in range(len(hollow_sites)):
        for j in range(i + 1, len(hollow_sites)):
            xy_i = np.append(hollow_sites[i], surface_z)
            xy_j = np.append(hollow_sites[j], surface_z)
            _, dists = find_mic((xy_i - xy_j).reshape(1, 3), cell, pbc=pbc)
            d = float(dists[0])
            if min_fragment_sep <= d <= max_adjacent_sep:
                pairs.append((i, j))

    if not pairs:
        return None

    idx = placement_id % len(pairs)
    i, j = pairs[idx]
    pos1 = np.append(hollow_sites[i], surface_z + z_offset)
    pos2 = np.append(hollow_sites[j], surface_z + z_offset)

    result = Atoms(symbols=syms, positions=[pos1, pos2])
    result.set_cell(cell)
    result.set_pbc(slab.get_pbc())
    return result


def generate_conformer_placement(
    conformers: list[Atoms],
    energies: list[float],
    slab: Atoms,
    placement_id: int,
    config: AdsorptionConfig | None = None,
    smiles: str | None = None,
    slab_for_sites: Atoms | None = None,
) -> Atoms | None:
    """Generate a single adsorbate placement on *slab*.

    Each *placement_id* produces a deterministic but distinct combination
    of (conformer, rotation, xy-position, z-height).  When placement_mode
    is "sites" or "auto", xy is chosen from symmetry-unique adsorption
    sites (atop, bridge, hollow) for efficient sampling.

    Shape-based strategies apply in all placement modes (including random):
    linear molecules prioritize vertical (binding-atom-down); flat molecules
    with aromatic rings and electronegative atoms use parallel-short
    (π-stacking) and electronegative-atom-down placements.  Pass *smiles*
    for RDKit-based aromatic detection; otherwise flat+EN is inferred from
    symbols.  Multiple electronegative atoms are cycled across placements.
    Use ``config.flat_aromatic_parallel_fraction`` to tune the parallel/EN-down
    ratio (default 0.5).

    When ``config.skip_topology_check`` is True (e.g. for dissociative
    adsorption like H2 → 2H), a fraction of placements use a dissociative
    strategy: homonuclear diatomics are placed with atoms at different
    hollow sites to seed dissociation.
    """
    if config is None:
        config = AdsorptionConfig()

    if not conformers:
        logger.warning("No conformers available for placement")
        return None

    rng = random.Random(config.seed + placement_id)

    adsorbate = select_conformer_for_placement(
        conformers,
        energies,
        placement_id,
        sampling=config.conformer_sampling,
        temperature=config.boltzmann_temperature,
        rng=rng,
    )

    use_dissociative = config.skip_topology_check and _is_dissociable_diatomic(
        adsorbate
    )
    if use_dissociative:
        diss_ads = _generate_dissociative_placement(
            adsorbate, slab, placement_id, config, rng, slab_for_sites=slab_for_sites
        )
        if diss_ads is not None:
            ok, _ = geom.check_initial_placement_distance(
                diss_ads,
                slab,
                min_distance=config.min_initial_distance,
                min_contact_ratio=config.min_contact_ratio,
                max_initial_distance=config.max_initial_distance,
            )
            if ok:
                return diss_ads

    ads_pos = adsorbate.get_positions().copy()
    ads_pos -= np.mean(ads_pos, axis=0)

    surface_z = float(np.max(slab.get_positions()[:, 2]))
    cell = np.array(slab.get_cell())
    normal = np.array([0.0, 0.0, 1.0])

    x, y, site_type, unique_sites = _get_xy_for_placement(
        placement_id, config, slab, cell, rng
    )

    site = None
    if unique_sites:
        site = unique_sites[placement_id % len(unique_sites)]
    # For envelope sites, use site-local z; for atop/bridge/hollow use global surface_z
    surface_ref = (
        float(site["z"])
        if site and site.get("site_type") == "envelope" and "z" in site
        else surface_z
    )

    def _check(atoms: Atoms) -> tuple[bool, float]:
        return geom.check_initial_placement_distance(
            atoms,
            slab,
            min_distance=config.min_initial_distance,
            min_contact_ratio=config.min_contact_ratio,
            max_initial_distance=config.max_initial_distance,
        )

    z_base_lo, z_base_hi = sts._compute_site_z_base(
        config, slab, site, adsorbate.get_chemical_symbols()
    )
    if site_type and site_type in _SITE_Z_OFFSETS:
        offset = _SITE_Z_OFFSETS[site_type]
        z_base_lo = z_base_lo + offset
        z_base_hi = z_base_hi + offset

    shape, _, _ = geom._classify_molecule_shape(ads_pos)
    flat_aromatic = _is_flat_aromatic(shape, smiles, adsorbate.get_chemical_symbols())

    use_site_rotation = site_type is not None and unique_sites is not None
    n_sites = len(unique_sites) if unique_sites else 1
    rotation_cycle = placement_id // n_sites

    parallel_fraction = config.flat_aromatic_parallel_fraction
    use_parallel_short = (placement_id % _PARALLEL_EN_CYCLE_SIZE) < int(
        _PARALLEL_EN_CYCLE_SIZE * parallel_fraction
    )

    logger.debug(
        "placement_id=%d shape=%s flat_aromatic=%s use_parallel_short=%s",
        placement_id,
        shape,
        flat_aromatic,
        use_parallel_short,
    )

    if (
        flat_aromatic
        and (site_type is not None or not use_site_rotation)
        and use_parallel_short
    ):
        z_base_lo = max(_PARALLEL_Z_FLOOR_ANGSTROM, z_base_lo - 0.4)
        z_base_hi = max(z_base_lo + 0.3, z_base_hi - 0.6)

    tilt_angles_full = [0.0, 15.0, 30.0, 45.0, 60.0, 90.0]
    tilt_angles_parallel = [0.0, 15.0, 30.0]
    azimuth_angles = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]

    symbols = adsorbate.get_chemical_symbols()
    binders = geom._binding_atom_candidates(symbols)
    n_binders = len(binders)
    en_atom_index: int | None = None
    if n_binders > 1 and not use_parallel_short:
        en_atom_index = rotation_cycle % n_binders

    if use_site_rotation:
        n_orientations = 2 * len(tilt_angles_full) * len(azimuth_angles)
        flat_idx = (placement_id // n_sites) % n_orientations
        n_flat = len(tilt_angles_full) * len(azimuth_angles)

        if shape == "linear":
            use_flat = flat_idx < int(n_orientations * (1 - _LINEAR_VERTICAL_FRACTION))
        elif flat_aromatic:
            use_flat = use_parallel_short
        else:
            use_flat = flat_idx < n_flat

        if use_flat:
            base_pos = geom._flat_orientation_from_principal_axis(
                ads_pos,
                normal,
                azimuth_in_plane_deg=(placement_id % 36) * 10.0,
                face_flip=(placement_id % 2 == 1) if flat_aromatic else False,
            )
            tilt_angles = (
                tilt_angles_parallel
                if flat_aromatic and use_parallel_short
                else tilt_angles_full
            )
        else:
            base_pos = geom._surface_aligned_rotation(
                ads_pos, normal, placement_id, symbols, en_atom_index=en_atom_index
            )
            if base_pos is None:
                base_pos = ads_pos.copy()
            tilt_angles = tilt_angles_full
        n_tilt = len(tilt_angles)
        n_az = len(azimuth_angles)
        rot_idx = (placement_id // n_sites) % (n_tilt * n_az)
        tilt_idx = rot_idx % n_tilt
        az_idx = rot_idx // n_tilt
        tilt = tilt_angles[tilt_idx]
        az = azimuth_angles[az_idx]
        rotated_pos = geom._rotation_with_tilt(base_pos, normal, tilt, az)
        jitter = 0.02 * (rng.random() - 0.5)
        rotated_pos = rotated_pos + jitter * np.ones_like(rotated_pos)
    else:
        n_flat = len(tilt_angles_full) * len(azimuth_angles)
        if shape == "linear":
            use_flat = (rotation_cycle % _LINEAR_VERTICAL_CYCLE) != 0
        elif flat_aromatic:
            use_flat = use_parallel_short
        else:
            use_flat = (rotation_cycle % 2) == 0

        if use_flat:
            base_pos = geom._flat_orientation_from_principal_axis(
                ads_pos,
                normal,
                azimuth_in_plane_deg=(placement_id % 36) * 10.0,
                face_flip=(placement_id % 2 == 1) if flat_aromatic else False,
            )
            tilt_angles = (
                tilt_angles_parallel
                if flat_aromatic and use_parallel_short
                else tilt_angles_full
            )
        else:
            base_pos = geom._surface_aligned_rotation(
                ads_pos, normal, placement_id, symbols, en_atom_index=en_atom_index
            )
            if base_pos is None:
                base_pos = ads_pos.copy()
            tilt_angles = tilt_angles_full
        n_tilt = len(tilt_angles)
        n_az = len(azimuth_angles)
        rot_idx = placement_id % (n_tilt * n_az)
        tilt_idx = rot_idx % n_tilt
        az_idx = rot_idx // n_tilt
        tilt = tilt_angles[tilt_idx]
        az = azimuth_angles[az_idx]
        rotated_pos = geom._rotation_with_tilt(base_pos, normal, tilt, az)

    max_attempts = 20
    for attempt in range(max_attempts):
        z_extra = rng.uniform(z_base_lo, z_base_hi)
        if attempt >= 10:
            z_extra += (attempt - 9) * 0.5

        test = rotated_pos.copy()
        test[:, 0] += x
        test[:, 1] += y
        test[:, 2] += surface_ref + z_extra

        adsorbate.set_positions(test)
        ok, _ = _check(adsorbate)
        if ok:
            return adsorbate

        if flat_aromatic and use_parallel_short:
            logger.debug(
                "Parallel placement failed initial distance (placement_id=%d, attempt=%d)",
                placement_id,
                attempt,
            )

        if not use_site_rotation and attempt % 3 == 2:
            rot = geom._random_rotation_matrix(rng)
            rotated_pos = (rot @ ads_pos.T).T

    if flat_aromatic and use_parallel_short:
        z_lo, z_hi = config.placement_z_range
        for attempt in range(10):
            x, y, _, _ = _get_xy_for_placement(placement_id, config, slab, cell, rng)
            z_extra = rng.uniform(
                z_lo + 0.5 + attempt * 0.3,
                z_hi + 0.5 + attempt * 0.5,
            )
            test = rotated_pos.copy()
            test[:, 0] += x
            test[:, 1] += y
            test[:, 2] += surface_ref + z_extra

            adsorbate.set_positions(test)
            ok, _ = _check(adsorbate)
            if ok:
                return adsorbate

    best_rot_pos, _ = geom._principal_axis_rotation(
        ads_pos, np.array([0.0, 0.0, 1.0]), placement_id
    )
    if best_rot_pos is not None:
        for attempt in range(10):
            x, y, _, _ = _get_xy_for_placement(placement_id, config, slab, cell, rng)
            z_extra = rng.uniform(
                config.placement_z_range[0] + attempt * 0.3,
                config.placement_z_range[1] + attempt * 0.5,
            )
            test = best_rot_pos.copy()
            test[:, 0] += x
            test[:, 1] += y
            test[:, 2] += surface_ref + z_extra

            adsorbate.set_positions(test)
            ok, _ = _check(adsorbate)
            if ok:
                return adsorbate

    logger.warning(
        "No valid placement found for conformer (placement_id=%d)", placement_id
    )
    return None
