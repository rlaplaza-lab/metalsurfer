"""Deterministic reconstruction of simulation inputs from PlacementRecord.

Given a :class:`PlacementRecord` (or its serialized form), this module
rebuilds the full slab+adsorbate structure and calculator settings so
that the binding energy can be recomputed identically.
"""

import logging

from ase import Atoms

from ..config import AdsorptionConfig
from ..models import PlacementDescriptor
from ..placement.generators import generate_placement_from_descriptor
from .schema import PlacementRecord

logger = logging.getLogger(__name__)


def record_to_placement_descriptor(record: PlacementRecord) -> PlacementDescriptor:
    """Extract a PlacementDescriptor from a PlacementRecord."""
    return PlacementDescriptor(
        conformer_index=record.conformer_index,
        orientation_type=record.orientation_type,
        face_flip=record.face_flip,
        en_atom_index=record.en_atom_index,
        site_index=record.site_index,
        site_type=record.site_type,
        tilt_deg=record.tilt_deg,
        azimuth_deg=record.azimuth_deg,
        azimuth_in_plane_deg=record.azimuth_in_plane_deg,
        z_fraction=record.z_fraction,
        placement_index=record.placement_id,
        x=record.x,
        y=record.y,
        z=record.z,
        shape=record.shape,
        slab_indices=record.slab_indices,
    )


def record_to_config(record: PlacementRecord) -> AdsorptionConfig:
    """Build an AdsorptionConfig that matches the computation context."""
    ctx = record.context
    return AdsorptionConfig(
        model_name=ctx.model_name,
        fmax=ctx.fmax,
        stage1_steps=ctx.stage1_steps,
        stage2_steps=ctx.stage2_steps,
        device=ctx.device,
        seed=ctx.seed,
        placement_z_range=ctx.placement_z_range,
        placement_z_scale_by_covalent_radius=ctx.placement_z_scale_by_covalent_radius,
        min_initial_distance=ctx.min_initial_distance,
        min_contact_ratio=ctx.min_contact_ratio,
        top_layer_tolerance=ctx.top_layer_tolerance,
        relax_top_layer=ctx.relax_top_layer,
    )


def reconstruct_placement(
    record: PlacementRecord,
    conformers: list[Atoms],
    slab: Atoms,
    smiles: str | None = None,
) -> Atoms | None:
    """Rebuild the pre-optimization adsorbate structure from a record.

    Uses :func:`generate_placement_from_descriptor` which deterministically
    applies the stored orientation, tilt, azimuth, and position to reproduce
    the initial slab+adsorbate geometry.

    Parameters
    ----------
    record : PlacementRecord
        The stored record to reproduce.
    conformers : list[Atoms]
        Conformer library (same as used in the original run).
    slab : Atoms
        The clean slab (same as used in the original run).
    smiles : str, optional
        SMILES string for aromatic detection. Falls back to record.smiles.

    Returns
    -------
    Atoms or None
        The adsorbate structure with positions matching the original
        placement, or None if reconstruction fails.
    """
    descriptor = record_to_placement_descriptor(record)
    config = record_to_config(record)
    if smiles is None:
        smiles = record.smiles

    adsorbate = generate_placement_from_descriptor(
        descriptor, conformers, slab, config, smiles=smiles
    )
    if adsorbate is None:
        logger.warning(
            "Failed to reconstruct placement for record %s (placement_id=%d)",
            record.record_hash(),
            record.placement_id,
        )
    return adsorbate


def verify_record_reproducibility(
    record: PlacementRecord,
    conformers: list[Atoms],
    slab: Atoms,
    calculator,
    ts_model=None,
    energy_tolerance: float = 0.01,
) -> dict[str, object]:
    """Re-run a single placement and compare to stored energy.

    Returns a dict with keys:
        - ``"reproducible"``: bool
        - ``"stored_energy"``: float (eV)
        - ``"recomputed_energy"``: float or None (eV)
        - ``"energy_diff"``: float or None (eV)
        - ``"record_hash"``: str
    """
    from ..optimization import optimize_adsorbate_slab_batched

    result = {
        "reproducible": False,
        "stored_energy": record.energy_adsorption,
        "recomputed_energy": None,
        "energy_diff": None,
        "record_hash": record.record_hash(),
    }

    adsorbate = reconstruct_placement(record, conformers, slab)
    if adsorbate is None:
        logger.error("Cannot reconstruct placement; verification failed")
        return result

    combined = slab + adsorbate
    combined.set_pbc([True, True, True])
    combined.calc = calculator

    config = record_to_config(record)
    optimized_list = optimize_adsorbate_slab_batched(
        [combined], slab, ts_model, config=config
    )

    if not optimized_list or optimized_list[0] is None:
        logger.error("Optimization failed during verification")
        return result

    opt_atoms = optimized_list[0]
    if opt_atoms.calc is None:
        opt_atoms.calc = calculator
    e_adslab = opt_atoms.get_potential_energy()
    e_ads = e_adslab - record.energy_slab - record.energy_adsorbate

    diff = abs(e_ads - record.energy_adsorption)
    result["recomputed_energy"] = e_ads
    result["energy_diff"] = diff
    result["reproducible"] = diff <= energy_tolerance

    if result["reproducible"]:
        logger.info(
            "Record %s verified: E_ads=%.4f eV (diff=%.4f eV)",
            record.record_hash(),
            e_ads,
            diff,
        )
    else:
        logger.warning(
            "Record %s NOT reproducible: stored=%.4f, recomputed=%.4f (diff=%.4f eV)",
            record.record_hash(),
            record.energy_adsorption,
            e_ads,
            diff,
        )

    return result
