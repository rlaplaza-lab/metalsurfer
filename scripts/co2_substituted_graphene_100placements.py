#!/usr/bin/env python3
"""Run CO2 adsorption on an oxidized graphene monolayer (N-doped + O adatoms) for many placements.

Produces results_co2_graphene/adsorption_energies_detailed.csv for BO benchmarking.
Use conda env metalsurfer: conda run -n metalsurfer python scripts/co2_substituted_graphene_100placements.py

Slab: graphite mp-48 (0,0,1) → monolayer → 10% C→N substitution → ~10% O adatoms at hollow sites
(oxidized / N-doped graphene) so the surface is heterogeneous and placement matters.

If GPU runs out of memory, set METALSURFER_DEVICE=cpu or pass --device cpu.
"""

import argparse
import logging
import os

from ase import Atoms

from metalsurfer import AdsorptionConfig, run_adsorption
from metalsurfer._logging import configure_logging
from metalsurfer.surface_prep import (
    create_slab_from_bulk,
    deposit_adatoms,
    substitute_alloy,
)


def _configure_logging(debug: bool = False) -> None:
    level_name = "DEBUG" if debug else "INFO"
    configure_logging(default_level=level_name)
    if debug:
        logging.getLogger("metalsurfer.filters").setLevel(logging.DEBUG)
        logging.getLogger("metalsurfer.workflow").setLevel(logging.DEBUG)


SURFACE_TYPE = "co2_graphene"
RESULTS_DIR = f"results_{SURFACE_TYPE}"
# Graphite (Materials Project); (0,0,1) = basal plane
BULK_ID = "mp-48"
MILLER = (0, 0, 1)
SUPERCELL = (3, 3, 1)  # larger slab for more sites and O adatoms (oxidized graphene)
TOP_LAYER_TOLERANCE = 0.5  # Angstrom; atoms within this of z_max form the top layer
N_SUBSTITUTE_FRACTION = 0.10  # 10% C -> N for N-doped graphene
O_ADATOM_COVERAGE = 0.12  # fraction of hollow sites with O (oxidized graphene)
CO2_SMILES = "O=C=O"
NUM_PLACEMENTS = 250


def slab_to_monolayer(atoms: Atoms, z_tolerance: float = 0.5) -> Atoms:
    """Keep only the top layer of a slab (atoms with z >= z_max - z_tolerance)."""
    pos = atoms.get_positions()
    z_max = float(pos[:, 2].max())
    mask = pos[:, 2] >= (z_max - z_tolerance)
    return atoms[mask].copy()


def main():
    parser = argparse.ArgumentParser(
        description="CO2 on graphene monolayer. "
        "You may get fewer than num_placements: placement generation can skip attempts that fail "
        "(e.g. no valid site, overlap, or geometry), so 'Generated N/num_placements' can have N < num_placements."
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("METALSURFER_DEVICE", "cuda"),
        help="Device: cuda or cpu",
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    parser.add_argument(
        "--no-debug", action="store_true", help="Disable DEBUG (default: debug is on)"
    )
    args = parser.parse_args()
    debug = (not args.no_debug) or (
        os.environ.get("METALSURFER_DEBUG", "").lower() in ("1", "true", "yes")
    )
    _configure_logging(debug=debug)
    device = args.device if args.device in ("cuda", "cpu") else "cuda"

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1) Build graphite slab, then take single layer (graphene monolayer)
    slab = create_slab_from_bulk(
        bulk_id=BULK_ID,
        miller_indices=MILLER,
        supercell=SUPERCELL,
        results_dir=RESULTS_DIR,
    )
    monolayer = slab_to_monolayer(slab.atoms, z_tolerance=TOP_LAYER_TOLERANCE)
    slab = monolayer
    logging.info(
        "Graphene monolayer: %d atoms (from %s %s supercell %s)",
        len(slab.atoms),
        BULK_ID,
        MILLER,
        SUPERCELL,
    )

    config = AdsorptionConfig(
        material_type="slab",
        model_name="uma-s-1p1",
        seed=42,
        num_conformers=10,
        num_placements=NUM_PLACEMENTS,
        bo_enabled=False,
        device=device,
        fmax=0.05,
        stage1_steps=100,
        stage2_steps=350,
        max_force_convergence=0.08,
        auto_resize_slab=True,
        min_pbc_image_separation=8.0,
        skip_topology_check=True,
        # Graphene monolayer has no sub-surface: without this, frozen=0 (all atoms are "top layer")
        relax_top_layer=False,
        # All placements (e.g. 250) fit in one GPU pass; skip memory estimation and use more VRAM
        autobatcher_max_memory_padding=0.8,
        autobatcher_max_memory_scaler=500,
    )

    # 2) Substitute 10% C -> N (N-doped graphene).
    slab = substitute_alloy(
        slab,
        host_symbol="C",
        guest_symbol="N",
        guest_fraction=N_SUBSTITUTE_FRACTION,
        calculator=None,
        relax=False,
        config=config,
        results_dir=RESULTS_DIR,
        seed=config.seed,
    )
    logging.info("N-doped slab: %d atoms", len(slab.atoms))

    # 3) Deposit O at hollow sites (oxidized graphene). calculator=None => one random variant.
    slab = deposit_adatoms(
        slab,
        adatom_symbol="O",
        coverage_fraction=O_ADATOM_COVERAGE,
        calculator=None,
        config=config,
        results_dir=RESULTS_DIR,
        seed=config.seed,
    )
    logging.info(
        "Oxidized slab: %d atoms (C+N monolayer + O adatoms)",
        len(slab.atoms),
    )

    campaign = run_adsorption(
        slab=slab,
        molecules=[(CO2_SMILES, "CO2")],
        config=config,
        surface_type=SURFACE_TYPE,
        system_name="graphene_N10_O12",
    )
    summary = campaign.molecule_summaries[0]
    if summary.best_adsorption_energy is not None:
        logging.info(
            "Best E_ads = %.4f eV; %d results -> %s",
            summary.best_adsorption_energy,
            summary.n_valid_placements,
            RESULTS_DIR,
        )
    else:
        logging.warning("No valid placements.")

    return 0 if summary.best_adsorption_energy is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
