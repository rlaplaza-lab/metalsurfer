#!/usr/bin/env python3
"""Compute binding energies of furanic molecules on graphene oxide (GO) monolayer.

Molecules: HMF, BHMF, BHMTHF, 5-MF, MFA, DMF, MTHFA, DMTHF.

Slab: Random GO model R1 from Mouhat et al., Nature Commun. 2020 (citable-data).
Loaded from https://github.com/fxcoudert/citable-data. Top GO layer is relaxed.

Uses BO pipeline: 300 placements max in passes of 100 (100 initial random + up to 2 BO passes of 100).
Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"
"""

import argparse
import logging
import os
from io import StringIO
from urllib.request import urlopen

import numpy as np
from ase.io import read

from metalsurfer import AdsorptionConfig, configure_logging, run_adsorption_bo
from metalsurfer.surface_prep import prepare_substrate

# List of smiles and molecule name pairs
MOLECULES = [
    ("C(=O)C1OC(CO[H])=CC=1", "HMF"),
    ("C(O[H])C1OC(CO[H])=CC=1", "BHMF"),
    ("C(O[H])C1OC(CO[H])CC1", "BHMTHF"),
    ("C(=O)C1OC(C)=CC=1", "5-MF"),
    ("C(O[H])C1OC(C)=CC=1", "MFA"),
    ("C1(C)OC(C)=CC=1", "DMF"),
    ("C(O[H])C1OC(C)CC1", "MTHFA"),
    ("CC1OC(C)CC1", "DMTHF"),
]


def _configure_logging(debug: bool = False) -> None:
    level_name = "DEBUG" if debug else "INFO"
    configure_logging(default_level=level_name)
    if debug:
        logging.getLogger("metalsurfer.filters").setLevel(logging.DEBUG)
        logging.getLogger("metalsurfer.workflow").setLevel(logging.DEBUG)


CITABLE_BASE = (
    "https://raw.githubusercontent.com/fxcoudert/citable-data/master"
    "/122-Mouhat_NatureCommun_2020/models/GO"
)
MIN_CALCULATOR_CELL_C_ANG = 18.0


def _ensure_calculator_safe_pbc_and_vacuum(atoms) -> None:
    """Use slab PBC and ensure enough z separation for the MLIP calculator."""
    atoms.set_pbc([True, True, False])
    cell = atoms.get_cell().copy()
    c_vec = np.array(cell[2], dtype=float)
    c_len = float(np.linalg.norm(c_vec))
    if c_len >= MIN_CALCULATOR_CELL_C_ANG:
        return
    if c_len < 1e-8:
        cell[2] = [0.0, 0.0, MIN_CALCULATOR_CELL_C_ANG]
    else:
        cell[2] = c_vec * (MIN_CALCULATOR_CELL_C_ANG / c_len)
    atoms.set_cell(cell, scale_atoms=False)
    logging.info(
        "Increased GO cell c-vector from %.3f A to %.3f A for image separation",
        c_len,
        MIN_CALCULATOR_CELL_C_ANG,
    )


def _load_go_slab(subdir: str):
    """Load GO monolayer from Mouhat et al. Nature Commun. 2020 (citable-data)."""
    xyz_url = f"{CITABLE_BASE}/{subdir}/GO.xyz"
    cell_url = f"{CITABLE_BASE}/{subdir}/cell_parameters.dat"
    with urlopen(xyz_url) as resp:
        atoms = read(StringIO(resp.read().decode()), format="xyz")
    with urlopen(cell_url) as resp:
        line = resp.read().decode().splitlines()[0]
    parts = line.split()
    a, b, c = float(parts[2]), float(parts[3]), float(parts[4])
    atoms.set_cell([a, b, c])
    _ensure_calculator_safe_pbc_and_vacuum(atoms)
    return atoms


def main():
    parser = argparse.ArgumentParser(
        description="Furanic molecules on graphene oxide with BO (up to 300 placements, passes of 100)"
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("METALSURFER_DEVICE", "cuda"),
        help="Device: cuda or cpu",
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()
    debug = args.debug or (
        os.environ.get("METALSURFER_DEBUG", "").lower() in ("1", "true", "yes")
    )
    _configure_logging(debug=debug)
    device = args.device if args.device in ("cuda", "cpu") else "cuda"

    results_subdir = "furanics_go_r1"
    results_dir = f"results_{results_subdir}"
    os.makedirs(results_dir, exist_ok=True)

    config = AdsorptionConfig(
        material_type="slab",
        slab_relaxation_mode="none",  # keep published GO geometry
        model_name="uma-s-1p1",
        seed=42,
        num_conformers=10,
        num_placements=250,
        autobatcher_max_memory_padding=0.8,
        device=device,
        skip_topology_check=False,
        skip_desorption_check=False,
        stage1_steps=50,
        stage2_steps=500,
        debug_write_initial_placements=False,
        bo_enabled=True,
        bo_initial_random=100,
        bo_batch_size=100,
        # bo_total_budget = acquisition batches after initial (not total evals).
        bo_total_budget=2,  # 100 initial + 2×100 ≈ 300 evaluations
    )

    slab = _load_go_slab("random/R1")
    logging.info("GO slab (random R1): %d atoms", len(slab))

    slab = prepare_substrate(
        slab=slab,
        config=config,
        results_dir=results_dir,
        relax_top_layer=True,  # Allow top GO layer to relax with adsorbate
    )

    campaign = run_adsorption_bo(
        slab=slab,
        molecules=MOLECULES,
        config=config,
        surface_type=results_subdir,
        system_name="GO_R1",
    )

    print()
    print(
        campaign.format_summary(
            title="Binding energy summary (graphene oxide, random R1)",
            results_dir=results_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
