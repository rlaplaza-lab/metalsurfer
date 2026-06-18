#!/usr/bin/env python3
"""Compute binding energy of vanillin on H-saturated Ni(111).

Loads the saturated slab from H2 saturation run (e.g. scripts/h2_ni111_saturation.py)
and runs vanillin adsorption using envelope placement for the non-planar H-covered surface.

Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"

Prerequisites:
  - Run scripts/h2_ni111_saturation.py first to generate the saturated slab
  - Or provide path to a pre-adsorbed slab XYZ file
"""

import argparse
import os

import pandas as pd
from ase.io import read

from metalsurfer import AdsorptionConfig, configure_logging, run_adsorption
from metalsurfer.surface_prep import prepare_substrate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute vanillin binding on H2/Ni(111) from a saved saturation state. "
            "By default uses final saturated slab; pass --step N for intermediate state."
        )
    )
    parser.add_argument(
        "--saturation-dir",
        default="results_h2_ni111_saturation",
        help="Directory produced by scripts/h2_ni111_saturation.py",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="1-based saturation step to use as slab input. Defaults to final saturated slab.",
    )
    return parser.parse_args()


def resolve_saturation_slab_path(saturation_dir: str, step: int | None) -> str:
    xyz_root = f"{saturation_dir}/xyz_structures/H2_saturation"
    summary_path = f"{saturation_dir}/saturation_summary.csv"
    details_path = f"{saturation_dir}/saturation_details.csv"

    if step is None:
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"Missing saturation summary: {summary_path}")
        summary_df = pd.read_csv(summary_path)
        row = summary_df[summary_df["molecule"] == "H2"]
        if row.empty:
            raise ValueError("No H2 row found in saturation_summary.csv")
        final_path = str(row.iloc[0]["final_slab_path"]).strip()
        if not final_path:
            raise ValueError(
                "H2 saturation has no final slab path (likely not saturated yet)"
            )
        if not os.path.exists(final_path):
            raise FileNotFoundError(f"Final slab path does not exist: {final_path}")
        return final_path

    if step <= 0:
        raise ValueError("--step must be >= 1")
    if not os.path.exists(details_path):
        raise FileNotFoundError(f"Missing saturation details: {details_path}")

    details_df = pd.read_csv(details_path)
    rows = details_df[(details_df["molecule"] == "H2") & (details_df["step"] == step)]
    if rows.empty:
        raise ValueError(f"H2 step {step} not found in saturation_details.csv")

    step_path = str(rows.iloc[0]["step_structure_path"]).strip()
    if not step_path:
        raise ValueError(f"H2 step {step} has empty step_structure_path")
    if not os.path.exists(step_path):
        raise FileNotFoundError(
            f"Step structure path does not exist: {step_path} (root: {xyz_root})"
        )
    return step_path


def main():
    configure_logging(default_level="INFO")
    args = parse_args()
    if args.step is None:
        results_subdir = "vanillin_h_saturated_ni111_final"
    else:
        results_subdir = f"vanillin_h_saturated_ni111_step_{args.step:03d}"
    results_dir = f"results_{results_subdir}"
    saturation_dir = args.saturation_dir
    try:
        saturated_xyz = resolve_saturation_slab_path(saturation_dir, args.step)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return 1

    # Load H-saturated slab and finalize for campaign APIs (PBC, constraints).
    saturated_atoms = read(saturated_xyz)

    config = AdsorptionConfig(
        material_type="slab",
        model_name="uma-m-1p1",
        seed=42,
        num_conformers=10,
        num_placements=250,
        autobatcher_max_memory_padding=0.8,
        device="cuda",
        skip_topology_check=False,
        skip_desorption_check=False,
        stage1_steps=50,
        stage2_steps=500,
        top_layer_tolerance=2.0,  # Include top metal + H in top layer for envelope
    )

    slab = prepare_substrate(
        slab=saturated_atoms,
        config=config,
        results_dir=results_dir,
        align=False,
        relax_top_layer=True,  # Allow top layer to relax with adsorbate
    )

    # Clean metal slab for frozen indices (subsurface metal only)
    base_slab = prepare_substrate(
        bulk_id="mp-23",
        miller_indices=(1, 1, 1),
        supercell=(3, 3, 1),
        config=config,
        results_dir=results_dir,
        relax_top_layer=True,
    )

    smiles = "c1(C=O)cc(OC)c(O)cc1"
    campaign = run_adsorption(
        slab=slab,
        molecules=[(smiles, "vanillin")],
        config=config,
        surface_type=results_subdir,
        system_name="Ni_111_H_saturated",
        process_kwargs={"base_slab_for_frozen": base_slab.atoms},
    )
    summary = campaign.molecule_summaries[0]
    if summary.best_adsorption_energy is not None:
        print(
            f"\nBinding energy of vanillin on H-saturated Ni(111): "
            f"{summary.best_adsorption_energy:.4f} eV"
        )
        print("  (E_ads = E(slab+vanillin) - E(saturated slab) - E(vanillin))")
        print(
            f"  Orientations: {summary.n_parallel}/{summary.n_valid_placements} parallel, "
            f"{summary.n_endown} EN-down"
        )
        print(f"  {campaign.format_results_saved_line(results_dir=results_dir)}")
    else:
        print("No valid placements found.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
