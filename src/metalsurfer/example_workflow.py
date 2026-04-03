#!/usr/bin/env python3
"""Example: screen molecules on surfaces using the canonical metalsurfer API.

Usage:
    python -m metalsurfer.example_workflow run --smiles-file smiles.csv
    python -m metalsurfer.example_workflow run-bo --smiles-file smiles.csv
    python -m metalsurfer.example_workflow saturate --smiles-file smiles.csv

Demonstrates key features:
- Deterministic placement generation via Voronoi tessellation for slabs, nanoparticles, and porous materials.
- Explicit material-type specification: "slab" (default), "nanoparticle" (isolated clusters), or "porous" (3D-periodic).
- Surface engineering: alloy substitution and adatom deposition before screening via prepare_slab().
- Bayesian optimization–guided candidate selection for efficient exploration.
- Sequential saturation simulations to estimate surface loading limits.
"""

import argparse

from metalsurfer import (
    AdsorptionConfig,
    run_adsorption,
    run_adsorption_bo,
    run_saturation,
    run_saturation_bo,
    save_saturation_results,
    write_run_metadata,
)
from metalsurfer._logging import configure_logging
from metalsurfer.cli.cli_output import (
    format_saturation_complete,
    format_screening_complete,
)
from metalsurfer.surface_prep import prepare_slab


def _write_metadata_if_available(
    *,
    run_metadata: dict[str, float],
    surface_type: str,
    config: AdsorptionConfig,
    smiles_file: str,
) -> None:
    """Persist run metadata only when timing/counts were populated."""
    if not run_metadata:
        return
    write_run_metadata(
        surface_type=surface_type,
        config=config,
        smiles_file=smiles_file,
        n_molecules=int(run_metadata["n_molecules"]),
        total_configs=int(run_metadata["total_configs"]),
        t_ref_s=float(run_metadata["t_ref_s"]),
        t_total_s=float(run_metadata["t_total_s"]),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Adsorption screening on arbitrary surfaces"
    )

    # surface
    parser.add_argument(
        "--bulk-id", type=str, default="mp-33", help="Materials Project bulk ID"
    )
    parser.add_argument(
        "--miller", type=int, nargs=3, default=[0, 0, 1], help="Miller indices"
    )
    parser.add_argument(
        "--supercell", type=int, nargs=3, default=[2, 2, 1], help="Supercell repeat"
    )

    # alloy
    parser.add_argument(
        "--alloy-host",
        type=str,
        default=None,
        help="Host element for alloy substitution",
    )
    parser.add_argument(
        "--alloy-guest",
        type=str,
        default=None,
        help="Guest element for alloy substitution",
    )
    parser.add_argument(
        "--alloy-fraction", type=float, default=0.0, help="Guest fraction (0-1)"
    )

    # adatom
    parser.add_argument(
        "--adatom-symbol", type=str, default=None, help="Adatom element symbol"
    )
    parser.add_argument(
        "--adatom-coverage",
        type=float,
        default=0.0,
        help="Adatom coverage fraction (0-1)",
    )

    # primary knobs
    parser.add_argument(
        "--model", type=str, default="uma-s-1p1", help="MLIP model name"
    )
    parser.add_argument(
        "--num-conformers", type=int, default=10, help="RDKit conformers per molecule"
    )
    parser.add_argument(
        "--num-placements", type=int, default=100, help="Placements per molecule"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (cuda or cpu)"
    )

    # I/O
    parser.add_argument(
        "--smiles-file", type=str, default="smiles.csv", help="SMILES CSV file"
    )
    parser.add_argument(
        "--surface-type", type=str, default="manual", help="Label for results directory"
    )
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run", help="Run non-Bayesian screening")
    run_bo_parser = subparsers.add_parser("run-bo", help="Run Bayesian screening")
    run_bo_parser.add_argument("--bo-initial-random", type=int, default=20)
    run_bo_parser.add_argument("--bo-batch-size", type=int, default=10)
    run_bo_parser.add_argument("--bo-total-budget", type=int, default=100)
    run_bo_parser.add_argument(
        "--bo-acquisition", choices=["lcb", "ei", "pi"], default="lcb"
    )
    run_bo_parser.add_argument("--bo-ucb-kappa", type=float, default=1.96)
    run_bo_parser.add_argument(
        "--bo-surrogate",
        choices=["random_forest", "extra_trees", "gradient_boost", "ridge"],
        default="random_forest",
    )
    saturate_parser = subparsers.add_parser("saturate", help="Run saturation screening")
    saturate_parser.add_argument(
        "--bo-enabled", action="store_true", help="Enable Bayesian optimization"
    )

    args = parser.parse_args()
    configure_logging(default_level="INFO")

    config = AdsorptionConfig(
        material_type="slab",  # Explicit material type: "slab", "nanoparticle", or "porous"
        model_name=args.model,
        num_conformers=args.num_conformers,
        num_placements=args.num_placements,
        device=args.device,
        seed=args.seed,
        saturation=args.command == "saturate",
        bo_enabled=args.command == "run-bo"
        or (args.command == "saturate" and getattr(args, "bo_enabled", False)),
        bo_initial_random=getattr(args, "bo_initial_random", 10),
        bo_batch_size=getattr(args, "bo_batch_size", 10),
        bo_total_budget=getattr(args, "bo_total_budget", 100),
        bo_acquisition=getattr(args, "bo_acquisition", "lcb"),
        bo_ucb_kappa=getattr(args, "bo_ucb_kappa", 1.96),
        bo_surrogate=getattr(args, "bo_surrogate", "random_forest"),
    )

    slab = prepare_slab(
        bulk_id=args.bulk_id,
        miller_indices=tuple(args.miller),
        supercell=tuple(args.supercell),
        results_dir=f"results_{args.surface_type}",
        alloy_host=args.alloy_host,
        alloy_guest=args.alloy_guest,
        alloy_fraction=args.alloy_fraction,
        adatom_symbol=args.adatom_symbol,
        adatom_coverage=args.adatom_coverage,
        model_name=args.model,
        device=args.device,
        config=config,
    )

    skip = args.skip_existing and not args.force_rerun
    if args.command == "saturate":
        bo_enabled = getattr(args, "bo_enabled", False)
        run_sat_fn = run_saturation_bo if bo_enabled else run_saturation
        run_metadata: dict[str, float] = {}
        saturation_results = run_sat_fn(
            slab=slab,
            molecules=args.smiles_file,
            config=config,
            surface_type=args.surface_type,
            skip_existing=skip,
            run_metadata_out=run_metadata,
        )
        save_saturation_results(
            saturation_results,
            surface_type=args.surface_type,
            config=config,
        )
        _write_metadata_if_available(
            run_metadata=run_metadata,
            surface_type=args.surface_type,
            config=config,
            smiles_file=args.smiles_file,
        )
        total_steps = sum(len(sr.steps) for sr in saturation_results)
        total_mols = sum(sr.n_molecules_at_saturation for sr in saturation_results)
        print("")
        print(
            format_saturation_complete(
                label=f"Saturation ({len(saturation_results)} molecules)",
                n_molecules_at_saturation=total_mols,
                total_steps=total_steps,
                results_dir=f"results_{args.surface_type}",
            )
        )
    else:
        run_fn = run_adsorption_bo if args.command == "run-bo" else run_adsorption
        run_metadata = {}
        campaign = run_fn(
            slab=slab,
            molecules=args.smiles_file,
            config=config,
            surface_type=args.surface_type,
            skip_existing=skip,
            run_metadata_out=run_metadata,
        )
        _write_metadata_if_available(
            run_metadata=run_metadata,
            surface_type=args.surface_type,
            config=config,
            smiles_file=args.smiles_file,
        )
        print("")
        print(format_screening_complete(campaign.total_configurations))


if __name__ == "__main__":
    main()
