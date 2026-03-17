#!/usr/bin/env python3
"""Example: screen molecules on surfaces. Usage: python -m metalsurfer.example_workflow --smiles-file smiles.csv"""

import argparse
import logging

from metalsurfer import (
    AdsorptionConfig,
    create_slab_from_bulk,
    deposit_adatoms,
    run_saturation_screening,
    run_screening,
    run_screening_bayesian,
    setup_single_model,
    substitute_alloy,
)
from metalsurfer.io_results import (
    save_molecule_results,
    save_saturation_results,
    save_summary_results,
    setup_directories,
    write_run_metadata,
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
    parser.add_argument(
        "--saturation",
        action="store_true",
        help="Sequential saturation: add molecules until best E_ads >= 0",
    )
    parser.add_argument(
        "--bayesian",
        action="store_true",
        help="Use Bayesian optimisation (RF surrogate + UCB) for placement selection",
    )
    parser.add_argument(
        "--bo-initial-random",
        type=int,
        default=20,
        help="BO: initial random placements (default 20)",
    )
    parser.add_argument(
        "--bo-batch-size",
        type=int,
        default=10,
        help="BO: batch size per acquisition step (default 10)",
    )
    parser.add_argument(
        "--bo-total-budget",
        type=int,
        default=100,
        help="BO: total placement evaluations per molecule (default 100)",
    )
    parser.add_argument(
        "--bo-acquisition",
        type=str,
        choices=["lcb", "ei", "pi"],
        default="lcb",
        help="BO: acquisition function (default lcb)",
    )
    parser.add_argument(
        "--bo-ucb-kappa",
        type=float,
        default=1.96,
        help="BO: LCB kappa (default 1.96)",
    )
    parser.add_argument(
        "--bo-surrogate",
        type=str,
        choices=["random_forest", "extra_trees", "gradient_boost", "ridge"],
        default="random_forest",
        help="BO: surrogate model architecture (default random_forest)",
    )

    parser.add_argument("--seed", type=int, default=42, help="Global random seed")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(ctx_prefix)s%(message)s",
        datefmt="%H:%M:%S",
    )
    from metalsurfer._logging import ContextFilter

    logging.getLogger().addFilter(ContextFilter())

    config = AdsorptionConfig(
        model_name=args.model,
        num_conformers=args.num_conformers,
        num_placements=args.num_placements,
        device=args.device,
        seed=args.seed,
        saturation=args.saturation,
        bo_enabled=args.bayesian,
        bo_initial_random=args.bo_initial_random,
        bo_batch_size=args.bo_batch_size,
        bo_total_budget=args.bo_total_budget,
        bo_acquisition=args.bo_acquisition,
        bo_ucb_kappa=args.bo_ucb_kappa,
        bo_surrogate=args.bo_surrogate,
    )

    # build surface
    slab = create_slab_from_bulk(
        bulk_id=args.bulk_id,
        miller_indices=tuple(args.miller),
        supercell=tuple(args.supercell),
        results_dir=f"results_{args.surface_type}",
    )

    # optional alloy substitution
    if args.alloy_guest and args.alloy_fraction > 0:
        host = args.alloy_host
        if host is None:
            host = sorted(set(slab.atoms.get_chemical_symbols()))[0]
        calc, _ = setup_single_model(config.model_name, config.device)
        slab = substitute_alloy(
            slab,
            host_symbol=host,
            guest_symbol=args.alloy_guest,
            guest_fraction=args.alloy_fraction,
            calculator=calc,
            config=config,
            results_dir=f"results_{args.surface_type}",
        )

    # optional adatom deposition
    if args.adatom_symbol and args.adatom_coverage > 0:
        calc, _ = setup_single_model(config.model_name, config.device)
        slab = deposit_adatoms(
            slab,
            adatom_symbol=args.adatom_symbol,
            coverage_fraction=args.adatom_coverage,
            calculator=calc,
            config=config,
            results_dir=f"results_{args.surface_type}",
        )

    # compute (pure)
    skip = args.skip_existing and not args.force_rerun
    setup_directories([args.surface_type])

    if config.saturation:
        run_metadata = {}
        saturation_results = run_saturation_screening(
            slab,
            smiles_file=args.smiles_file,
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
        if run_metadata:
            write_run_metadata(
                surface_type=args.surface_type,
                config=config,
                smiles_file=args.smiles_file,
                n_molecules=run_metadata["n_molecules"],
                total_configs=run_metadata["total_configs"],
                t_ref_s=run_metadata["t_ref_s"],
                t_total_s=run_metadata["t_total_s"],
            )
        total_steps = sum(len(sr.steps) for sr in saturation_results)
        total_mols = sum(sr.n_molecules_at_saturation for sr in saturation_results)
        print(
            f"\nSaturation complete: {len(saturation_results)} molecules, "
            f"{total_steps} steps, {total_mols} molecules at saturation"
        )
    else:
        run_metadata = {}
        run_fn = run_screening_bayesian if config.bo_enabled else run_screening
        all_run_results = run_fn(
            slab,
            smiles_file=args.smiles_file,
            config=config,
            surface_type=args.surface_type,
            skip_existing=skip,
            run_metadata_out=run_metadata,
        )
        if run_metadata:
            write_run_metadata(
                surface_type=args.surface_type,
                config=config,
                smiles_file=args.smiles_file,
                n_molecules=run_metadata["n_molecules"],
                total_configs=run_metadata["total_configs"],
                t_ref_s=run_metadata["t_ref_s"],
                t_total_s=run_metadata["t_total_s"],
            )
        for rr in all_run_results:
            save_molecule_results(
                rr.molecule,
                rr.results,
                surface_type=args.surface_type,
                config=config,
            )
        save_summary_results(
            all_run_results,
            surface_type=args.surface_type,
            config=config,
        )

        total = sum(len(rr.results) for rr in all_run_results)
        print(f"\nScreening complete: {total} total configurations")


if __name__ == "__main__":
    main()
