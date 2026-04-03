#!/usr/bin/env python3
"""CLI entry point for adsorption screening.

Usage examples::

    metalsurfer run --smiles-file molecules.csv --bulk-id mp-33 --miller 1 1 1
    metalsurfer run --molecule '[H][H]' H2 --slab-file slab.xyz --material-type slab
    metalsurfer run-bo --smiles-file molecules.csv --bulk-id mp-33
    metalsurfer saturate --smiles-file molecules.csv --bulk-id mp-33
    metalsurfer saturate --bo-enabled --smiles-file molecules.csv --bulk-id mp-33
"""

from __future__ import annotations

import argparse

from metalsurfer import AdsorptionConfig, save_saturation_results, write_run_metadata
from metalsurfer._logging import configure_logging
from metalsurfer.campaigns import (
    run_adsorption,
    run_adsorption_bo,
    run_saturation,
    run_saturation_bo,
)
from metalsurfer.cli.cli_output import (
    format_binding_summary,
    format_saturation_complete,
    format_screening_complete,
)
from metalsurfer.surface_prep import prepare_slab


def _build_common_parser() -> argparse.ArgumentParser:
    """Return the top-level parser with global arguments."""
    parser = argparse.ArgumentParser(
        prog="metalsurfer",
        description=(
            "Adsorption screening on arbitrary surfaces.\n\n"
            "Molecules can be supplied via --smiles-file (CSV) or inline with\n"
            "one or more --molecule SMILES NAME pairs.\n\n"
            "The surface can be constructed from the Materials Project with\n"
            "--bulk-id, or loaded from a file with --slab-file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Logging
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    log_group.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress all but warning-level and higher messages",
    )

    # ── Surface construction ──────────────────────────────────────────────────
    surface_group = parser.add_argument_group("surface construction")
    slab_src = surface_group.add_mutually_exclusive_group()
    slab_src.add_argument(
        "--bulk-id",
        type=str,
        default="mp-33",
        metavar="ID",
        help="Materials Project structure ID (default: %(default)s)",
    )
    slab_src.add_argument(
        "--slab-file",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a pre-built slab (POSCAR or XYZ). Mutually exclusive with --bulk-id.",
    )
    surface_group.add_argument(
        "--miller",
        type=int,
        nargs=3,
        default=[0, 0, 1],
        metavar=("H", "K", "L"),
        help="Miller indices (default: 0 0 1). Ignored when --slab-file is used.",
    )
    surface_group.add_argument(
        "--supercell",
        type=int,
        nargs=3,
        default=[2, 2, 1],
        metavar=("NX", "NY", "NZ"),
        help="Supercell repeat (default: 2 2 1). Ignored when --slab-file is used.",
    )
    surface_group.add_argument(
        "--material-type",
        choices=["slab", "nanoparticle", "porous"],
        default="slab",
        help="Surface material type (default: %(default)s)",
    )

    # ── Surface modifiers ─────────────────────────────────────────────────────
    mod_group = parser.add_argument_group("surface modifiers")
    mod_group.add_argument(
        "--alloy-host",
        type=str,
        default=None,
        metavar="ELEMENT",
        help="Host element for alloy substitution (default: majority element)",
    )
    mod_group.add_argument(
        "--alloy-guest",
        type=str,
        default=None,
        metavar="ELEMENT",
        help="Guest element to substitute into the surface",
    )
    mod_group.add_argument(
        "--alloy-fraction",
        type=float,
        default=0.0,
        metavar="FRAC",
        help="Fraction of host sites to replace with the guest (0–1, default: 0)",
    )
    mod_group.add_argument(
        "--adatom-symbol",
        type=str,
        default=None,
        metavar="ELEMENT",
        help="Element symbol of adatoms to deposit onto the surface",
    )
    mod_group.add_argument(
        "--adatom-coverage",
        type=float,
        default=0.0,
        metavar="FRAC",
        help="Surface coverage fraction for adatom deposition (0–1, default: 0)",
    )

    # ── Primary screening knobs ───────────────────────────────────────────────
    run_group = parser.add_argument_group("screening parameters")
    run_group.add_argument(
        "--model",
        type=str,
        default="uma-s-1p1",
        metavar="NAME",
        help="MLIP model name (default: %(default)s)",
    )
    run_group.add_argument(
        "--num-conformers",
        type=int,
        default=10,
        metavar="N",
        help="RDKit conformers to generate per molecule (default: %(default)s)",
    )
    run_group.add_argument(
        "--num-placements",
        type=int,
        default=100,
        metavar="N",
        help="Placement candidates to evaluate per molecule (default: %(default)s)",
    )
    run_group.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device (default: %(default)s)",
    )
    run_group.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="INT",
        help="Global random seed (default: %(default)s)",
    )

    # ── Molecule input ────────────────────────────────────────────────────────
    mol_group = parser.add_argument_group("molecule input")
    mol_src = mol_group.add_mutually_exclusive_group()
    mol_src.add_argument(
        "--smiles-file",
        type=str,
        default="smiles.csv",
        metavar="PATH",
        help="CSV file with 'smiles' and 'name' columns (default: %(default)s)",
    )
    mol_src.add_argument(
        "--molecule",
        dest="molecules",
        action="append",
        nargs=2,
        metavar=("SMILES", "NAME"),
        help=(
            "Inline molecule specification.  May be repeated.\n"
            "Example: --molecule 'CCO' ethanol --molecule '[H][H]' H2"
        ),
    )

    # ── I/O ──────────────────────────────────────────────────────────────────
    io_group = parser.add_argument_group("output")
    io_group.add_argument(
        "--surface-type",
        type=str,
        default="manual",
        metavar="LABEL",
        help="Label for the results directory (results_<LABEL>/, default: %(default)s)",
    )
    io_group.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Override the output directory.  When set, --surface-type is ignored "
            "for directory naming and this path is used directly."
        ),
    )
    io_group.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip molecules already present in the summary CSV (default: on)",
    )
    io_group.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore --skip-existing and recompute all molecules",
    )

    return parser


def _add_bo_args(subparser: argparse.ArgumentParser) -> None:
    """Add Bayesian optimisation arguments to a subparser."""
    bo = subparser.add_argument_group("Bayesian optimisation")
    bo.add_argument(
        "--bo-initial-random",
        type=int,
        default=20,
        metavar="N",
        help="Random evaluations before BO kicks in (default: %(default)s)",
    )
    bo.add_argument(
        "--bo-batch-size",
        type=int,
        default=10,
        metavar="N",
        help="Candidates per BO acquisition batch (default: %(default)s)",
    )
    bo.add_argument(
        "--bo-total-budget",
        type=int,
        default=100,
        metavar="N",
        help="Total BO evaluation budget (default: %(default)s)",
    )
    bo.add_argument(
        "--bo-acquisition",
        choices=["lcb", "ei", "pi"],
        default="lcb",
        help="Acquisition function (default: %(default)s)",
    )
    bo.add_argument(
        "--bo-ucb-kappa",
        type=float,
        default=1.96,
        metavar="κ",
        help="UCB exploration parameter κ (default: %(default)s)",
    )
    bo.add_argument(
        "--bo-surrogate",
        choices=["random_forest", "extra_trees", "gradient_boost", "ridge"],
        default="random_forest",
        help="Surrogate model type (default: %(default)s)",
    )


def _resolve_molecules_arg(args: argparse.Namespace) -> list[tuple[str, str]] | str:
    """Return molecules as in-memory list or CSV path depending on CLI input."""
    if args.molecules:
        return [(smiles, name) for smiles, name in args.molecules]
    return args.smiles_file


def _resolve_surface_type(args: argparse.Namespace) -> str:
    """Return the effective surface_type label."""
    if args.output_dir:
        # When --output-dir is set, derive label from the last path component so
        # the results files still get sensible names.
        import os

        return os.path.basename(args.output_dir.rstrip("/\\")) or "manual"
    return args.surface_type


def _resolve_output_dir(args: argparse.Namespace, surface_type: str) -> str:
    return args.output_dir if args.output_dir else f"results_{surface_type}"


def _build_config(
    args: argparse.Namespace, *, saturation: bool = False, bo: bool = False
) -> AdsorptionConfig:
    return AdsorptionConfig(
        model_name=args.model,
        num_conformers=args.num_conformers,
        num_placements=args.num_placements,
        device=args.device,
        seed=args.seed,
        material_type=args.material_type,
        saturation=saturation,
        bo_enabled=bo,
        bo_initial_random=getattr(args, "bo_initial_random", 20),
        bo_batch_size=getattr(args, "bo_batch_size", 10),
        bo_total_budget=getattr(args, "bo_total_budget", 100),
        bo_acquisition=getattr(args, "bo_acquisition", "lcb"),
        bo_ucb_kappa=getattr(args, "bo_ucb_kappa", 1.96),
        bo_surrogate=getattr(args, "bo_surrogate", "random_forest"),
    )


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


def main() -> None:  # noqa: C901 (acceptable complexity for a CLI dispatcher)
    parser = _build_common_parser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "run",
        help="Standard (non-Bayesian) adsorption screening",
        description="Exhaustively evaluate placement candidates using deterministic sampling.",
    )

    run_bo_parser = subparsers.add_parser(
        "run-bo",
        help="Bayesian-optimisation-guided adsorption screening",
        description="Sample-efficient screening using a surrogate model to select candidates.",
    )
    _add_bo_args(run_bo_parser)

    saturate_parser = subparsers.add_parser(
        "saturate",
        help="Sequential saturation screening",
        description=(
            "Iteratively adsorb molecules until the best adsorption energy is "
            "no longer favourable (E_ads ≥ 0).  Pass --bo-enabled to use "
            "Bayesian optimisation at each saturation step."
        ),
    )
    saturate_parser.add_argument(
        "--bo-enabled",
        action="store_true",
        help="Use Bayesian optimisation at each saturation step",
    )
    _add_bo_args(saturate_parser)

    args = parser.parse_args()

    # ── Logging ───────────────────────────────────────────────────────────────
    if args.verbose:
        configure_logging(default_level="DEBUG")
    elif args.quiet:
        configure_logging(default_level="WARNING")
    else:
        configure_logging(default_level="INFO")

    surface_type = _resolve_surface_type(args)
    results_dir = _resolve_output_dir(args, surface_type)
    skip = args.skip_existing and not args.force_rerun
    molecules = _resolve_molecules_arg(args)
    smiles_file = molecules if isinstance(molecules, str) else "<inline-molecules>"

    # ── Build surface ─────────────────────────────────────────────────────────
    slab = prepare_slab(
        bulk_id=args.bulk_id if not args.slab_file else None,
        miller_indices=tuple(args.miller),
        supercell=tuple(args.supercell),
        slab_file=args.slab_file,
        results_dir=results_dir,
        alloy_host=args.alloy_host,
        alloy_guest=args.alloy_guest,
        alloy_fraction=args.alloy_fraction,
        adatom_symbol=args.adatom_symbol,
        adatom_coverage=args.adatom_coverage,
        model_name=args.model,
        device=args.device,
    )

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if args.command == "saturate":
        bo_enabled = getattr(args, "bo_enabled", False)
        config = _build_config(args, saturation=True, bo=bo_enabled)
        run_metadata: dict[str, float] = {}
        run_fn = run_saturation_bo if bo_enabled else run_saturation
        saturation_results = run_fn(
            slab=slab,
            molecules=molecules,
            config=config,
            surface_type=surface_type,
            skip_existing=skip,
            run_metadata_out=run_metadata,
        )
        save_saturation_results(
            saturation_results,
            surface_type=surface_type,
            config=config,
        )
        _write_metadata_if_available(
            run_metadata=run_metadata,
            surface_type=surface_type,
            config=config,
            smiles_file=smiles_file,
        )
        total_steps = sum(len(sr.steps) for sr in saturation_results)
        total_mols = sum(sr.n_molecules_at_saturation for sr in saturation_results)
        print("")
        print(
            format_saturation_complete(
                label=f"Saturation ({len(saturation_results)} molecules)",
                n_molecules_at_saturation=total_mols,
                total_steps=total_steps,
                results_dir=results_dir,
            )
        )

    else:
        bo = args.command == "run-bo"
        config = _build_config(args, saturation=False, bo=bo)
        run_metadata = {}
        run_fn = run_adsorption_bo if bo else run_adsorption
        campaign = run_fn(
            slab=slab,
            molecules=molecules,
            config=config,
            surface_type=surface_type,
            skip_existing=skip,
            run_metadata_out=run_metadata,
            write_metadata=True,
        )
        _write_metadata_if_available(
            run_metadata=run_metadata,
            surface_type=surface_type,
            config=config,
            smiles_file=smiles_file,
        )
        print("")
        if campaign.molecule_summaries:
            print(
                format_binding_summary(
                    title=f"Results — {surface_type}",
                    molecule_summaries=campaign.molecule_summaries,
                    results_dir=results_dir,
                )
            )
        else:
            print(format_screening_complete(campaign.total_configurations))


if __name__ == "__main__":
    main()
