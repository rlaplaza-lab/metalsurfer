#!/usr/bin/env python3
"""Compute binding energies of furanic molecules on semi-ordered graphene oxide (GO).

Molecules: HMF, BHMF, BHMTHF, 5-MF, MFA, DMF, MTHFA, DMTHF.

Slab: Semi-ordered GO model SO1 from Mouhat et al., Nature Commun. 2020 (citable-data).
Loaded from https://github.com/fxcoudert/citable-data. GO layer is fully frozen.

Uses BO pipeline: 100 placements in batches of 20 (20 initial random + 4 batches of 20).
Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"
"""

import argparse
import logging
import os
from io import StringIO
from urllib.request import urlopen

from ase.io import read

from metalsurfer import (
    AdsorptionConfig,
    calculate_reference_energies,
    create_slab_from_atoms,
    process_molecule_bayesian,
    save_single_molecule_results,
    save_summary_results,
    screening_run_result,
    setup_single_model,
    write_run_settings,
)
from metalsurfer.placement import classify_adsorbate_orientation

# (SMILES, molecule_name)
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
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if debug:
        logging.getLogger("metalsurfer.filters").setLevel(logging.DEBUG)
        logging.getLogger("metalsurfer.workflow").setLevel(logging.DEBUG)


CITABLE_BASE = (
    "https://raw.githubusercontent.com/fxcoudert/citable-data/master"
    "/122-Mouhat_NatureCommun_2020/models/GO"
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
    atoms.set_pbc([True, True, False])
    return create_slab_from_atoms(atoms)


def main():
    parser = argparse.ArgumentParser(
        description="Furanic molecules on semi-ordered GO (SO1) with BO (100 placements, batches of 20)"
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

    results_subdir = "furanics_go_so1"
    results_dir = f"results_{results_subdir}"
    os.makedirs(results_dir, exist_ok=True)

    slab = _load_go_slab("semi_ordered/SO1")
    logging.info("GO slab (semi-ordered SO1): %d atoms", len(slab.atoms))

    config = AdsorptionConfig(
        model_name="uma-s-1p1",
        seed=42,
        num_conformers=10,
        num_placements=100,
        autobatcher_max_memory_padding=0.5,
        device=device,
        skip_topology_check=False,
        skip_desorption_check=False,
        stage1_steps=50,
        stage2_steps=500,
        debug_write_initial_placements=False,
        relax_top_layer=False,  # Freeze entire GO layer
        bo_enabled=True,
        bo_initial_random=20,
        bo_batch_size=20,
        bo_total_budget=100,
    )

    calculator, ts_model = setup_single_model(config.model_name, config.device)

    smiles_list = [s for s, _ in MOLECULES]
    molecule_names = [n for _, n in MOLECULES]

    ref = calculate_reference_energies(
        slab,
        calculator,
        molecules=molecule_names,
        smiles_list=smiles_list,
        ts_model=ts_model,
        config=config,
    )

    e_slab = ref.slab_energy
    print(f"E_slab (GO SO1) = {e_slab:.4f} eV")
    for name in molecule_names:
        e_mol = ref.get_molecule_energy(name)
        if e_mol is not None:
            print(f"  E_{name} = {e_mol:.4f} eV")
        else:
            print(f"  E_{name} = (failed)")

    summary = []
    all_run_results = []

    for smiles, name in MOLECULES:
        print(f"\n--- Processing {name} ---")
        results = process_molecule_bayesian(
            smiles,
            name,
            slab,
            calculator,
            ref,
            ts_model=ts_model,
            config=config,
            surface_type=results_subdir,
        )

        if results:
            save_single_molecule_results(
                name,
                results,
                surface_type=results_subdir,
                system_name="GO_SO1",
                config=config,
                write_csv=False,
            )
            all_run_results.append(screening_run_result(name, results))
            best = min(results, key=lambda r: r.energy_adsorption)
            surface_symbols = set(slab.atoms.get_chemical_symbols())
            slab_size = next(
                (
                    i
                    for i, s in enumerate(results[0].atoms.get_chemical_symbols())
                    if s not in surface_symbols
                ),
                None,
            )
            orientations = []
            if slab_size is not None:
                orientations = [
                    classify_adsorbate_orientation(r.atoms, slab_size) for r in results
                ]
            n_parallel = sum(1 for o in orientations if o == "parallel")
            print(
                f"Binding energy of {name} on GO (SO1): {best.energy_adsorption:.4f} eV"
            )
            if orientations:
                print(
                    f"  Orientations: {n_parallel}/{len(results)} parallel, "
                    f"{len(results) - n_parallel} EN-down"
                )
            summary.append((name, best.energy_adsorption, len(results)))
        else:
            print(f"No valid placements found for {name}.")
            summary.append((name, None, 0))

    if all_run_results:
        save_summary_results(
            all_run_results,
            surface_type=results_subdir,
            config=config,
        )
        write_run_settings(
            results_subdir,
            config,
            campaign="multi_molecule_binding",
            n_molecules=len(all_run_results),
            molecules=[rr.molecule for rr in all_run_results],
            n_configurations=sum(len(rr.results) for rr in all_run_results),
        )

    print("\n" + "=" * 60)
    print("Binding energy summary (graphene oxide, semi-ordered SO1)")
    print("=" * 60)
    print("(E_ads = E(slab+molecule) - E(slab) - E(molecule); negative = favorable)")
    print()
    for name, e_ads, n_results in summary:
        if e_ads is not None:
            print(f"  {name:12s}: {e_ads:+.4f} eV  ({n_results} valid placements)")
        else:
            print(f"  {name:12s}: (no valid placements)")
    print()
    print(f"Results saved to {results_dir}/ (XYZ, POSCAR, CSV)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
