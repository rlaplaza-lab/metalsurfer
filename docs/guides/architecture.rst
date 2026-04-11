Architecture
============

This page describes the library architecture, public API layers, and
computational data flow.  For the full technical reference see the
`CORE_SYSTEM_EXPLANATION.md <https://github.com/rlaplaza-lab/metalsurfer/blob/main/CORE_SYSTEM_EXPLANATION.md>`_
file in the repository root.


Public API Layers
-----------------

The package re-exports a curated set of symbols through lazy imports in
``metalsurfer.__init__`` so heavy modules are loaded only when first
accessed.

**1. Run-Mode APIs** — canonical high-level entry points:

- :func:`~metalsurfer.run_adsorption` — standard multi-molecule screening.
- :func:`~metalsurfer.run_adsorption_bo` — Bayesian optimization-guided screening.
- :func:`~metalsurfer.run_saturation` — sequential saturation.
- :func:`~metalsurfer.run_saturation_bo` — saturation with BO-guided placement selection.

**2. Surface Preparation** — :func:`~metalsurfer.prepare_slab` provides a
single call for bulk→slab construction, alloy substitution, and adatom
deposition.

**3. Mid-Level Per-Molecule APIs** — useful for embedding metalsurfer
inside custom research loops:

- ``process_molecule(...)``
- ``process_molecule_bayesian(...)``
- ``calculate_reference_energies(...)``
- ``load_molecules(...)``

**4. Infrastructure APIs** — surface construction, placement generation,
optimization, filtering, and result persistence helpers, all importable
from the top-level ``metalsurfer`` namespace.


End-to-End Computational Flow
-----------------------------

Across all run modes, the physical pipeline follows seven stages:

1. **Surface Preparation** — build from a Materials Project bulk ID and
   Miller indices, or accept existing ASE Atoms.  Optional: alloy
   substitution, adatom deposition, supercell expansion.

2. **Reference Energy Construction** — compute the clean-slab energy and
   isolated-molecule energies.  Adsorption energy is defined as:

   .. math::

      E_\mathrm{ads} = E_\mathrm{adsorbate+slab} - E_\mathrm{slab} - E_\mathrm{molecule}

3. **Conformer Generation** — embed molecules from SMILES with RDKit,
   optimize, and deduplicate with RMSD and energy thresholds.

4. **Placement Specification** — enumerate deterministic
   :class:`~metalsurfer.PlacementSpec` candidates over conformer, site,
   orientation, tilt, azimuth, and height.

5. **Optimization** — relax candidate adsorbate-slab systems using the
   configured MLIP backend (TorchSim / FairChem).

6. **Validation and Filtering** — geometry validation, decomposition
   detection, desorption check, duplicate removal, energy-cap filtering.

7. **Aggregation and Persistence** — rank surviving structures by
   adsorption energy and write CSV summaries, XYZ files, POSCAR files,
   and metadata JSON.


Run Modes
---------

**Standard Screening** enumerates a deterministic placement pool for each
molecule, relaxes every sampled candidate, and returns the surviving
low-energy configurations.

**Bayesian Screening** replaces exhaustive placement evaluation with a
surrogate-guided loop:

1. Enumerate the candidate pool.
2. Evaluate an initial random subset.
3. Build features from placement descriptors.
4. Fit a surrogate model.
5. Score unevaluated candidates with an acquisition function.
6. Evaluate the next batch and repeat until the budget is exhausted.

Supported acquisition functions: ``lcb``, ``ei``, ``pi``.
Supported surrogates: ``random_forest``, ``extra_trees``,
``gradient_boost``, ``ridge``.

**Sequential Saturation** evolves the slab state step by step — run
screening, select the best result, update the slab, and repeat until
adsorption is no longer favorable (``E_ads ≥ 0``) or no valid placements
remain.  When ``multi_molecule_saturation=True``, all molecules compete at
each step.


Module Layout
-------------

::

   metalsurfer/
   ├── campaigns.py          # run_adsorption, run_adsorption_bo, ...
   ├── config.py             # AdsorptionConfig + validation
   ├── conformers.py         # SMILES → conformers
   ├── filters.py            # decomposition / desorption / duplicate filtering
   ├── io_results.py         # CSV, XYZ, POSCAR, metadata persistence
   ├── models.py             # typed result dataclasses
   ├── optimization.py       # MLIP setup, batched relaxation
   ├── surface_prep.py       # prepare_slab convenience wrapper
   ├── surfaces.py           # slab construction, alloy, adatom
   ├── symmetry.py           # spglib-based symmetry analysis
   │
   ├── cli/                  # CLI entry point (metalsurfer command)
   ├── ml/                   # BO surrogates, dataset, features
   ├── placement/            # Voronoi sites, orientation, placement generation
   └── workflow/             # orchestration by run mode
       ├── core.py           # standard per-molecule screening
       ├── bayesian.py       # BO-guided per-molecule screening
       ├── screening.py      # file-driven multi-molecule loops
       ├── saturation.py     # sequential / multi-mol saturation
       ├── reference.py      # reference energy preparation
       └── shared.py         # validation helpers, common setup


Output Structure
----------------

The default output root is ``results_{surface_type}/``.  Common artifacts:

- ``adsorption_energies_detailed.csv`` — per-placement results with
  descriptor fields.
- ``adsorption_energy_summary.csv`` — aggregate statistics per molecule.
- ``saturation_details.csv`` / ``saturation_summary.csv`` — saturation
  step results.
- ``run_metadata.json`` — timing, counts, and config snapshot.
- ``xyz_structures/`` — optimized structures in XYZ format.
- ``vasp_inputs/`` — POSCAR files for DFT follow-up.
