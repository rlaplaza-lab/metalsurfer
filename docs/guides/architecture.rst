Architecture
============

This page describes the library architecture, public API layers, and
computational data flow.  For the full technical reference see the
`CORE_SYSTEM_EXPLANATION.md <https://github.com/rlaplaza/metalsurfer/blob/main/CORE_SYSTEM_EXPLANATION.md>`_
file in the repository root.


Public API Layers
-----------------

The package re-exports a curated set of symbols through lazy imports in
``metalsurfer.__init__`` so heavy modules are loaded only when first
accessed.

**1. Run-Mode APIs** — canonical high-level entry points:

- :func:`~metalsurfer.run_adsorption` — standard multi-molecule screening; returns :class:`~metalsurfer.BindingCampaignResult`.
- :func:`~metalsurfer.run_adsorption_bo` — Bayesian optimization-guided screening; returns :class:`~metalsurfer.BindingCampaignResult`.
- :func:`~metalsurfer.run_saturation` — sequential saturation; returns :class:`~metalsurfer.SaturationCampaignResult` (per-molecule runs via ``.runs``).
- :func:`~metalsurfer.run_saturation_bo` — saturation with BO-guided placement selection; returns :class:`~metalsurfer.SaturationCampaignResult`.

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
``gradient_boost``, ``ridge``.  Per-sample transfer weights (``bo_transfer_*``)
apply only to tree surrogates; ``gradient_boost`` and ``ridge`` use
unweighted fits when those weights would otherwise be passed.

**Sequential Saturation** evolves the slab state step by step — run
screening, optionally filter step candidates with the topology rearrangement
guard (``saturation_discard_topology_rearrangements``, default ``True``),
select the best surviving result, update the slab, and repeat until
adsorption is no longer favorable (``E_ads ≥ 0``) or no valid placements
remain.  The guard is connectivity-only: it checks that the adsorbate pool
has the expected number of connected fragments, so coupled adsorbates or
unexpected splits are not carried forward while strong adsorbate-material
interactions that preserve connectivity remain allowed.  When
``multi_molecule_saturation=True``, all molecules compete at each step.

Saturation captures ``base_slab`` once after surface prep.  Placement
relaxation freezes that substrate block (``relax_top_layer=False`` freezes
every atom in the reference; default ``True`` freezes all but the top layer).
Prep equilibration uses ``slab_relaxation_mode`` separately — see
:doc:`surface_engineering`.  Step-1 ``auto_resize_slab`` may repeat the
substrate in-plane; standard, Bayesian, and saturation workflows expand the
freeze reference to the full repeated substrate (for both
``relax_top_layer=True`` and ``False``).  Competitive multi-molecule
saturation pre-resizes once before the step-1 molecule loop so every adsorbate
competes on the same footprint.


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

Result-Object Export and Formatting
-----------------------------------

Formatting and tabular export are now first-class methods on result objects,
so user scripts can stay on top-level imports without pulling helpers from
internal modules.

- :class:`~metalsurfer.ScreeningResult` provides ``to_row(...)``.
- :class:`~metalsurfer.ScreeningRunResult` provides ``to_rows(...)`` and
  ``to_dataframe(...)`` for detailed tables plus ``to_summary_row()``.
- :class:`~metalsurfer.SaturationStepResult` provides ``to_detail_row(...)``
  and ``to_rows(...)`` for step-level exports.
- :class:`~metalsurfer.SaturationRunResult` provides
  ``to_flattened_runs()`` and ``format_completion(...)``.
- :class:`~metalsurfer.SaturationCampaignResult` provides
  ``format_completion(...)`` and ``format_failure_summary()`` for campaign runs.
- :class:`~metalsurfer.BindingCampaignResult` provides
  ``format_summary(...)``, ``format_screening_complete()``, and
  ``format_results_saved_line(...)``.
