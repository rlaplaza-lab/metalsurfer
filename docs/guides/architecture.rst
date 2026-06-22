Architecture
============

This page describes the library architecture, public API layers, and
computational data flow.  For the full technical reference see the
`CORE_SYSTEM_EXPLANATION.md <https://github.com/rlaplaza-lab/metalsurfer/blob/main/CORE_SYSTEM_EXPLANATION.md>`_
file in the repository root (including TorchSim GPU batching, saturation BO
transfer learning, design heuristics, and comparisons with AdsorbML and BOSS).


Public API Layers
-----------------

The package re-exports a curated set of symbols through lazy imports in
``metalsurfer.__init__`` so heavy modules are loaded only when first
accessed.  The four layers below are the **public** surface; see
``CORE_SYSTEM_EXPLANATION.md`` for an additional internal workflow layer
(``_bootstrap_screening_run``, ``run_saturation_screening``, etc.) used by
campaigns but not re-exported from the top-level package.

**1. Run-Mode APIs** — canonical high-level entry points:

- :func:`~metalsurfer.run_adsorption` — standard multi-molecule screening; returns :class:`~metalsurfer.BindingCampaignResult`.
- :func:`~metalsurfer.run_adsorption_bo` — Bayesian optimization-guided screening; returns :class:`~metalsurfer.BindingCampaignResult`.
- :func:`~metalsurfer.run_saturation` — sequential saturation; returns :class:`~metalsurfer.SaturationCampaignResult` (per-molecule runs via ``.runs``).
- :func:`~metalsurfer.run_saturation_bo` — saturation with BO-guided placement selection; returns :class:`~metalsurfer.SaturationCampaignResult`.

**2. Surface Preparation** — :mod:`metalsurfer.surface_prep` builds a
campaign-ready substrate (optional bulk→slab, alloy, adatoms). By default
:func:`~metalsurfer.surface_prep.prepare_substrate` **equilibrates ionic
positions** (``slab_relaxation_mode="ionic_only"``) and attaches ASE
``FixAtoms`` via prep kwargs (default: freeze the entire substrate during
adsorption). For pre-built or DFT slabs pass
``align=False`` and/or ``slab_relaxation_mode="none"`` when the geometry must
not move. Size in-plane with
:func:`~metalsurfer.surface_prep.resize_substrate_for_molecule` **before**
calling campaign APIs. See :doc:`Substrate preparation <../api/surface_prep>`.

**3. Mid-Level Per-Molecule APIs** — useful for embedding metalsurfer
inside custom research loops:

- ``process_molecule(...)``
- ``process_molecule_bayesian(...)``
- ``calculate_reference_energies(...)``
- ``load_molecules(...)``

**4. Infrastructure APIs** — placement generation, optimization, filtering,
and result persistence helpers importable from the top-level ``metalsurfer``
namespace. Substrate construction and prep orchestration live in
:mod:`metalsurfer.surface_prep` (see :doc:`Substrate preparation <../api/surface_prep>`).


End-to-End Computational Flow
-----------------------------

Across all run modes, the physical pipeline follows seven stages:

1. **Surface Preparation** — build from a Materials Project bulk ID and
   Miller indices, or accept existing ASE Atoms.  Optional: alloy
   substitution, adatom deposition, supercell expansion.  Finalize with
   :func:`~metalsurfer.surface_prep.prepare_substrate` (PBC, ``FixAtoms``, validation) before
   calling campaign APIs.

2. **Reference Energy Construction** — compute the clean-slab energy and
   isolated-molecule energies.  Adsorption energy is defined as:

   .. math::

      E_\mathrm{ads} = E_\mathrm{adsorbate+slab} - E_\mathrm{slab} - E_\mathrm{molecule}

3. **Conformer Generation** — embed molecules from SMILES with RDKit,
   optimize, and deduplicate with RMSD and energy thresholds.

4. **Placement Specification** — enumerate deterministic
   ``PlacementSpec`` candidates over conformer, site,
   orientation, tilt, azimuth, and height.  Site detection is
   Voronoi-based and **orientation-aware** (slab normal, not Cartesian
   ``z``): hybrid topology + Voronoi for slabs, full-framework Voronoi for
   nanoparticles, periodic images for porous cells.  Homonuclear diatomics
   on slabs or nanoparticles can use a **dissociative** branch when
   ``skip_topology_check=True`` (two surface sites, site-specific outward
   normals).  See
   ``placement/sites.py`` and the `Site detection` section in
   `CORE_SYSTEM_EXPLANATION.md
   <https://github.com/rlaplaza-lab/metalsurfer/blob/main/CORE_SYSTEM_EXPLANATION.md>`_.

5. **Optimization** — relax candidate adsorbate-slab systems using the
   configured MLIP backend (TorchSim / FairChem).

6. **Validation and Filtering** — geometry validation, decomposition
   detection, desorption check, duplicate removal, energy-cap filtering.

7. **Aggregation and Persistence** — rank surviving structures by
   adsorption energy and write CSV summaries, XYZ files, and metadata JSON.
   VASP-format files (POSCAR/INCAR/KPOINTS and reference-slab POSCARs) are
   opt-in via ``write_vasp_inputs=True`` on :class:`~metalsurfer.AdsorptionConfig`.


Placement materialization and ML injectivity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each ``PlacementSpec`` is materialized deterministically by
``generate_placement_from_spec`` into absolute coordinates and a unit
quaternion (``PlacementDescriptor`` / ``PlacementPose``).  Bayesian
screening builds surrogate inputs with ``build_spec_features_geometry_aware``:
specs are resolved to poses first; ``extract_features`` then reads **only**
``x_abs``, ``y_abs``, ``z_abs``, ``conformer_index``, and the quaternion.
``site_index``, ``site_type``, orientation labels, and ``z_fraction`` are
logged in CSV/dataset rows but are **not** model features—so the BO loop
learns a geometry-only map and replay via spec, descriptor, or pose stays
consistent.  Details: `ML/BO injectivity` in
`CORE_SYSTEM_EXPLANATION.md
<https://github.com/rlaplaza-lab/metalsurfer/blob/main/CORE_SYSTEM_EXPLANATION.md>`_.


Run Modes
---------

**Standard Screening** enumerates a deterministic placement pool for each
molecule, relaxes every sampled candidate, and returns the surviving
low-energy configurations.

**Bayesian Screening** replaces exhaustive placement evaluation with a
surrogate-guided loop:

1. Enumerate the candidate pool.
2. Evaluate an initial random subset.
3. Build **geometry-aware** features (materialize each spec, extract absolute
   pose coordinates and quaternion—not raw ``site_index`` or orientation
   labels).
4. Fit a surrogate model.
5. Score unevaluated candidates with an acquisition function.
6. Evaluate the next batch and repeat until the budget is exhausted.

Supported acquisition functions: ``lcb``, ``ei``, ``pi``.
Supported surrogates: ``random_forest``, ``extra_trees``,
``gradient_boost``, ``ridge``, ``ensemble``.  Per-sample transfer weights
(``bo_transfer_*``) apply to tree surrogates, ``ridge``, and ``ensemble``;
``gradient_boost`` rejects sample weights and cannot be used with
``bo_transfer_enabled``.

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
relaxation honors ASE ``FixAtoms`` on that reference (set during prep via
:func:`~metalsurfer.surface_prep.apply_surface_constraints`).  Prep equilibration uses ``slab_relaxation_mode``
separately — see :doc:`surface_engineering`.  In-plane supercell sizing must
be completed during prep before calling campaign APIs.


Module Layout
-------------

::

   metalsurfer/
   ├── campaigns.py          # run_adsorption, run_adsorption_bo, ...
   ├── config.py             # AdsorptionConfig + validation
   ├── conformers.py         # SMILES → conformers
   ├── filters.py            # decomposition / desorption / duplicate filtering
   ├── io_results.py         # CSV, XYZ, optional VASP I/O, metadata persistence
   ├── models.py             # typed result dataclasses
   ├── optimization.py       # MLIP setup, batched relaxation
   ├── surface_prep/         # canonical substrate prep API (prepare_substrate, …)
   ├── surfaces.py           # slab construction, alloy, adatom (implementation)
   ├── symmetry.py           # spglib-based symmetry analysis
   │
   ├── ml/                   # BO surrogates, dataset, features
   ├── placement/            # orientation-aware Voronoi/topology sites, geometry, generators
   └── workflow/             # orchestration by run mode
       ├── core.py           # standard per-molecule screening
       ├── bayesian.py       # BO-guided per-molecule screening
       ├── saturation.py     # sequential / multi-mol saturation
       ├── reference.py      # reference energy preparation
       └── shared.py         # bootstrap, molecule preamble, validation, autotune


Output Structure
----------------

The default output root is ``results_{surface_type}/``.  Common artifacts:

- ``adsorption_energies_detailed.csv`` — per-placement results with
  descriptor fields.
- ``adsorption_energy_summary.csv`` — aggregate statistics per molecule.
- ``saturation_details.csv`` / ``saturation_summary.csv`` — saturation
  step results.
- ``saturation_placements_detailed.csv`` and ``step_{NNN}_placements/`` when
  ``saturation_save_all_placements=True`` (default).
- ``run_metadata.json`` — config snapshot plus optional timing/count metadata (``write_settings`` and/or ``write_metadata`` on campaign APIs; merged incrementally into one file).
- ``ml_dataset.csv``, ``ml_dataset_metadata.json`` — ML placement records from
  :class:`~metalsurfer.DatasetLogger` during binding campaigns and saturation.
- ``xyz_structures/`` — optimized structures in XYZ format.
- ``vasp_inputs/`` — optional POSCAR/INCAR/KPOINTS bundles for DFT follow-up
  (written only when ``write_vasp_inputs=True``).

Campaign ``save_results`` controls CSV/XYZ persistence; VASP bundles require
``config.write_vasp_inputs=True`` in addition.

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
  Pass ``write_vasp_inputs=config.write_vasp_inputs`` to ``format_completion``
  so the saved-files suffix matches actual output (``(XYZ, CSV)`` vs
  ``(XYZ, POSCAR, CSV)``).
- :class:`~metalsurfer.BindingCampaignResult` provides
  ``format_summary(...)``, ``format_screening_complete()``, and
  ``format_results_saved_line(...)``.
