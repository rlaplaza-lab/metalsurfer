# Metalsurfer Core-System Explanation

## Purpose And Scope

`metalsurfer` is a Python library for adsorption-energy screening on arbitrary surfaces. It combines structure preparation, conformer generation, deterministic placement enumeration, MLIP-backed relaxation, and chemistry-aware post-filtering into one reproducible workflow stack.

The core system supports three primary run modes:

- standard adsorption screening,
- Bayesian optimization-guided screening,
- sequential saturation screening, including optional multi-molecule competition.

This document describes the library architecture, the public API layers, the computational data flow, and the output model. It excludes ad hoc scripts under `scripts/` and benchmark harnesses under `benchmarks/`.

## Public API Layers

The package boundary in `metalsurfer.__init__` re-exports a curated set of symbols through lazy imports so heavy modules are loaded only when first accessed.

The public surface is organized into four practical layers:

### 1. Run-Mode APIs

These are the canonical high-level entry points, all importable directly from `metalsurfer`:

- `run_adsorption(...)` — standard multi-molecule screening from an in-memory list of `(smiles, name)` tuples **or** a CSV path.
- `run_adsorption_bo(...)` — Bayesian optimization-guided screening over the same interface; forces `bo_enabled=True`.
- `run_saturation(...)` — sequential saturation; `molecules` accepts a list of tuples or a CSV path.
- `run_saturation_bo(...)` — saturation with BO-guided placement selection.

All four accept:

- a prepared `SlabContainer` (or plain ASE `Atoms`),
- `molecules`: either `list[tuple[str, str]]` of `(smiles, name)` pairs or a `str` path to a SMILES CSV,
- an `AdsorptionConfig`,
- a `surface_type` label.

`run_adsorption` and `run_adsorption_bo` return a typed `BindingCampaignResult`.
`run_saturation` and `run_saturation_bo` return `list[SaturationRunResult]` or `list[MultiMolSaturationRunResult]`.

### 2. Surface Preparation API

`prepare_slab(...)` in `metalsurfer.surface_prep` provides a single call for bulk→slab construction, alloy substitution, and adatom deposition in one step.

### 3. Workflow APIs (internal)

These are the file-driven orchestration helpers used internally by the run-mode APIs:

- `_run_screening_common(...)`
- `run_saturation_screening(...)`

They load molecules from a SMILES CSV, compute references, execute the requested workflow, and return typed result collections.

### 4. Mid-Level Per-Molecule APIs

These power the workflow entry points and are useful when embedding Metalsurfer inside custom research loops:

- `process_molecule(...)`
- `process_molecule_bayesian(...)`
- `calculate_reference_energies(...)`
- `load_molecules(...)`

These functions expose the core screening mechanics without imposing a full batch run.

### 4. Infrastructure APIs

Supporting public helpers include:

- surface construction and editing: `create_slab_from_bulk`, `create_slab_from_atoms`, `substitute_alloy`, `deposit_adatoms`, `auto_resize_slab_for_molecule`
- placement generation: `enumerate_placement_specs`, `generate_placement_from_spec`, `generate_placement_from_descriptor`
- optimization and calculators: `setup_calculator`, `setup_single_model`, `setup_torchsim_model`, `optimize_isolated_molecules_batched`, `optimize_adsorbate_slab_batched`
- result persistence: `save_single_molecule_results`, `save_summary_results`, `save_saturation_results`, `save_multi_mol_saturation_results`, `write_run_metadata`, `write_run_settings`

## Module Architecture

The package is intentionally split into modules with narrow responsibilities.

- `config.py`: `AdsorptionConfig` plus validation of physical and workflow hyperparameters.
- `models.py`: typed data contracts for references, placements, screening, saturation, timing, and campaign summaries.
- `surfaces.py`: slab construction, alloy substitution, adatom deposition, supercell resizing, and slab wrappers.
- `conformers.py`: SMILES-to-conformer generation, isolated-molecule optimization, and conformer selection.
- `placement/`: site discovery, orientation logic, deterministic placement specifications, and placement materialization.
- `symmetry.py`: symmetry analysis and site equivalence logic built on `spglib`.
- `optimization.py`: model setup, constrained relaxation, top-layer identification, frozen-atom policies, and autobatching.
- `filters.py`: decomposition, desorption, and duplicate filtering.
- `workflow/`: orchestration split by run mode and shared helpers.
- `campaigns.py`: in-memory multi-molecule campaign wrappers.
- `io_results.py`: CSV, XYZ, VASP, and metadata persistence.
- `ml/`: dataset logging, schema/context rows, feature extraction, surrogate training, evaluation, and acquisition utilities.

The workflow subpackage is further divided into:

- `workflow/core.py`: standard per-molecule placement screening.
- `workflow/bayesian.py`: BO-guided per-molecule screening.
- `workflow/screening.py`: file-driven multi-molecule screening loops.
- `workflow/saturation.py`: sequential and multi-molecule saturation loops.
- `workflow/reference.py`: reference-energy preparation.
- `workflow/shared.py`: validation helpers, failure formatting, and shared screening setup logic.

## Run Modes

### Standard Screening

Standard screening enumerates a deterministic placement pool for each molecule, relaxes every sampled candidate, filters the optimized structures, and returns the surviving low-energy adsorption configurations.

This is the baseline mode when broad coverage is preferred over aggressive sample efficiency.

### Bayesian Screening

Bayesian screening keeps the same conformer generation, placement descriptors, optimization backend, and filters, but changes how candidates are chosen for evaluation.

The BO loop is:

1. Enumerate the candidate pool.
2. Evaluate an initial random subset.
3. Build features from placement descriptors.
4. Fit a surrogate model.
5. Score unevaluated candidates with an acquisition function.
6. Evaluate the next batch and repeat until the budget is exhausted.

Supported acquisition modes are `lcb`, `ei`, and `pi`. Supported surrogate families are `random_forest`, `extra_trees`, `gradient_boost`, and `ridge`. For `gradient_boost` and `ridge`, optional `sample_weight` arguments are ignored (tree ensembles use weights when provided; transfer-learning weights in the workflow apply only where the fitted estimator supports them).

Failed placements can optionally be fed back as penalized negatives through `bo_include_failure_negatives` and `bo_failure_penalty_*`, which helps the surrogate learn which regions of placement space are unproductive.

### Sequential Saturation

Saturation screening evolves the slab state step by step:

1. Run screening on the current slab.
2. Select the best valid adsorption result.
3. Treat that optimized adsorbate-slab as the next slab state.
4. Repeat until the best adsorption is non-favorable (`E_ads >= 0`) or no valid placements remain.

Important saturation-specific behavior:

- the slab auto-resize path is intentionally limited to the first step,
- symmetry-breaking checks can switch later steps from symmetry-reduced site sampling to more comprehensive enumeration,
- BO transfer memory can carry observations from earlier saturation steps into later ones,
- when `multi_molecule_saturation=True` and the input file contains multiple molecules, the code runs a competitive saturation loop and chooses the best overall molecule at each step.

## End-To-End Computational Flow

Across run modes, the physical pipeline has the same major stages.

### 1. Surface Preparation

Users may begin with a Materials Project bulk ID and Miller indices, or wrap existing ASE atoms.

Optional modifications include:

- alloy substitution via `substitute_alloy(...)`,
- adatom deposition via `deposit_adatoms(...)`,
- in-plane supercell expansion via `auto_resize_slab_for_molecule(...)` when periodic-image separation would otherwise be too small.

### 2. Reference Energy Construction

The reference stage computes:

- the clean slab energy,
- isolated-molecule energies for every molecule in the run.

Adsorption energy is then defined as:

`E_ads = E_adsorbate+slab - E_slab - E_molecule`

If a reference energy is missing, the workflow can either skip that molecule or fail hard, depending on `fail_on_missing_reference`.

### 3. Conformer Generation

Molecules are embedded from SMILES with RDKit, optimized, optionally rescored with the MLIP backend, and deduplicated with RMSD and energy thresholds.

The conformer stage is controlled by fields such as:

- `num_conformers`,
- `conformer_sampling`,
- `boltzmann_temperature`,
- `energy_dedup_threshold`,
- `rmsd_dedup_threshold`.

### 4. Placement Specification And Materialization

Placement is driven by deterministic `PlacementSpec` objects. A spec records the abstract choice of conformer, site, orientation, tilt, azimuth, and height, while `PlacementDescriptor` records the resolved geometry used for replay and analysis.

The main placement path is:

`enumerate_placement_specs(...) -> generate_placement_from_spec(...) -> PlacementDescriptor`

Benefits of this design:

- reproducibility from seeds and descriptors,
- easy logging for downstream ML,
- clean separation between candidate enumeration and expensive optimization.

### 5. Optimization

Candidate adsorbate-slab systems are relaxed with the configured backend, typically TorchSim/FairChem through `setup_single_model(...)` and the batched optimization helpers.

The optimizer layer includes:

- top-layer detection,
- frozen-atom policies,
- configurable optimizers (`fire`, `lbfgs`, `bfgs`),
- GPU-memory-aware autobatching and cache reuse for saturation.

### 6. Validation And Filtering

After relaxation, the system applies several postchecks:

- geometry validation,
- adsorption-distance validation,
- decomposition detection,
- duplicate removal,
- adsorption-energy cap filtering.

`skip_topology_check` and `skip_desorption_check` allow expert users to loosen these checks for special cases such as dissociative adsorption workflows.

### 7. Aggregation And Persistence

Retained results are ranked by adsorption energy and converted into typed outputs such as:

- `ScreeningResult`
- `ScreeningRunResult`
- `SaturationStepResult`
- `SaturationRunResult`
- `MultiMolSaturationStepResult`
- `MultiMolSaturationRunResult`
- `BindingCampaignResult`

Persistence is handled separately from compute logic through `io_results.py`.

## Material Types And Site Generation

`AdsorptionConfig.material_type` is an explicit public contract with three valid values:

- `"slab"`
- `"nanoparticle"`
- `"porous"`

This choice affects:

- periodic boundary handling,
- adsorption-site discovery,
- local surface-normal construction,
- adsorption validation distances.

### Voronoi-Based Sites

Adsorption sites are generated from Voronoi geometry over the framework atoms. The same general machinery is used across supported material types.

Important placement hyperparameters include:

- `voronoi_probe_radius`: minimum distance from framework atoms to a candidate site,
- `voronoi_max_site_distance`: upper distance bound for chemically relevant sites,
- `top_layer_tolerance`: top-layer identification for slabs,
- `symmetry_tolerance` and `site_equivalence_tolerance`: symmetry and deduplication tolerances.

Site classes include atop, bridge, hollow, and pore-like positions depending on local geometry.

### Symmetry Handling

`symmetry.py` uses `spglib` to group equivalent sites and detect symmetry breaking. Failed symmetry dataset construction raises `SymmetryAnalysisError` (including wrapped `spglib` failures) instead of returning placeholder symmetry data.

`get_symmetry_aware_sites()` returns an empty list when Voronoi yields no sites; otherwise it propagates `SymmetryAnalysisError` if spglib or orbit verification fails (no warning-and-`None` at the placement layer). `workflow/shared._resolve_site_context_for_sampling` catches that exception once, logs a single INFO line, and falls back to the core unified (cluster-deduplicated) Voronoi sites—appropriate when symmetry is broken or spglib cannot treat the structure.

## Configuration Model

`AdsorptionConfig` centralizes both physical and workflow configuration.

Representative groups of fields are:

### Sampling And Placement

- `num_conformers`, `num_placements`
- `placement_x_range`, `placement_y_range`, `placement_z_range`
- `placement_z_scale_by_covalent_radius`
- `min_initial_distance`, `max_initial_distance`, `min_contact_ratio`
- `flat_aromatic_parallel_fraction`

### Relaxation And Validation

- `model_name`, `device`
- `fmax`, `stage1_steps`, `stage2_steps`, `reference_optimization_steps`
- `relax_top_layer`, `freeze_symbols`
- `min_interatomic_distance`, `max_force_convergence`
- `binding_distance_threshold`, `max_adsorption_energy`

### Reproducibility And Strictness

- `seed`
- `fail_on_missing_reference`
- `fail_on_conformer_failure`
- `debug_write_initial_placements`

### Surface Sizing And Runtime Control

- `auto_resize_slab`, `min_pbc_image_separation`
- `ts_optimizer`, `steps_between_swaps`
- `autobatcher_max_memory_padding`, `autobatcher_max_memory_scaler`, `autobatcher_max_atoms_to_try`
- `saturation_autobatcher_reuse`, `saturation_autobatcher_reuse_growth_atoms`, `saturation_autobatcher_reuse_growth_fraction`

### Bayesian Optimization

- `bo_enabled`
- `bo_initial_random`, `bo_batch_size`, `bo_total_budget`
- `bo_ucb_kappa`, `bo_acquisition`, `bo_surrogate`
- `bo_candidate_pool_size`
- `bo_include_failure_negatives`
- `bo_failure_penalty_default`, `bo_failure_penalty_overrides`
- `bo_transfer_enabled` and the `bo_transfer_*` trust, similarity, and weighting controls

### Saturation Behavior

- `saturation`
- `multi_molecule_saturation`

The saturation-related booleans are metadata and workflow-selection hints; the actual behavior is determined by which API is called and, for multi-molecule saturation, by the number of molecules loaded.

## Output Model

The default output root is `results_{surface_type}`.

Common artifacts include:

- `adsorption_energies_detailed.csv`
- `adsorption_energy_summary.csv`
- `saturation_details.csv`
- `saturation_summary.csv`
- `run_metadata.json`
- `xyz_structures/...`
- `vasp_inputs/...`

Key details:

- detailed CSV rows include placement-descriptor fields plus reproducibility context from the config,
- saturation outputs store per-step slab structures, adsorbate-only XYZ files, and BO transfer diagnostics,
- both `write_run_metadata(...)` and `write_run_settings(...)` write the same `run_metadata.json` path through a shared metadata builder, so later writes replace earlier content for that results directory.

## Dataset Logging And ML Support

The `ml` package is not limited to BO. It also supports:

- placement-level dataset logging,
- schema-versioned context rows,
- feature extraction from descriptors and saved datasets,
- grouped cross-validation,
- model training and evaluation,
- prediction and ranking utilities.

This makes the core workflow useful both as a simulator and as a data-generation engine for future surrogate models.

## Dependency And Runtime Model

Core dependencies include `numpy`, `ase`, `pandas`, `rdkit`, `scipy`, `scikit-learn`, and `spglib`.

Optional acceleration and model execution depend on packages such as `torch`, `torch-sim-atomistic`, and FairChem-related components. The package keeps these dependencies explicit and raises actionable errors when optional stacks are unavailable.

The codebase is built for modern Python, currently Python 3.12+.

## Design Summary

Metalsurfer is best understood as a layered adsorption-screening engine:

- deterministic candidate generation at the placement level,
- shared physical validation across all run modes,
- optional BO for sample efficiency,
- optional saturation for evolving-surface studies,
- structured outputs for reproducibility and downstream ML.

The central architectural choice is separation of concerns: surface preparation, candidate enumeration, optimization, filtering, orchestration, and persistence are implemented as distinct layers with typed interfaces between them.
