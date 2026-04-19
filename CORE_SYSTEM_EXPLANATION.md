# Metalsurfer: core system

## Purpose and scope

Developer-oriented architecture companion to [`README.md`](README.md): module layout, how runs execute end to end, and typed outputs. The README is the canonical overview of install, the four run-mode entry points, and quickstarts.

`scripts/` are not documented here.

## Public API layers

The package boundary in `metalsurfer.__init__` re-exports a curated set of symbols through lazy imports so heavy modules are loaded only when first accessed.

The public API is organized into five layers:

### 1. Run-mode APIs

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

### 2. Surface preparation API

`prepare_slab(...)` in `metalsurfer.surface_prep` provides a single call for bulk→slab construction, alloy substitution, and adatom deposition in one step.

### 3. Workflow APIs (internal)

These are the file-driven orchestration helpers used internally by the run-mode APIs:

- `_run_screening_common(...)`
- `run_saturation_screening(...)`

They load molecules from a SMILES CSV, compute references, execute the requested workflow, and return typed result collections.

### 4. Mid-level per-molecule APIs

These power workflow entry points and are useful when embedding the library in custom loops:

- `process_molecule(...)`
- `process_molecule_bayesian(...)`
- `calculate_reference_energies(...)`
- `load_molecules(...)`

They expose screening mechanics without requiring a full batch run.

### 5. Infrastructure APIs

Supporting helpers include:

- surface construction and editing: `create_slab_from_bulk`, `create_slab_from_atoms`, `substitute_alloy`, `deposit_adatoms`, `auto_resize_slab_for_molecule`
- placement generation: `enumerate_placement_specs`, `generate_placement_from_spec`, `generate_placement_from_descriptor`
- optimization and calculators: `setup_calculator`, `setup_single_model`, `setup_torchsim_model`, `optimize_isolated_molecules_batched`, `optimize_adsorbate_slab_batched`
- result persistence: `save_single_molecule_results`, `save_summary_results`, `save_saturation_results`, `save_multi_mol_saturation_results`, `write_run_metadata`, `write_run_settings`

## Module architecture

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

## Implementation mechanics

This section describes what the code actually does inside the modules named above. It is the right place to look when debugging geometry, symmetry, or batching behavior.

### Surfaces and slab containers

`surfaces.SlabContainer` wraps ASE `Atoms` with metadata used through screening. `create_slab_from_bulk` pulls a conventional cell from the Materials Project ecosystem, builds a slab with the requested Miller indices and vacuum, and returns a container. Alloy substitution and adatom deposition optionally draw random structural variants and rank them with a supplied calculator. `auto_resize_slab_for_molecule` expands in-plane supercells when PBC image separation would be too tight for the adsorbate footprint.

### Site detection (`placement/sites.py`)

1. **Framework geometry:** Atomic positions are replicated to periodic images (3×3×1 for typical slabs, 3×3×3 for fully periodic porous cells, none for clusters) before Voronoi construction.
2. **Voronoi vertices:** `scipy.spatial.Voronoi` runs on the extended point cloud. Vertices are filtered to the primary cell (where the cell matrix is invertible), then tested with a KDTree over framework atoms: each vertex must lie between `voronoi_probe_radius` and `voronoi_max_site_distance` from the nearest atom to count as an accessible adsorption site.
3. **Enrichment:** When `voronoi_site_enrichment` is true, ridges of the Voronoi graph are subdivided so sparse or stepped surfaces gain extra candidate points that pass the same accessibility tests.
4. **Typing:** Each vertex is classified as atop, bridge, hollow, or pore-like using distance ratios to nearby atoms, or—on slabs only—using a 2D Delaunay triangulation of top-layer atoms when `site_classification_method == "delaunay"`. Local surface normals and a small **environment fingerprint** (neighbor element multiset + site type) are stored on each site dict.
5. **Deduplication:** `_cluster_equivalent_sites` merges geometrically close sites using `scipy.spatial.KDTree.query_pairs` under a metric that depends on `material_type` (in-plane fractional + z for slabs, 3D fractional with MIC for porous, Cartesian for nanoparticles). Pairs are only merged when fingerprints match, so chemically distinct neighbors are not collapsed.

`placement.generators._get_unique_sites_for_specs` calls `get_unified_sites` with the user’s `AdsorptionConfig` Voronoi and classification fields, then runs `_cluster_equivalent_sites` with `site_equivalence_tolerance`. It returns a `SiteContext` (`sites`, `use_sites`, `source`).

### How the workflow chooses sites (`workflow/shared.py`)

`_resolve_site_context_for_sampling` is the single place that decides which site list feeds placement enumeration:

1. **Cache:** Results are memoized from a hash of slab positions, cell, PBC flags, and the boolean `symmetry_broken` (bounded cache, lock-protected).
2. **Core path:** It always builds the clustered Voronoi set via `_get_unique_sites_for_specs` first.
3. **Symmetry breaking:** If `symmetry_broken` is true (e.g. after prior saturation steps), symmetry reduction is skipped and the core sites are used.
4. **Symmetry reduction:** Otherwise it calls `get_symmetry_aware_sites`, which internally runs `get_unified_sites` again and then `SymmetryAnalyzer.analyze_site_symmetry`. The workflow currently passes `top_layer_tolerance`, `symmetry_tolerance`, `material_type`, and `enrich`; other Voronoi arguments use the defaults of `get_symmetry_aware_sites` unless the call site is extended. If that function returns a non-empty list, those sites replace the core list; if it returns `None` or an empty list, the workflow falls back to the core clustered sites.

### Placement enumeration and materialization

- **`placement/policy.py`:** `build_batch_placement_specs` expands a Cartesian product over conformers, site indices, orientation mode (including flat-aromatic parallel vs EN-down branches and a dissociative branch), discretized tilts, azimuths, and z-fractions. The raw grid is capped internally; if it is larger than `n_desired`, a uniform random subsample is taken with the given integer `seed`. `max_batch_placement_specs` uses the default `PLACEMENT_GRID_COUNT_SEED` so capacity counts stay consistent with that enumeration logic.
- **`placement/generators.py`:** `enumerate_placement_specs` supplies molecule- and slab-derived metadata (shape, binders, dissociative flag, site indices) and forwards `AdsorptionConfig.seed` into the policy layer. Optional `adaptive_parallel_fraction` adjusts the parallel vs EN-down split for flat aromatics from SMILES/symbol heuristics.
- **`workflow/shared._materialize_spec_placements`:** For each `PlacementSpec`, calls `generate_placement_from_spec_with_reason`. Successes yield combined slab+adsorbate `Atoms`, `PlacementDescriptor` rows, and integer placement ids; failures append `PlacementFailureEvent` records (used for BO negative feedback when enabled).

### Conformer generation (`conformers.py`)

RDKit parses SMILES, adds hydrogens, and embeds up to `num_conformers` conformers with `EmbedMultipleConfs(..., randomSeed=config.seed)`. Each conformer is MMFF-relaxed. If a TorchSim model or ASE calculator is provided, conformers are energy-scored (batched `batch_static` when a model is available); otherwise energies are placeholders. `remove_duplicate_conformers` collapses near-duplicate geometries using RMSD and energy thresholds from `AdsorptionConfig`.

### Relaxation (`optimization.py`)

`compute_frozen_indices` identifies which slab atoms are below the top layer (within `top_layer_tolerance`) and marks them frozen when `relax_top_layer` is true; adsorbate atoms are never frozen. `optimize_adsorbate_slab_batched` feeds variable-size combined systems through TorchSim’s `InFlightAutoBatcher`, applies fixed-atom constraints, and runs the configured optimizer (`fire`, `lbfgs`, or `bfgs`). For saturation, `base_slab_for_frozen` can point at the original clean slab so the freeze mask stays aligned with the pristine surface even as adsorbates accumulate.

### Post-relaxation filtering (`filters.py`)

`filter_results` runs one pipeline over `ScreeningResult` objects: **decomposition** checks (graph connectivity of the adsorbate at several multiples of covalent radii, elemental formula match to reference SMILES, bond-pair counts, and coordination-number fingerprints), **desorption** via minimum adsorbate–surface distance against `binding_distance_threshold`, **energy** cap via `max_adsorption_energy`, and **duplicate** removal among surviving structures.

### Symmetry analysis (`symmetry.py`)

`SymmetryAnalyzer` prepares either the ASE cell as a 3D periodic lattice or, for clusters, an orthorhombic box with padding so periodic images do not interact. It calls `spglib.get_symmetry_dataset` (errors wrapped as `SymmetryAnalysisError`). Symmetry operations are converted to 4×4 Cartesian matrices; equivalent sites are grouped by applying fractional-space operations with minimum-image-aware distance tests, optionally restricted to planar (xy) distances for flat top layers. Internal orbit checks verify that grouped sites are actually related by at least one operation within `symmetry_tolerance`.

### Bayesian screening (`workflow/bayesian.py` + `ml/bayesian.py`)

The workflow enumerates a finite candidate pool of `PlacementSpec` objects, evaluates an initial random subset, converts each evaluated structure’s `PlacementDescriptor` into feature rows (`ml/features.py`), and fits a surrogate (`train_surrogate`: tree ensembles or ridge/HGB). Unevaluated candidates are scored with LCB, EI, or PI (`lcb_scores`, `ei_scores`, `pi_scores`). Tree models expose epistemic spread as across-tree standard deviation; ridge/HGB use zero uncertainty so EI/PI degrade to deterministic ranking. Batches are selected until `bo_total_budget` is exhausted; optional failure penalties add synthetic negative labels when `bo_include_failure_negatives` is enabled.

## Run modes

### Standard screening

Standard screening enumerates a deterministic placement pool for each molecule, relaxes every sampled candidate, filters the optimized structures, and returns the surviving low-energy adsorption configurations.

This is the baseline mode when broad coverage is preferred over aggressive sample efficiency.

### Bayesian screening

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

### Sequential saturation

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

## End-to-end computational flow

Across run modes, the physical pipeline has the same major stages.

### 1. Surface preparation

Users may begin with a Materials Project bulk ID and Miller indices, or wrap existing ASE atoms.

Optional modifications include:

- alloy substitution via `substitute_alloy(...)`,
- adatom deposition via `deposit_adatoms(...)`,
- in-plane supercell expansion via `auto_resize_slab_for_molecule(...)` when periodic-image separation would otherwise be too small.

### 2. Reference energy construction

The reference stage computes:

- the clean slab energy,
- isolated-molecule energies for every molecule in the run.

Adsorption energy is then defined as:

`E_ads = E_adsorbate+slab - E_slab - E_molecule`

If a reference energy is missing, the workflow can either skip that molecule or fail hard, depending on `fail_on_missing_reference`.

### 3. Conformer generation

Molecules are embedded from SMILES with RDKit, optimized, optionally rescored with the MLIP backend, and deduplicated with RMSD and energy thresholds.

The conformer stage is controlled by fields such as:

- `num_conformers`,
- `conformer_sampling`,
- `boltzmann_temperature`,
- `energy_dedup_threshold`,
- `rmsd_dedup_threshold`.

### 4. Placement specification and materialization

Placement is driven by deterministic `PlacementSpec` objects. A spec records the abstract choice of conformer, site, orientation, tilt, azimuth, and height, while `PlacementDescriptor` records the resolved geometry used for replay and analysis.

The main placement path is:

`enumerate_placement_specs(...) -> generate_placement_from_spec(...) -> PlacementDescriptor`

Benefits of this design:

- reproducibility from seeds and descriptors,
- easy logging for downstream ML,
- clean separation between candidate enumeration and expensive optimization.

**Batch placement policy:** `build_batch_placement_specs` (in `metalsurfer.placement.policy`) enumerates the combinatorial grid of conformers, sites, orientations, tilts, azimuths, and z-fractions (subject to an internal size cap), then uniformly subsamples to `n_desired` when the grid is larger. The integer `seed` controls subsampling; the default `PLACEMENT_GRID_COUNT_SEED` aligns `max_batch_placement_specs` with the cardinality of an uncapped enumeration. `enumerate_placement_specs` forwards `AdsorptionConfig.seed`, or an explicit override. For step-by-step site detection, workflow resolution, and materialization, see **Implementation mechanics** above.

### 5. Optimization

Candidate adsorbate-slab systems are relaxed with the configured backend, typically TorchSim/FairChem through `setup_single_model(...)` and the batched optimization helpers.

The optimizer layer includes:

- top-layer detection,
- frozen-atom policies,
- configurable optimizers (`fire`, `lbfgs`, `bfgs`),
- GPU-memory-aware autobatching and cache reuse for saturation.

### 6. Validation and filtering

After relaxation, the system applies several postchecks:

- geometry validation,
- adsorption-distance validation,
- decomposition detection,
- duplicate removal,
- adsorption-energy cap filtering.

`skip_topology_check` and `skip_desorption_check` allow expert users to loosen these checks for special cases such as dissociative adsorption workflows.

### 7. Aggregation and persistence

Retained results are ranked by adsorption energy and converted into typed outputs such as:

- `ScreeningResult`
- `ScreeningRunResult`
- `SaturationStepResult`
- `SaturationRunResult`
- `MultiMolSaturationStepResult`
- `MultiMolSaturationRunResult`
- `BindingCampaignResult`

Persistence is handled separately from compute logic through `io_results.py`.

## Material types and site generation

`AdsorptionConfig.material_type` is an explicit public contract with three valid values:

- `"slab"`
- `"nanoparticle"`
- `"porous"`

This choice affects:

- periodic boundary handling,
- adsorption-site discovery,
- local surface-normal construction,
- adsorption validation distances.

### Voronoi-based sites

Adsorption sites are generated from Voronoi geometry over the framework atoms. The same general machinery is used across supported material types; the numbered pipeline (periodic images, vertex filtering, enrichment, typing, clustering) is spelled out under **Implementation mechanics**.

Important placement hyperparameters include:

- `voronoi_probe_radius`: minimum distance from framework atoms to a candidate site,
- `voronoi_max_site_distance`: upper distance bound for chemically relevant sites,
- `top_layer_tolerance`: top-layer identification for slabs,
- `symmetry_tolerance` and `site_equivalence_tolerance`: symmetry and deduplication tolerances.

Site classes include atop, bridge, hollow, and pore-like positions depending on local geometry.

### Symmetry handling

`symmetry.py` uses `spglib` to group equivalent sites. Failed dataset construction raises `SymmetryAnalysisError` (including wrapped `spglib` failures). Orbit construction and Cartesian operation mapping are described under **Implementation mechanics**.

`get_symmetry_aware_sites()` returns `None` if Voronoi yields no sites. If `SymmetryAnalyzer` raises `SymmetryAnalysisError`, the placement module logs a warning and returns `None`. The workflow then uses clustered Voronoi sites from `get_unified_sites` via `workflow.shared._resolve_site_context_for_sampling` (see **How the workflow chooses sites** in **Implementation mechanics**).

## Configuration model

`AdsorptionConfig` centralizes both physical and workflow configuration.

Representative groups of fields are:

### Sampling and placement

- `num_conformers`, `num_placements`
- `placement_x_range`, `placement_y_range`, `placement_z_range`
- `placement_z_scale_by_covalent_radius`
- `min_initial_distance`, `max_initial_distance`, `min_contact_ratio`
- `flat_aromatic_parallel_fraction`, `adaptive_parallel_fraction`
- `voronoi_probe_radius`, `voronoi_max_site_distance`, `voronoi_site_enrichment`
- `site_classification_method` (`distance_ratio` or `delaunay` for slabs)
- `rough_slab_local_z` (per-site surface reference on non-planar slabs)

### Relaxation and validation

- `model_name`, `device`
- `fmax`, `stage1_steps`, `stage2_steps`, `reference_optimization_steps`
- `relax_top_layer`, `freeze_symbols`
- `min_interatomic_distance`, `max_force_convergence`
- `binding_distance_threshold`, `max_adsorption_energy`

### Reproducibility and strictness

- `seed`
- `fail_on_missing_reference`
- `fail_on_conformer_failure`
- `debug_write_initial_placements`

### Surface sizing and runtime control

- `auto_resize_slab`, `min_pbc_image_separation`
- `ts_optimizer`, `steps_between_swaps`
- `autobatcher_max_memory_padding`, `autobatcher_max_memory_scaler`, `autobatcher_max_atoms_to_try`
- `saturation_autobatcher_reuse`, `saturation_autobatcher_reuse_growth_atoms`, `saturation_autobatcher_reuse_growth_fraction`

### Bayesian optimization

- `bo_enabled`
- `bo_initial_random`, `bo_batch_size`, `bo_total_budget`
- `bo_ucb_kappa`, `bo_acquisition`, `bo_surrogate`
- `bo_candidate_pool_size`
- `bo_include_failure_negatives`
- `bo_failure_penalty_default`, `bo_failure_penalty_overrides`
- `bo_transfer_enabled` and the `bo_transfer_*` trust, similarity, and weighting controls

### Saturation behavior

- `saturation`
- `multi_molecule_saturation`

The saturation-related booleans are metadata and workflow-selection hints; the actual behavior is determined by which API is called and, for multi-molecule saturation, by the number of molecules loaded.

## Output model

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

## Dataset logging and ML support

The `ml` package is not limited to BO. It also supports:

- placement-level dataset logging,
- schema-versioned context rows,
- feature extraction from descriptors and saved datasets,
- grouped cross-validation,
- model training and evaluation,
- prediction and ranking utilities.

This makes the core workflow useful both as a simulator and as a data-generation engine for future surrogate models.

## Dependencies and runtime

Core dependencies include `numpy`, `ase`, `pandas`, `rdkit`, `scipy`, `scikit-learn`, and `spglib`.

Optional acceleration and model execution depend on packages such as `torch`, `torch-sim-atomistic`, and FairChem-related components. The package keeps these dependencies explicit and raises actionable errors when optional stacks are unavailable.

The codebase is built for modern Python, currently Python 3.12+.

## Design summary

The library is best understood as a layered adsorption-screening engine:

- deterministic candidate generation at the placement level,
- shared physical validation across all run modes,
- optional BO for sample efficiency,
- optional saturation for evolving-surface studies,
- structured outputs for reproducibility and downstream ML.

The central architectural choice is separation of concerns: surface preparation, candidate enumeration, optimization, filtering, orchestration, and persistence are implemented as distinct layers with typed interfaces between them.
