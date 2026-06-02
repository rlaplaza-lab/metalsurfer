# Metalsurfer: core system

## Purpose and scope

Developer-oriented companion to [`README.md`](README.md) and the Sphinx [architecture guide](docs/guides/architecture.rst): module layout, implementation mechanics, and typed outputs. Install, run modes, and quickstarts stay in the README. Contributor workflow: [development guide](docs/guides/development.rst).

`scripts/` call `run_saturation` / `run_saturation_bo`; on-disk layout follows the campaign API (`step_*_placements/`, `saturation_placements_detailed.csv` when `saturation_save_all_placements` is true).

## Public API layers

Lazy re-exports in `metalsurfer.__init__` organize the public API into five layers:

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
`run_saturation` and `run_saturation_bo` return `SaturationCampaignResult` (access per-molecule runs via `.runs`).

With `save_results=True` (default), both call `save_saturation_results(..., config=config)` so files under `results_{surface_type}/` reflect the same `AdsorptionConfig` (including `saturation_save_all_placements` for per-step `step_*_placements/` trees and `saturation_placements_detailed.csv`).

### 2. Surface preparation API

`prepare_slab(...)` in `metalsurfer.surface_prep` provides a single call for bulk→slab construction, alloy substitution, and adatom deposition in one step.

### 3. Workflow APIs (internal)

These are orchestration helpers used internally by the run-mode APIs:

- `_run_screening_common(...)`
- `run_saturation_screening(...)`

They normalize molecule input (in-memory `(smiles, name)` pairs or SMILES CSV path), compute references, execute the requested workflow, and return typed result collections.

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
4. **Symmetry reduction:** Otherwise it calls `get_symmetry_aware_sites` with the same Voronoi parameters as site discovery and with `raw_sites` set to the unclustered list from step 2, so Voronoi is not run twice. `SymmetryAnalyzer.analyze_site_symmetry` then reduces to symmetry-unique orbits. If that succeeds and the result is non-empty, those sites replace the core clustered list. On `SymmetryAnalysisError`, or if there are no symmetry-reduced sites, the workflow keeps the clustered Voronoi sites from step 2 (errors are logged at INFO in `_resolve_site_context_for_sampling`).

### Placement enumeration and materialization

- **`placement/policy.py`:** `build_batch_placement_specs` expands a Cartesian product over conformers, site indices, orientation mode (including flat-aromatic parallel vs EN-down branches and a dissociative branch), discretized tilts, azimuths, and z-fractions. The raw grid is capped internally; if it is larger than `n_desired`, a uniform random subsample is taken with the given integer `seed`. `max_batch_placement_specs` uses the default `PLACEMENT_GRID_COUNT_SEED` so capacity counts stay consistent with that enumeration logic.
- **`placement/generators.py`:** `enumerate_placement_specs` supplies molecule- and slab-derived metadata (shape, binders, dissociative flag, site indices) and forwards `AdsorptionConfig.seed` into the policy layer. Optional `adaptive_parallel_fraction` adjusts the parallel vs EN-down split for flat aromatics from SMILES/symbol heuristics.
- **`workflow/shared._materialize_spec_placements`:** For each `PlacementSpec`, calls `generate_placement_from_spec_with_reason`. Successes yield combined slab+adsorbate `Atoms`, `PlacementDescriptor` rows, and integer placement ids; failures append `PlacementFailureEvent` records (used for BO negative feedback when enabled).

### Conformer generation (`conformers.py`)

RDKit parses SMILES, adds hydrogens, and embeds up to `num_conformers` conformers with `EmbedMultipleConfs(..., randomSeed=config.seed)`. Each conformer is MMFF-relaxed. If a TorchSim model or ASE calculator is provided, conformers are energy-scored (batched `batch_static` when a model is available); otherwise energies are placeholders. `remove_duplicate_conformers` collapses near-duplicate geometries using RMSD and energy thresholds from `AdsorptionConfig`.

### Relaxation (`optimization.py`)

There are **two relaxation phases** with different freeze policies:

1. **Surface preparation** (`create_slab_from_bulk`, `deposit_adatoms`, `prepare_slab`) uses ASE optimizers and `AdsorptionConfig.slab_relaxation_mode` (`none`, `ionic_only`, `cell_only`, `full`). This is where the clean slab and any adatoms are equilibrated once before screening. Output files such as `clean_slab.xyz` (pre-adatom) and `clean_slab_Au20.xyz` (after adatom deposition) record that prep geometry; they are **not** the TorchSim freeze reference unless you pass that structure into the workflow yourself.

2. **Adsorption / saturation** uses TorchSim `FixAtoms` via `optimize_adsorbate_slab_batched`. `compute_frozen_indices` identifies which slab atoms are frozen: when `relax_top_layer` is true, only atoms below the top layer (within `top_layer_tolerance`) stay fixed; when `relax_top_layer` is false, every atom in the freeze reference is fixed. Adsorbate atoms in the combined system are never frozen. For saturation, `base_slab_for_frozen` is captured at the start of `run_saturation_screening` (post-`prepare_slab`) so indices `0 .. len(base_slab)-1` stay fixed while later adsorbate units (indices `>= len(base_slab)` on the evolving slab) may still relax. If `auto_resize_slab` repeats the substrate in-plane on step 1, every workflow path (standard, Bayesian, saturation) expands that freeze reference to the full resized substrate—regardless of `relax_top_layer`—so repeated tiles are not left free to move. Competitive multi-molecule saturation pre-resizes once before the step-1 candidate loop.

### Post-relaxation filtering (`filters.py`)

`filter_results` runs one pipeline over `ScreeningResult` objects: **decomposition** checks (graph connectivity of the adsorbate at several multiples of covalent radii, elemental formula match to reference SMILES, bond-pair counts, and coordination-number fingerprints), **desorption** via minimum adsorbate–surface distance against `binding_distance_threshold`, **energy** cap via `max_adsorption_energy`, and **duplicate** removal among surviving structures. During saturation placement, decomposition uses `adsorbate_prefix_atoms=len(slab)` so only the newly added adsorbate is checked; `adsorbate_connected_components` supports the separate saturation-step guard that re-validates every adsorbate unit on the slab before best-slab selection.

### Symmetry analysis (`symmetry.py`)

`SymmetryAnalyzer` prepares either the ASE cell as a 3D periodic lattice or, for clusters, an orthorhombic box with padding so periodic images do not interact. It calls `spglib.get_symmetry_dataset` (errors wrapped as `SymmetryAnalysisError`). Symmetry operations are converted to 4×4 Cartesian matrices; equivalent sites are grouped by applying fractional-space operations with minimum-image-aware distance tests, optionally restricted to planar (xy) distances for flat top layers. Internal orbit checks verify that grouped sites are actually related by at least one operation within `symmetry_tolerance`.

### Bayesian screening (`workflow/bayesian.py` + `ml/bayesian.py`)

Enumerate a finite `PlacementSpec` pool, evaluate a random initial subset, build features from each `PlacementDescriptor`, and fit a surrogate with `train_surrogate` (tree ensembles, `HistGradientBoostingRegressor`, or ridge). Score remaining candidates with LCB, EI, or PI; tree models use forest variance for uncertainty, while ridge and HGB report no uncertainty (EI/PI use the deterministic limits in the acquisition helpers). Iterate in batches until `bo_total_budget`. When `bo_include_failure_negatives` is set, failed placements are labeled with configurable penalties. Transfer learning (`bo_transfer_enabled`) uses per-sample weights and requires `random_forest` or `extra_trees` (validated in `AdsorptionConfig`).

## Run modes and pipeline

High-level flow (surface prep → references → conformers → placement → relaxation → filtering → persistence) is summarized in [docs/guides/architecture.rst](docs/guides/architecture.rst). This file focuses on code-level behavior.

**Standard screening** — enumerate placements, relax all sampled candidates, filter, rank.

**Bayesian screening** — same pipeline; surrogate-guided candidate selection (see **Bayesian screening** under **Implementation mechanics**).

**Sequential saturation** — repeat screening on an evolving slab until `E_ads ≥ 0` or no valid placements. Notable flags:

- compare optimized structures to the post-prep substrate file (e.g. `clean_slab_Au20_POSCAR` when adatoms were deposited), not `clean_slab.xyz` from before adatoms;
- auto-resize only on the first step (freeze reference is updated if the substrate is repeated in-plane);
- `multi_molecule_saturation=True` with multiple molecules → competitive per-step selection;
- `saturation_discard_topology_rearrangements=True` (default): connectivity-only guard on the full adsorbate pool before ranking (`adsorbate_connected_components`); set `False` for energy-only ranking; skipped when `skip_topology_check=True`.

Adsorption energy: `E_ads = E_adsorbate+slab - E_slab - E_molecule`. Typed outputs (`ScreeningResult`, `SaturationRunResult`, `BindingCampaignResult`, …) are persisted via `io_results.py`.

## Material types and site generation

`AdsorptionConfig.material_type` is one of `"slab"`, `"nanoparticle"`, or `"porous"`. Site dicts from `get_unified_sites` always include `material_type`; placement code requires that key when a site is provided (use `config.material_type` when there is no site).

This choice affects:

- periodic boundary handling,
- adsorption-site discovery,
- local surface-normal construction,
- adsorption validation distances.

### Voronoi and symmetry

Site discovery uses Voronoi over framework atoms (pipeline under **Site detection** in **Implementation mechanics**). Key hyperparameters: `voronoi_probe_radius`, `voronoi_max_site_distance`, `top_layer_tolerance`, `symmetry_tolerance`, `site_equivalence_tolerance`.

`get_symmetry_aware_sites()` may raise `SymmetryAnalysisError`; the workflow then falls back to clustered Voronoi sites from `_get_unique_sites_for_specs` (see **How the workflow chooses sites**).

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
- `saturation_save_all_placements`
- `saturation_discard_topology_rearrangements` (default `True`: discard rearranged/coupled candidates before per-step best-slab selection)
- `saturation_autobatcher_reuse`, `saturation_autobatcher_reuse_growth_atoms`, `saturation_autobatcher_reuse_growth_fraction`

The `saturation` and `multi_molecule_saturation` fields are metadata and workflow-selection hints; the actual behavior is determined by which API is called and, for multi-molecule saturation, by the number of molecules loaded.

## Output model

Default root: `results_{surface_type}/` with detailed/summary CSVs, `run_metadata.json`, `xyz_structures/`, `vasp_inputs/`. Saturation runs optionally write `step_{NNN}_placements/` and `saturation_placements_detailed.csv` when `saturation_save_all_placements` is true. `write_run_metadata` and `write_run_settings` share the same metadata builder (later writes replace earlier content).

## Dataset logging and ML support

The `ml` package is not limited to BO. It also supports:

- placement-level dataset logging,
- schema-versioned context rows,
- feature extraction from descriptors and saved datasets,
- grouped cross-validation,
- model training and evaluation,
- prediction and ranking utilities.

## Dependencies and runtime

Core dependencies include `numpy`, `ase`, `pandas`, `rdkit`, `scipy`, `scikit-learn`, and `spglib`.

Optional acceleration and model execution depend on packages such as `torch`, `torch-sim-atomistic`, and FairChem-related components. The package keeps these dependencies explicit and raises actionable errors when optional stacks are unavailable.

The codebase is built for modern Python, currently Python 3.12+.

## Design summary

Layers: surface prep → candidate enumeration → optimization → filtering → orchestration → I/O, with typed dataclasses between stages. BO and saturation are optional paths on the same core.
