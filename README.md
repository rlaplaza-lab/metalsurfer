# Metalsurfer

![Metalsurfer Logo](docs/_static/logo_metalsurfer.svg)

Library for adsorption on arbitrary materials (slabs, nanoparticles, and periodic porous frameworks).

Metalsurfer is substrate-agnostic: pass any ASE `Atoms` object—periodic slab, fully periodic porous framework, or non-periodic cluster—after optional prep with `prepare_substrate` (equilibration, PBC, ASE `FixAtoms`). Supply adsorbates as SMILES; the library builds conformers, finds adsorption sites (orientation-aware Voronoi/topology hybrid, material-aware via `AdsorptionConfig.material_type`), deposits candidates with orientation/height sampling, relaxes with an MLIP, validates geometry, and ranks by adsorption energy. The four `run_*` campaign APIs orchestrate screening, Bayesian placement search, or sequential saturation on that pipeline.

**Documentation**: https://metalsurfer.readthedocs.io

## Install

Requires **Python 3.12 or newer**.

Core dependencies only:

```bash
pip install -e .
```

For TorchSim/FairChem-backed relaxation and the developer toolchain:

```bash
pip install -e ".[mlip,dev]"
```

The `[dev]` extra includes **ruff**, **mypy**, type stubs, and pytest tooling.

## Quick Examples

Examples in `examples/` demonstrate basic usage and an advanced saturation workflow:

### H2 Adsorption on Pt Nanocluster
```bash
# 12-atom Pt cluster with H2 adsorption
python examples/h2_pt12_binding_energy.py
```

### CO2 Adsorption in MOF
```bash
# Real MOF structure (RUBTAK01) with CO2 adsorption
python examples/co2_mof_binding_energy.py
```

### Ethene Adsorption on Ru(0001) Slab
```bash
# Ru(0001) slab with ethene adsorption
python examples/ethene_ru_slab_binding_energy.py
```

### Bipyridine Saturation on Defected Au(111)
```bash
# Au(111) with 20% Au adatoms; fixed substrate during placement relaxations
python examples/bipyridine_au111_defects_saturation_raw.py
```

### Camphor on Cu(111) (Bayesian, GPU-heavy)
```bash
# BO placement search on a literature DFT slab (slab_relaxation_mode="none")
python examples/camphor_cu111_binding_energy.py
```

The HPC-oriented copy of the bipyridine workflow is `scripts/bipyridine_au111_defects_saturation_raw.py`.

These examples span Pt, Ru, MOF, Au(111), and Cu(111); the same API accepts any ASE `Atoms` or prepared slab.

- Use pure ASE for receptor preparation (or `prepare_substrate` from a bulk id)
- Quick demos use modest explicit placement counts; omit `num_placements` to autotune to GPU parallel capacity (see `AdsorptionConfig`)
- The bipyridine HPC script uses many more placements (see `scripts/`)
- Demonstrate different material types (`nanoparticle`, `porous`, and `slab`)
- Produce XYZ structures and CSV results (VASP inputs are opt-in via `write_vasp_inputs=True`)

## Python API

The library exposes four high-level entry points:

| Function | Role |
| -------- | ---- |
| `run_adsorption` | Standard screening: enumerate placements, relax, filter, rank. |
| `run_adsorption_bo` | Same pipeline with Bayesian optimization over placement candidates. |
| `run_saturation` | Sequential saturation: repeated adsorption onto an evolving slab. Returns `SaturationCampaignResult`. |
| `run_saturation_bo` | Saturation with BO-guided placement selection. Returns `SaturationCampaignResult`. |

Each accepts either an in-memory `list[tuple[str, str]]` of `(smiles, name)` pairs or a path to a SMILES CSV as `molecules`. `run_saturation` and `run_saturation_bo` require an explicit `molecules` argument (there is no default file). With `skip_existing=True` (default), binding campaigns skip molecules already in `adsorption_energies_detailed.csv`; saturation campaigns skip molecules already in `saturation_summary.csv` (both input forms).

### Surfaces: ASE Atoms, bulk prep, or containers

Surfaces are **not** tied to a specific element. Pass any `ase.Atoms` you already have (clusters, slabs, MOFs from CIF, saturated intermediates from XYZ), or build one with `prepare_substrate(bulk_id=...)`. Use `SlabContainer` only when you need its metadata helpers.

`AdsorptionConfig.material_type` (`slab`, `nanoparticle`, or `porous`) controls placement and validation geometry, not the chemical symbols in the structure.

All four `run_*` entry points accept plain `Atoms` or `SlabContainer`, but the substrate must be **campaign-ready** before the call: **equilibrated ionic positions** (via `prepare_substrate`, default `slab_relaxation_mode="ionic_only"`), correct PBC for `material_type`, bottom-anchored slab geometry when applicable, and ASE `FixAtoms` attached during prep (default: entire substrate frozen).

**Slab geometry:** For `material_type="slab"`, set the adsorption surface at `max(z)` with vacuum above. Alignment, PBC, freeze constraints, and in-plane sizing happen during **prep** (`prepare_substrate`, `create_slab_from_atoms`, `resize_substrate_for_molecule` from `metalsurfer.surface_prep`) — campaign APIs validate only. See the [surface engineering guide](https://metalsurfer.readthedocs.io/en/latest/guides/surface_engineering.html) for details.

**Prep vs adsorption:** `slab_relaxation_mode` equilibrates the substrate **before** campaigns. During placement relaxation, only adsorbate atoms and any substrate atoms **not** in ASE `FixAtoms` can move. With `relax_top_layer=True` on `prepare_substrate`, the free atoms depend on `material_type` (slab top layer, nanoparticle outer shell, porous pore boundary). See the [surface engineering guide](https://metalsurfer.readthedocs.io/en/latest/guides/surface_engineering.html). For custom patterns, attach your own ASE constraints during prep.

Example:

```python
from ase.build import fcc111

from metalsurfer import AdsorptionConfig, run_adsorption
from metalsurfer.surface_prep import prepare_substrate

config = AdsorptionConfig(
    material_type="slab",  # "slab", "nanoparticle", or "porous"
    seed=42,
)
slab = prepare_substrate(
    slab=fcc111("Ru", size=(3, 3, 3), vacuum=12.0),
    config=config,
    results_dir="results_ru111_from_ase",
)
result = run_adsorption(
    slab=slab,
    molecules=[("O", "water")],
    config=config,
    surface_type="ru111_from_ase_atoms",
)
```

You may pass `SlabContainer` or bare `Atoms` to `run_*` once prep is complete.

### Slab sizing and PBC

Prepare substrates **outside** campaign APIs:

- Use a large enough `supercell` in `prepare_substrate`, or call `auto_resize_substrate_for_molecule` / `resize_substrate_for_molecule` (from `metalsurfer.surface_prep`) after conformer generation.
- `min_pbc_image_separation` (default 8 Å) controls the resize helper.
- Campaign entry validates PBC, slab anchoring, vacuum, and freeze constraints. Adsorbate-size / in-plane image-separation checks run after conformer generation (use the resize helpers during prep when needed).

### 1. Standard Screening

Use the campaign API when your driving script already has the molecule list in memory and you want a typed `BindingCampaignResult` back.

```python
from metalsurfer import AdsorptionConfig, run_adsorption
from metalsurfer.surface_prep import prepare_substrate

config = AdsorptionConfig(
    material_type="slab",  # "slab", "nanoparticle", or "porous"
    seed=42,
    num_conformers=8,
    num_placements=80,  # or omit to autotune to GPU parallel capacity
)

slab = prepare_substrate(
    bulk_id="mp-33",
    miller_indices=(0, 0, 1),
    config=config,
    results_dir="results_Ru0001",
)

molecules = [
    ("CC", "ethane"),
    ("C=C", "ethene"),
    ("C#C", "acetylene"),
]

result = run_adsorption(
    slab=slab,
    molecules=molecules,
    config=config,
    surface_type="Ru0001",
)

print(result.mode)
print(result.total_configurations)
for summary in result.molecule_summaries:
    print(summary.molecule, summary.best_adsorption_energy)
```

Pass a CSV path to `run_adsorption` for file-driven batch screening (same XYZ/CSV outputs and `BindingCampaignResult` fields as an in-memory list). CSV files use two columns `(smiles, name)` and may include an optional header row (`smiles,molecule`):

```python
from metalsurfer import AdsorptionConfig, run_adsorption
from metalsurfer.surface_prep import prepare_substrate

config = AdsorptionConfig(
    material_type="slab",  # "slab", "nanoparticle", or "porous"
    seed=42
)
slab = prepare_substrate(
    bulk_id="mp-33",
    miller_indices=(0, 0, 1),
    config=config,
    results_dir="results_Ru0001",
)

result = run_adsorption(
    slab=slab,
    molecules="molecules.csv",
    config=config,
    surface_type="Ru0001",
    skip_existing=True,  # default: skip molecules already in adsorption_energies_detailed.csv
)
```

Use `run_adsorption_bo` (not `bo_enabled=True` on `run_adsorption`) for Bayesian placement search; the non-BO entry point emits a warning if `bo_enabled=True` is set on the config.

### 2. Bayesian Screening

Bayesian mode keeps the same physical pipeline and output types, but replaces exhaustive placement evaluation with surrogate-guided candidate selection.

```python
from metalsurfer import (
    AdsorptionConfig,
    run_adsorption_bo,
)
from metalsurfer.surface_prep import prepare_substrate

config = AdsorptionConfig(
    material_type="slab",  # "slab", "nanoparticle", or "porous"
    seed=42,
    bo_enabled=True,  # defaults: ridge surrogate, EI acquisition, autotuned batch sizes
)

slab = prepare_substrate(
    bulk_id="mp-33",
    miller_indices=(0, 0, 1),
    config=config,
    results_dir="results_Ru0001_bo",
)

result = run_adsorption_bo(
    slab=slab,
    molecules=[("O=C=O", "co2"), ("O", "water")],
    config=config,
    surface_type="Ru0001_bo",
)

print(result.mode)
print(result.failure_summaries)
```

Relevant BO configuration fields live on `AdsorptionConfig`:

- `num_placements` (default `None`: autotune to GPU parallel capacity at runtime)
- `bo_initial_random`, `bo_batch_size` (default `None`: autotune to GPU parallel capacity)
- `bo_total_budget` (default `18`: acquisition batches after the initial random batch)
- Total evaluations once auto fields resolve: `bo_initial_random + bo_total_budget * bo_batch_size`
- `bo_acquisition` with `"lcb"`, `"ei"`, or `"pi"`
- `bo_surrogate` with `"random_forest"`, `"extra_trees"`, `"gradient_boost"`, `"ridge"`, or `"ensemble"` (default: `"ridge"`)
- `bo_transfer_*` for saturation transfer BO (default: weighted mode with 2-step memory window, recency and occupancy decay; `gradient_boost` does not support transfer sample weights)
- `bo_include_failure_negatives` and `bo_failure_penalty_*` for learning from failed placements

### 3. Sequential Saturation

Saturation mode repeatedly adsorbs the current best configuration onto the evolving slab until adsorption is no longer favorable or no valid placements remain.

```python
from metalsurfer import AdsorptionConfig, MultiMolSaturationRunResult, run_saturation
from metalsurfer.surface_prep import prepare_substrate

config = AdsorptionConfig(
    material_type="slab",  # "slab", "nanoparticle", or "porous"
    seed=42,
    num_conformers=6,
    num_placements=60,
)

slab = prepare_substrate(
    bulk_id="mp-33",
    miller_indices=(0, 0, 1),
    config=config,
    results_dir="results_Ru0001_sat",
)

# Persists to results_Ru0001_sat/ when save_results=True (default), using the same
# config (per-step best slabs plus step_*_placements/ when saturation_save_all_placements is true).
campaign = run_saturation(
    slab=slab,
    molecules="molecules.csv",
    config=config,
    surface_type="Ru0001_sat",
)

for entry in campaign.runs:
    if isinstance(entry, MultiMolSaturationRunResult):
        print(entry.molecules, entry.n_molecules_at_saturation)
    else:
        print(entry.molecule, entry.n_molecules_at_saturation)
```

Important saturation behaviors:

- **Prep vs adsorption relaxation:** `slab_relaxation_mode` (default `ionic_only`) equilibrates substrate ionic positions during `prepare_substrate`. Freeze policy is written to ASE `FixAtoms` via prep kwargs (default: entire substrate frozen). `relax_top_layer=True` is a material-aware shortcut (see [surface engineering guide](https://metalsurfer.readthedocs.io/en/latest/guides/surface_engineering.html)). Saturation pins `base_slab` at campaign start. Compare optimized structures to the matching prep snapshot (e.g. `clean_slab_Au20_*` after adatoms), not pre-adatom `clean_slab` files.
- In-plane supercell expansion must be done during prep (`auto_resize_substrate_for_molecule` / `resize_substrate_for_molecule` from `metalsurfer.surface_prep`) before calling campaign APIs.
- Call `run_saturation_bo` for Bayesian saturation; it forces BO on. The `bo_transfer_*` settings control cross-step observation reuse.
- When `multi_molecule_saturation=True` and multiple molecules are provided (in-memory list or CSV), the workflow switches to competitive saturation, where molecules compete for each step and the best overall adsorption wins.
- Competitive saturation with BO: call `run_saturation_bo`; each adsorbate trains and carries forward its own BO state independently (observations are not shared across adsorbates).
- By default, `saturation_save_all_placements=True` writes every validated placement per step under `xyz_structures/.../step_{NNN}_placements/`, plus `saturation_placements_detailed.csv`. Matching `vasp_inputs/...` trees are written only when `write_vasp_inputs=True`. Set `saturation_save_all_placements=False` to persist only the per-step best structures (smaller disk use).
- By default, `saturation_discard_topology_rearrangements=True` re-checks the full adsorbate pool on each candidate **before** choosing the step winner: adsorbates must form the expected number of connected fragments (connectivity-only guard). This catches inter-adsorbate coupling or unexpected splitting that per-placement filtering can miss while allowing strong adsorbate-material interactions that preserve adsorbate connectivity. Set `False` to rank only by `E_ads`; the guard is also skipped when `skip_topology_check=True`.
- Contributor test markers (`gpu`, `slow`): see the [development guide](https://metalsurfer.readthedocs.io/en/latest/guides/development.html).

### Surface setup and modifiers

Use [`metalsurfer.surface_prep`](https://metalsurfer.readthedocs.io/en/latest/api/surface_prep.html) as the single import path for substrate preparation. The orchestrator is [`prepare_substrate`](https://metalsurfer.readthedocs.io/en/latest/api/surface_prep.html#metalsurfer.surface_prep.prepare_substrate).

```python
from metalsurfer import AdsorptionConfig
from metalsurfer.surface_prep import prepare_substrate

config = AdsorptionConfig(material_type="slab", seed=42)

slab = prepare_substrate(
    bulk_id="mp-33",
    miller_indices=(0, 0, 1),
    alloy_host="Ru",
    alloy_guest="Cu",
    alloy_fraction=0.25,
    adatom_symbol="Sn",
    adatom_coverage=0.20,
    config=config,
    results_dir="results_Ru0001",
    adatom_relaxation_mode="ionic_only",  # optional: full clean slab once, ionic-only after adatoms
)
```

See the [Surface Engineering guide](https://metalsurfer.readthedocs.io/en/latest/guides/surface_engineering.html) for prep relaxation presets and substrate freeze behavior during adsorption.

For adatoms on an existing slab (e.g. frozen-base workflows), pass `slab=` after building the base:

```python
base_slab = prepare_substrate(bulk_id="mp-33", miller_indices=(0, 0, 1), config=config, results_dir=results_dir)
slab = prepare_substrate(
    slab=base_slab,
    adatom_symbol="Sn",
    adatom_coverage=0.10,
    config=config,
    results_dir=results_dir,
)
```

Lower-level helpers (`create_slab_from_bulk`, `substitute_alloy`, `deposit_adatoms`) are available from `metalsurfer.surface_prep` for custom research loops; finalize with `prepare_substrate(slab=...)` or `finalize_substrate` after manual PBC + `apply_surface_constraints` before calling campaign APIs.

`AdsorptionConfig.material_type` must be chosen explicitly:

- `"slab"`: in-plane periodic surfaces.
- `"nanoparticle"`: non-periodic clusters.
- `"porous"`: fully periodic porous frameworks.

This choice affects site generation, adsorption validation, and distance handling throughout the workflow.

## What the core pipeline does

See the introduction above for the high-level mental model. Across all run modes, the library follows the same structure:

1. Build or accept a surface structure.
2. Generate and deduplicate molecular conformers.
3. Enumerate deterministic `PlacementSpec` candidates over conformer, site, orientation, tilt, azimuth, and height. Site detection is orientation-aware (slab normal, hybrid topology + Voronoi); BO features use materialized absolute geometry only (`x_abs`, `y_abs`, `z_abs`, quaternion)—not `site_index` or orientation labels.
4. Materialize placements into full adsorbate-slab structures.
5. Relax structures with the configured MLIP backend.
6. Validate adsorption geometry and filter decomposed, desorbed, or duplicate structures.
7. Rank surviving structures and persist structures, CSV summaries, and metadata.

Placement generation is **orientation-aware** (slab normal, hybrid topology + Voronoi) and works across slabs, nanoparticles, and porous materials. Bayesian mode changes candidate selection, not the downstream physics or filtering stack; surrogate inputs are resolved absolute poses, not discrete site IDs.

## Results and persistence

The output directory is `results_{surface_type}`. Depending on run mode, the library may write:

- `adsorption_energies_detailed.csv`
- `adsorption_energy_summary.csv`
- `saturation_details.csv`
- `saturation_placements_detailed.csv` (saturation runs when `saturation_save_all_placements` is true: one row per step × placement with paths and descriptor context)
- `saturation_summary.csv`
- `run_metadata.json` (when `write_settings=True` and/or `write_metadata=True` on campaign APIs; merged incrementally)
- `ml_dataset.csv`, `ml_dataset_metadata.json` (from `DatasetLogger` during binding campaigns and saturation)
- `xyz_structures/...`
- `vasp_inputs/...` (only when `write_vasp_inputs=True`)

Campaign APIs save XYZ structures and summary tables by default (`run_saturation` / `run_saturation_bo` call `save_saturation_results(..., config=config)` so placement-tree output follows `saturation_save_all_placements` and the rest of `AdsorptionConfig`). VASP bundles require `write_vasp_inputs=True` on `AdsorptionConfig`. Workflow APIs return typed results and can be paired with `save_summary_results(...)`, `save_saturation_results(...)`, `save_multi_mol_saturation_results(...)`, and `write_run_metadata(...)` / `write_run_settings(...)` (both merge into `run_metadata.json`) for explicit persistence control (for example after `save_results=False` or custom paths).

Use `results_dir(surface_type)` from `metalsurfer.io_results` (or `metalsurfer.results_dir` via lazy import) for the canonical `results_{surface_type}/` path.

## Logging

Call `configure_logging()` at the start of scripts (all `examples/` and `scripts/` already do). Workflows emit structured logs with optional context (`molecule`, `surface_type`, `placement_id`, `seed`) via `log_context`.

Environment overrides:

- `METALSURFER_LOG_LEVEL` (default: `INFO`)
- `TORCHSIM_LOG_LEVEL` (default: `WARNING`)

Logs go to **stdout** by default so HPC job `.out` files capture progress. TorchSim stdout/stderr is captured during relaxation (`torchsim_output_capture` in `src/metalsurfer/_logging.py`).

## Development

```bash
pip install -e ".[dev]"    # GPU stack: ".[mlip,dev]"
ruff check . && ruff format --check . && mypy src/metalsurfer
python -m pytest tests/ -m "not dependency_behavior and not mlip and not gpu and not slow" \
  --cov=src/metalsurfer --cov-report=term-missing --tb=short -v
coverage report --fail-under=74
```

CI parity, coverage gates, GPU/slow test jobs: [development guide](https://metalsurfer.readthedocs.io/en/latest/guides/development.html). Architecture: [CORE_SYSTEM_EXPLANATION.md](CORE_SYSTEM_EXPLANATION.md).
