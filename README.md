# Metalsurfer

![Metalsurfer Logo](docs/_static/logo_metalsurfer.svg)

Library for adsorption on arbitrary materials (slabs, nanoparticles, and periodic porous frameworks).

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

The ``[dev]`` extra includes **ruff**, **mypy**, type stubs, and pytest tooling.

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

The HPC-oriented copy of this workflow is `scripts/bipyridine_au111_defects_saturation_raw.py`.

These examples span Pt, Ru, MOF, and Au(111); the same API accepts any ASE `Atoms` or prepared slab.

- Use pure ASE for receptor preparation (or `prepare_slab` from a bulk id)
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

Each accepts either an in-memory `list[tuple[str, str]]` of `(smiles, name)` pairs or a path to a SMILES CSV as `molecules`.

### Surfaces: ASE Atoms, bulk prep, or containers

Surfaces are **not** tied to a specific element. Pass any `ase.Atoms` you already have (clusters, slabs, MOFs from CIF, saturated intermediates from XYZ), or build one with `prepare_slab(bulk_id=...)` / `create_slab_from_bulk`. Use `SlabContainer` only when you need its metadata helpers.

`AdsorptionConfig.material_type` (`slab`, `nanoparticle`, or `porous`) controls placement and validation geometry, not the chemical symbols in the structure.

All four `run_*` entry points accept plain `Atoms` directly (recommended for custom scripts):

**Slab geometry:** For `material_type="slab"`, set the adsorption surface at `max(z)` with vacuum above; `create_slab_from_atoms`, `create_slab_from_bulk`, and bare `Atoms` with `material_type="slab"` are normalized automatically. See the [surface engineering guide](docs/guides/surface_engineering.rst) for details.

Example:

```python
from ase.build import fcc111

from metalsurfer import AdsorptionConfig, create_slab_from_atoms, run_adsorption

slab = create_slab_from_atoms(fcc111("Ru", size=(3, 3, 3), vacuum=12.0))

config = AdsorptionConfig(
    material_type="slab",  # "slab", "nanoparticle", or "porous"
    seed=42
)
result = run_adsorption(
    slab=slab,
    molecules=[("O", "water")],
    config=config,
    surface_type="ru111_from_ase_atoms",
)
```

`create_slab_from_atoms(...)` wraps bare `Atoms` in a `SlabContainer`; you may also pass `Atoms` directly when `material_type="slab"` is set.

### Slab sizing and PBC

For periodic slabs, sizing is flexible and material-agnostic:

- `auto_resize_slab` (default `True` for screening): repeats the substrate in-plane when `min_pbc_image_separation` requires it.
- Saturation: auto-resize runs only on **step 1**; if the substrate is tiled in-plane, the freeze reference expands to every repeated substrate tile.
- Tune per system with `supercell`, `min_pbc_image_separation`, or `auto_resize_slab=False` (as in the bipyridine example when the initial supercell is already large enough).

### 1. Standard Screening

Use the campaign API when your driving script already has the molecule list in memory and you want a typed `BindingCampaignResult` back.

```python
from metalsurfer import AdsorptionConfig, prepare_slab, run_adsorption

config = AdsorptionConfig(
    material_type="slab",  # "slab", "nanoparticle", or "porous"
    seed=42,
    num_conformers=8,
    num_placements=80,  # omit for None → autotune to GPU parallel capacity
)

slab = prepare_slab(
    bulk_id="mp-33",
    miller_indices=(0, 0, 1),
    config=config,
    results_dir="results_Ru001",
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
    surface_type="Ru001",
)

print(result.mode)
print(result.total_configurations)
for summary in result.molecule_summaries:
    print(summary.molecule, summary.best_adsorption_energy)
```

Pass a CSV path to `run_adsorption` for file-driven batch screening:

```python
from metalsurfer import AdsorptionConfig, prepare_slab, run_adsorption

config = AdsorptionConfig(
    material_type="slab",  # "slab", "nanoparticle", or "porous"
    seed=42
)
slab = prepare_slab(
    bulk_id="mp-33",
    miller_indices=(0, 0, 1),
    config=config,
    results_dir="results_Ru001",
)

result = run_adsorption(
    slab=slab,
    molecules="molecules.csv",
    config=config,
    surface_type="Ru001",
)
```

### 2. Bayesian Screening

Bayesian mode keeps the same physical pipeline and output types, but replaces exhaustive placement evaluation with surrogate-guided candidate selection.

```python
from metalsurfer import (
    AdsorptionConfig,
    prepare_slab,
    run_adsorption_bo,
)

config = AdsorptionConfig(
    material_type="slab",  # "slab", "nanoparticle", or "porous"
    seed=42,
    bo_enabled=True,  # defaults: ridge/ei, autotune batch sizes, 18 acquisition batches
)

slab = prepare_slab(
    bulk_id="mp-33",
    miller_indices=(0, 0, 1),
    config=config,
    results_dir="results_Ru001_bo",
)

result = run_adsorption_bo(
    slab=slab,
    molecules=[("O=C=O", "co2"), ("O", "water")],
    config=config,
    surface_type="Ru001_bo",
)

print(result.mode)
print(result.failure_summaries)
```

Relevant BO configuration fields live on `AdsorptionConfig`:

- `num_placements` (default `None`: autotune to GPU parallel capacity at runtime)
- `bo_initial_random`, `bo_batch_size` (default `None`: autotune to GPU parallel capacity), `bo_total_budget` (default `18`: number of acquisition batches after the initial random batch; total evaluations = `bo_initial_random + bo_total_budget * bo_batch_size` once auto fields resolve)
- `bo_acquisition` with `"lcb"`, `"ei"`, or `"pi"`
- `bo_surrogate` with `"random_forest"`, `"extra_trees"`, `"gradient_boost"`, `"ridge"`, or `"ensemble"` (default: `"ridge"`)
- `bo_transfer_*` for saturation transfer BO (default: weighted mode with 2-step memory window, recency and occupancy decay; `gradient_boost` does not support transfer sample weights)
- `bo_include_failure_negatives` and `bo_failure_penalty_*` for learning from failed placements

### 3. Sequential Saturation

Saturation mode repeatedly adsorbs the current best configuration onto the evolving slab until adsorption is no longer favorable or no valid placements remain.

```python
from metalsurfer import AdsorptionConfig, prepare_slab, run_saturation
from metalsurfer.models import MultiMolSaturationRunResult

config = AdsorptionConfig(
    material_type="slab",  # "slab", "nanoparticle", or "porous"
    seed=42,
    num_conformers=6,
    num_placements=60,
)

slab = prepare_slab(
    bulk_id="mp-33",
    miller_indices=(0, 0, 1),
    config=config,
    results_dir="results_Ru001_sat",
)

# Persists to results_Ru001_sat/ when save_results=True (default), using the same
# config (per-step best slabs plus step_*_placements/ when saturation_save_all_placements is true).
campaign = run_saturation(
    slab=slab,
    molecules="molecules.csv",
    config=config,
    surface_type="Ru001_sat",
)

for entry in campaign.runs:
    if isinstance(entry, MultiMolSaturationRunResult):
        print(entry.molecules, entry.n_molecules_at_saturation)
    else:
        print(entry.molecule, entry.n_molecules_at_saturation)
```

Important saturation behaviors:

- **Prep vs adsorption relaxation:** `slab_relaxation_mode` controls ASE equilibration during `prepare_slab` only. During placements, `relax_top_layer=False` freezes the post-prep substrate (`base_slab_for_frozen`, e.g. `clean_slab_Au20_*` after adatom deposition on the Au defect workflow). Compare optimized structures to that reference, not to pre-adatom `clean_slab` files.
- Auto-resize is only allowed on the first adsorption step; if the substrate is repeated in-plane, the freeze reference is expanded to cover every repeated in-plane substrate tile.
- When `bo_enabled=True`, the saturation loop can reuse prior-step BO observations through the `bo_transfer_*` settings.
- When `multi_molecule_saturation=True` and multiple molecules are provided (in-memory list or CSV), the workflow switches to competitive saturation, where molecules compete for each step and the best overall adsorption wins.
- Competitive saturation also supports `bo_enabled=True`. In that mode, each adsorbate trains and carries forward its own BO state independently; BO observations are not shared across adsorbates.
- By default, `saturation_save_all_placements=True` writes every validated placement per step under `xyz_structures/.../step_{NNN}_placements/`, plus `saturation_placements_detailed.csv`. Matching `vasp_inputs/...` trees are written only when `write_vasp_inputs=True`. Set `saturation_save_all_placements=False` to persist only the per-step best structures (smaller disk use).
- By default, `saturation_discard_topology_rearrangements=True` re-checks the full adsorbate pool on each candidate **before** choosing the step winner: adsorbates must form the expected number of connected fragments (connectivity-only guard). This catches inter-adsorbate coupling or unexpected splitting that per-placement filtering can miss while allowing strong adsorbate-material interactions that preserve adsorbate connectivity. Set `False` to rank only by `E_ads`; the guard is also skipped when `skip_topology_check=True`.
- Contributor test markers (`gpu`, `slow`): see the [development guide](https://metalsurfer.readthedocs.io/en/latest/guides/development.html).

### Surface setup and modifiers

Use [`prepare_slab`](https://metalsurfer.readthedocs.io/en/latest/api/surface_prep.html) to build or load a slab and optionally apply alloy substitution and adatom deposition in one call:

```python
from metalsurfer import AdsorptionConfig, prepare_slab

config = AdsorptionConfig(material_type="slab", seed=42)

slab = prepare_slab(
    bulk_id="mp-33",
    miller_indices=(0, 0, 1),
    alloy_host="Ru",
    alloy_guest="Cu",
    alloy_fraction=0.25,
    adatom_symbol="Sn",
    adatom_coverage=0.20,
    config=config,
    results_dir="results_Ru001",
    adatom_relaxation_mode="ionic_only",  # optional: full clean slab once, ionic-only after adatoms
)
```

See the [Surface Engineering guide](https://metalsurfer.readthedocs.io/en/latest/guides/surface_engineering.html) for prep relaxation presets and substrate freeze behavior during adsorption.

For adatoms on an existing slab (e.g. frozen-base workflows), pass ``slab=`` after building the base:

```python
base_slab = prepare_slab(bulk_id="mp-33", miller_indices=(0, 0, 1), config=config, results_dir=results_dir)
slab = prepare_slab(
    slab=base_slab,
    adatom_symbol="Sn",
    adatom_coverage=0.10,
    config=config,
    results_dir=results_dir,
)
```

Lower-level helpers (``create_slab_from_bulk``, ``substitute_alloy``, ``deposit_adatoms``) remain available for custom research loops.

`AdsorptionConfig.material_type` must be chosen explicitly:

- `"slab"`: in-plane periodic surfaces.
- `"nanoparticle"`: non-periodic clusters.
- `"porous"`: fully periodic porous frameworks.

This choice affects site generation, adsorption validation, and distance handling throughout the workflow.

## What the core pipeline does

Across all run modes, the library follows the same structure:

1. Build or accept a surface structure.
2. Generate and deduplicate molecular conformers.
3. Enumerate deterministic `PlacementSpec` candidates over conformer, site, orientation, tilt, azimuth, and height.
4. Materialize placements into full adsorbate-slab structures.
5. Relax structures with the configured MLIP backend.
6. Validate adsorption geometry and filter decomposed, desorbed, or duplicate structures.
7. Rank surviving structures and persist structures, CSV summaries, and metadata.

Placement generation is Voronoi-based and works across slabs, nanoparticles, and porous materials. Bayesian mode changes candidate selection, not the downstream physics or filtering stack.

## Results and persistence

The output directory is `results_{surface_type}`. Depending on run mode, the library may write:

- `adsorption_energies_detailed.csv`
- `adsorption_energy_summary.csv`
- `saturation_details.csv`
- `saturation_placements_detailed.csv` (saturation runs when `saturation_save_all_placements` is true: one row per step × placement with paths and descriptor context)
- `saturation_summary.csv`
- `run_metadata.json`
- `xyz_structures/...`
- `vasp_inputs/...` (only when `write_vasp_inputs=True`)

Campaign APIs save XYZ structures and summary tables by default (`run_saturation` / `run_saturation_bo` call `save_saturation_results(..., config=config)` so placement-tree output follows `saturation_save_all_placements` and the rest of `AdsorptionConfig`). VASP bundles require `write_vasp_inputs=True` on `AdsorptionConfig`. Workflow APIs return typed results and can be paired with `save_summary_results(...)`, `save_saturation_results(...)`, `save_multi_mol_saturation_results(...)`, `write_run_metadata(...)`, and `write_run_settings(...)` for explicit persistence control (for example after `save_results=False` or custom paths).

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
```

CI parity, coverage gates, GPU/slow test jobs: [development guide](https://metalsurfer.readthedocs.io/en/latest/guides/development.html). Architecture: [CORE_SYSTEM_EXPLANATION.md](CORE_SYSTEM_EXPLANATION.md).
