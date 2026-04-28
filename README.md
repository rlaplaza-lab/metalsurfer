# Metalsurfer

![Metalsurfer Logo](docs/_static/logo_metalsurfer.svg)

Library for adsorption-energy screening on arbitrary surfaces (slabs, nanoparticles, and periodic porous frameworks).

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

## Quick Examples

Two simple examples are provided in `scripts/` to demonstrate basic usage:

### H2 Adsorption on Pt Nanocluster
```bash
# 12-atom Pt cluster with H2 adsorption
python scripts/h2_pt12_binding_energy.py
```

### CO2 Adsorption in MOF
```bash
# Simple MOF structure with CO2 adsorption
python scripts/co2_mof_binding_energy.py
```

Both examples:
- Use pure ASE for receptor preparation
- Limit to 5 placements for quick testing
- Demonstrate different material types (`nanoparticle` vs `porous`)
- Produce XYZ structures, POSCAR files, and CSV results

## Python API

The library exposes four high-level entry points:

| Function | Role |
| -------- | ---- |
| `run_adsorption` | Standard screening: enumerate placements, relax, filter, rank. |
| `run_adsorption_bo` | Same pipeline with Bayesian optimization over placement candidates. |
| `run_saturation` | Sequential saturation: repeated adsorption onto an evolving slab. |
| `run_saturation_bo` | Saturation with BO-guided placement selection. |

Each accepts either an in-memory `list[tuple[str, str]]` of `(smiles, name)` pairs or a path to a SMILES CSV as `molecules`.

### ASE Atoms input

All run entry points accept a plain ASE `Atoms` object directly. This is the recommended path for user scripts.

Example:

```python
from ase.build import fcc111

from metalsurfer import AdsorptionConfig, run_adsorption

slab_atoms = fcc111("Ru", size=(3, 3, 3), vacuum=12.0)

config = AdsorptionConfig(
    material_type="slab",  # "slab", "nanoparticle", or "porous"
    seed=42
)
result = run_adsorption(
    slab=slab_atoms,
    molecules=[("O", "water")],
    config=config,
    surface_type="ru111_from_ase_atoms",
)
```

`create_slab_from_atoms(...)` wraps a bare `Atoms` in a `SlabContainer` with default metadata; passing `Atoms` directly to run entry points is preferred when you do not need the container.

### 1. Standard Screening

Use the campaign API when your driving script already has the molecule list in memory and you want a typed `BindingCampaignResult` back.

```python
from metalsurfer import AdsorptionConfig, create_slab_from_bulk, run_adsorption

config = AdsorptionConfig(
    material_type="slab",  # "slab", "nanoparticle", or "porous"
    seed=42,
    num_conformers=8,
    num_placements=80,
)

slab = create_slab_from_bulk(
    bulk_id="mp-33",
    miller_indices=(0, 0, 1),
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
from metalsurfer import AdsorptionConfig, create_slab_from_bulk, run_adsorption

config = AdsorptionConfig(
    material_type="slab",  # "slab", "nanoparticle", or "porous"
    seed=42
)
slab = create_slab_from_bulk(bulk_id="mp-33", miller_indices=(0, 0, 1))

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
    create_slab_from_bulk,
    run_adsorption_bo,
)

config = AdsorptionConfig(
    material_type="slab",  # "slab", "nanoparticle", or "porous"
    seed=42,
    bo_enabled=True,
    bo_initial_random=20,
    bo_batch_size=10,
    bo_total_budget=60,
    bo_acquisition="lcb",
    bo_surrogate="random_forest",
)

slab = create_slab_from_bulk(bulk_id="mp-33", miller_indices=(0, 0, 1))

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

- `bo_initial_random`, `bo_batch_size`, `bo_total_budget`
- `bo_acquisition` with `"lcb"`, `"ei"`, or `"pi"`
- `bo_surrogate` with `"random_forest"`, `"extra_trees"`, `"gradient_boost"`, or `"ridge"`
- `bo_transfer_*` for transfer-enabled saturation runs (per-sample weights are used only for tree surrogates; `gradient_boost` and `ridge` fits drop transfer weights)
- `bo_include_failure_negatives` and `bo_failure_penalty_*` for learning from failed placements

### 3. Sequential Saturation

Saturation mode repeatedly adsorbs the current best configuration onto the evolving slab until adsorption is no longer favorable or no valid placements remain.

```python
from metalsurfer import AdsorptionConfig, create_slab_from_bulk, run_saturation
from metalsurfer.io_results import save_saturation_results

config = AdsorptionConfig(
    material_type="slab",  # "slab", "nanoparticle", or "porous"
    seed=42,
    num_conformers=6,
    num_placements=60,
)

slab = create_slab_from_bulk(bulk_id="mp-33", miller_indices=(0, 0, 1))

saturation_results = run_saturation(
    slab=slab,
    molecules="molecules.csv",
    config=config,
    surface_type="Ru001_sat",
)

save_saturation_results(saturation_results, surface_type="Ru001_sat", config=config)

for result in saturation_results:
    print(result.molecule, result.n_molecules_at_saturation)
```

Important saturation behaviors:

- Auto-resize is only allowed on the first adsorption step so later steps keep the evolved slab footprint.
- When `bo_enabled=True`, the saturation loop can reuse prior-step BO observations through the `bo_transfer_*` settings.
- When `multi_molecule_saturation=True` and the CSV contains multiple molecules, the workflow switches to competitive saturation, where molecules compete for each step and the best overall adsorption wins.
- Competitive saturation also supports `bo_enabled=True`. In that mode, each adsorbate trains and carries forward its own BO state independently; BO observations are not shared across adsorbates.
- Recommended validation split: keep local validation focused on mocked or lightweight saturation tests, and reserve full-stack BO competitive saturation checks for a dedicated `gpu` + `slow` integration test in a GPU-capable environment.

### Surface setup and modifiers

The surface can be prepared programmatically before any run mode. Alloy substitution is applied first, then adatom deposition if both are used.

`calculator` is **optional** for both `substitute_alloy(...)` and `deposit_adatoms(...)`:

- Without a calculator: a valid modified slab is still created (fast structural modification).
- With a calculator: random variants are energy-scored and the lowest-energy variant is selected.
- For `substitute_alloy(...)`, optional post-selection relaxation only runs when both `relax=True` and `calculator` is provided.

Fast structural modification (no energy ranking):

```python
from metalsurfer.surface_prep import (
    create_slab_from_bulk,
    deposit_adatoms,
    substitute_alloy,
)

slab = create_slab_from_bulk(bulk_id="mp-33", miller_indices=(0, 0, 1))

slab = substitute_alloy(
    slab,
    host_symbol="Ru",
    guest_symbol="Cu",
    guest_fraction=0.25,
)

slab = deposit_adatoms(
    slab,
    adatom_symbol="Sn",
    coverage_fraction=0.20,
)
```

Energy-ranked variant selection (recommended when preparing a realistic modified surface):

```python
from metalsurfer import AdsorptionConfig, setup_single_model
from metalsurfer.surface_prep import (
    create_slab_from_bulk,
    deposit_adatoms,
    substitute_alloy,
)

config = AdsorptionConfig(
    material_type="slab"  # "slab", "nanoparticle", or "porous"
)
slab = create_slab_from_bulk(bulk_id="mp-33", miller_indices=(0, 0, 1))
calculator, _ = setup_single_model(config.model_name, config.device)

slab = substitute_alloy(
    slab,
    host_symbol="Ru",
    guest_symbol="Cu",
    guest_fraction=0.25,
    calculator=calculator,
    config=config,
)

slab = deposit_adatoms(
    slab,
    adatom_symbol="Sn",
    coverage_fraction=0.20,
    calculator=calculator,
    config=config,
)
```

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
- `saturation_summary.csv`
- `run_metadata.json`
- `xyz_structures/...`
- `vasp_inputs/...`

Campaign APIs save structures and summary tables by default. Workflow APIs return typed results and can be paired with `save_summary_results(...)`, `save_saturation_results(...)`, `save_multi_mol_saturation_results(...)`, `write_run_metadata(...)`, and `write_run_settings(...)` for explicit persistence control.

## Development

Commands below mirror the [GitHub Actions](.github/workflows/ci.yml) workflow: lint, fast tests with coverage, then optional extra test modules.

```bash
ruff check .
ruff format --check .
python -m pytest tests/ -m "not dependency_behavior and not mlip and not gpu and not slow" \
  --cov=src/metalsurfer --cov-report=term-missing --tb=short -v
coverage report --fail-under=74
python -m pytest tests/test_dependency_behavior.py -v --tb=short
python -m pytest tests/test_integration_seeded.py -v --tb=short
```

**Placement reproducibility:** `enumerate_placement_specs` uses `AdsorptionConfig.seed` when no explicit `seed` is passed. When the combinatorial placement grid is larger than `n_desired`, candidates are subsampled with that seed. For low-level experiments, `metalsurfer.placement.policy.build_batch_placement_specs` accepts an integer `seed`; the default `PLACEMENT_GRID_COUNT_SEED` keeps `max_batch_placement_specs` consistent with a full uncapped enumeration count.

**GPU and MLIP integration tests** (heavy TorchSim/FairChem workloads) are easiest to run in separate processes:

```bash
./scripts/run_gpu_tests.sh
```

Optional interpreter:

```bash
bash scripts/run_gpu_tests.sh "$(command -v python)"
```

All tests marked `slow` (often CUDA-dependent; skipped if the stack or device is missing):

```bash
python -m pytest tests/ -m slow --tb=short -v
```

Architecture and design rationale: [CORE_SYSTEM_EXPLANATION.md](CORE_SYSTEM_EXPLANATION.md).
