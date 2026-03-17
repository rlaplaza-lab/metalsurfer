# metalsurfer

Generic adsorption-energy screening on arbitrary surfaces.

## Install

```bash
pip install -e .
```

## Usage

### CLI: base slab

Screen molecules on a clean surface (e.g. Ru(001)):

```bash
adsorption-screening --smiles-file molecules.csv --bulk-id mp-33
```

### CLI: alloyed slab

Replace 25 % of the Ru atoms with Cu before screening:

```bash
adsorption-screening \
    --smiles-file molecules.csv \
    --bulk-id mp-33 \
    --alloy-host Ru --alloy-guest Cu --alloy-fraction 0.25
```

### CLI: adatom-decorated slab

Deposit Sn adatoms at 20 % of the hollow sites:

```bash
adsorption-screening \
    --smiles-file molecules.csv \
    --bulk-id mp-33 \
    --adatom-symbol Sn --adatom-coverage 0.2
```

### CLI: combined (alloy + adatom)

Both modifications can be combined in one run. Alloying is applied first,
then adatom deposition:

```bash
adsorption-screening \
    --smiles-file molecules.csv \
    --bulk-id mp-33 \
    --alloy-host Ru --alloy-guest Cu --alloy-fraction 0.25 \
    --adatom-symbol Sn --adatom-coverage 0.2
```

### CLI: sequential saturation

Add molecules one at a time until the slab is saturated (no negative
adsorption energy found). For each molecule in the SMILES file, runs an
independent saturation loop starting from the clean slab:

```bash
adsorption-screening \
    --smiles-file molecules.csv \
    --bulk-id mp-33 \
    --saturation
```

Outputs include `saturation_summary.csv`, `saturation_details.csv`, and
per-step XYZ structures in `xyz_structures/{molecule}_saturation/`.

### CLI: Bayesian screening

Use surrogate-guided placement selection (Random Forest + UCB) to explore
fewer placements while targeting low adsorption energies:

```bash
adsorption-screening \
    --smiles-file molecules.csv \
    --bulk-id mp-33 \
    --bayesian
```

Optional BO knobs: `--bo-initial-random`, `--bo-batch-size`, `--bo-total-budget`.

### Python API

The same workflow is available as a library. Below is a minimal example
that builds a slab, optionally modifies it, and runs the screening:

```python
from metalsurfer import (
    AdsorptionConfig,
    create_slab_from_bulk,
    substitute_alloy,
    deposit_adatoms,
    run_screening,
    setup_calculator,
)

config = AdsorptionConfig(seed=42, num_conformers=5, num_placements=50)

# 1. Build the base slab
slab = create_slab_from_bulk(bulk_id="mp-33", miller_indices=(0, 0, 1))

# 2. (optional) Alloy substitution — replace 25 % Ru with Cu
calc = setup_calculator(config.model_name, config.device)
slab = substitute_alloy(
    slab, host_symbol="Ru", guest_symbol="Cu",
    guest_fraction=0.25, calculator=calc, config=config,
)

# 3. (optional) Adatom deposition — place Sn at 20 % of hollow sites
slab = deposit_adatoms(
    slab, adatom_symbol="Sn", coverage_fraction=0.2,
    calculator=calc, config=config,
)

# 4. Run adsorption screening
results = run_screening(slab, smiles_file="molecules.csv", config=config)
```

Sequential saturation (add molecules until E_ads >= 0):

```python
from metalsurfer import (
    AdsorptionConfig,
    create_slab_from_bulk,
    run_saturation_screening,
    setup_calculator,
)
from metalsurfer.io_results import save_saturation_results, setup_directories

config = AdsorptionConfig(seed=42, num_conformers=5, num_placements=50, saturation=True)
slab = create_slab_from_bulk(bulk_id="mp-33", miller_indices=(0, 0, 1))

saturation_results = run_saturation_screening(
    slab, smiles_file="molecules.csv", config=config, surface_type="Ru001"
)
setup_directories(["Ru001"])
save_saturation_results(saturation_results, surface_type="Ru001")

for sr in saturation_results:
    print(f"{sr.molecule}: {sr.n_molecules_at_saturation} molecules at saturation")
```

Bayesian screening (surrogate-guided placement selection): use
`run_screening_bayesian` with `AdsorptionConfig(bo_enabled=True, ...)`.
Key options: `bo_initial_random`, `bo_batch_size`, `bo_total_budget`,
`bo_ucb_kappa` (defaults in `AdsorptionConfig`).

Both `guest_fraction` and `coverage_fraction` must be in `[0, 1]`;
a value of `0` is treated as a no-op (the unmodified slab is returned).

See `tests/` for examples.

## Development

- Tests: `pytest tests/`
- Lint/format: `ruff check . && ruff format .`

### Quick verification for slab modifiers

Run just the surface-modifier and filter tests (fast, no GPU required):

```bash
pytest tests/test_surfaces.py tests/test_filters.py -v
```

### Quick verification for saturation

Run the saturation unit tests (no MLIP/GPU required):

```bash
pytest tests/test_saturation.py -v
```

To run the full saturation integration test (real MLIP on GPU):

```bash
pytest tests/test_saturation.py -m "mlip and gpu" -v
```

Use `CUDA_VISIBLE_DEVICES=""` if you see CUDA OOM errors when running with GPU visible.
