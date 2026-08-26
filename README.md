# Metalsurfer

[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/) [![PyPI](https://img.shields.io/pypi/v/metalsurfer.svg)](https://pypi.org/project/metalsurfer/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Metalsurfer Logo](docs/_static/logo_metalsurfer.svg)

Library for adsorption on arbitrary materials (slabs, nanoparticles, and periodic porous frameworks).

Pass any ASE `Atoms` structure after optional prep with `prepare_substrate`, supply adsorbates as SMILES, and run screening, Bayesian placement search, or sequential saturation via the four `run_*` campaign APIs.

**Documentation:** https://metalsurfer.readthedocs.io

## Features

- **Substrate-agnostic** — periodic slabs, non-periodic clusters, and fully periodic porous frameworks
- **MLIP relaxation** — TorchSim/FairChem-backed optimization (UMA default, `task_name="oc25"`)
- **Orientation-aware placement** — hybrid topology + Voronoi site detection, material-aware via `AdsorptionConfig.material_type`
- **Four campaign modes** — standard screening, Bayesian screening, sequential saturation, and BO saturation; with competitive multi-molecule and multi-placement-per-step (n-tuplet) coverage modes
- **Surface prep** — equilibration, PBC, alloy/adatom modifiers, and ASE `FixAtoms` via `prepare_substrate`
- **Reproducible workflows** — seeded conformer and placement sampling; structured CSV/XYZ output

## Install

Requires **Python 3.12 or newer**.

Core dependencies only (library import and CPU-only workflow tests):

```bash
pip install -e .
```

**Running examples, scripts, or any `run_*` campaign requires the MLIP stack:**

```bash
pip install -e ".[mlip]"
```

For TorchSim/FairChem-backed relaxation plus the developer toolchain:

```bash
pip install -e ".[mlip,dev]"
```

See the [installation guide](https://metalsurfer.readthedocs.io/en/latest/guides/quickstart.html#installation) for editable installs and documentation build extras.

## Quick start

```python
from metalsurfer import AdsorptionConfig, prepare_substrate, run_adsorption

config = AdsorptionConfig(material_type="slab", seed=42)

slab = prepare_substrate(
    bulk_id="mp-33",
    miller_indices=(0, 0, 1),
    config=config,
    results_dir="results_Ru0001",
)

result = run_adsorption(
    slab=slab,
    molecules=[("C=C", "ethene")],
    config=config,
    surface_type="Ru0001",
)

for summary in result.molecule_summaries:
    print(summary.molecule, summary.best_adsorption_energy)
```

## Workflows

| Goal | Entry point | Documentation |
|------|-------------|---------------|
| Standard screening | `run_adsorption` | [Quick start](https://metalsurfer.readthedocs.io/en/latest/guides/quickstart.html) |
| Bayesian screening | `run_adsorption_bo` | [Quick start — Bayesian](https://metalsurfer.readthedocs.io/en/latest/guides/quickstart.html#bayesian-screening) |
| Sequential saturation | `run_saturation` | [Quick start — Saturation](https://metalsurfer.readthedocs.io/en/latest/guides/quickstart.html#sequential-saturation) |
| BO saturation | `run_saturation_bo` | [Quick start — Saturation](https://metalsurfer.readthedocs.io/en/latest/guides/quickstart.html#sequential-saturation) |
| Substrate preparation | `prepare_substrate` | [Surface engineering](https://metalsurfer.readthedocs.io/en/latest/guides/surface_engineering.html) |
| Configuration options | `AdsorptionConfig` | [Field reference](https://metalsurfer.readthedocs.io/en/latest/api/config.html) · [Configuration guide](https://metalsurfer.readthedocs.io/en/latest/guides/configuration.html) |
| YAML campaign | `load_campaign_yaml` + `run_campaign` | [`scripts/campaigns/`](scripts/campaigns/); `campaign:` is `adsorption` / `adsorption_bo` / `saturation` / `saturation_bo` ([API](https://metalsurfer.readthedocs.io/en/latest/api/campaigns.html)) |

Set `material_type` (defaults to `slab`) to match your substrate: `slab`, `nanoparticle`, or `porous`. See the [configuration guide](https://metalsurfer.readthedocs.io/en/latest/guides/configuration.html) for when to use each.

Output is written under `results_{surface_type}/` (XYZ structures, CSV summaries, optional VASP inputs). See [architecture — output structure](https://metalsurfer.readthedocs.io/en/latest/guides/architecture.html#output-structure) for layout details.

## Examples

Runnable scripts in [`examples/`](examples/) (requires `pip install -e ".[mlip]"`):

| Script | `material_type` | Notes |
|--------|-----------------|-------|
| [`examples/ethene_pt12_binding_energy.py`](examples/ethene_pt12_binding_energy.py) | `nanoparticle` | Ethene on a Pt₁₂ cluster |
| [`examples/co2_mof_binding_energy.py`](examples/co2_mof_binding_energy.py) | `porous` | CO₂ in a MOF (RUBTAK01) |
| [`examples/ethene_ru_slab_binding_energy.py`](examples/ethene_ru_slab_binding_energy.py) | `slab` | Ethene on Ru(0001) |
| [`examples/h2_ru_slab_binding_energy.py`](examples/h2_ru_slab_binding_energy.py) | `slab` | H₂ dissociative adsorption (`enable_dissociative_placement` + `skip_topology_check`) |
| [`examples/water_oh_rutile_saturation.py`](examples/water_oh_rutile_saturation.py) | `slab` | Water + OH⁻ competing on rutile TiO₂(110) (multi-molecule + n-tuplet saturation) |
| [`examples/camphor_cu111_binding_energy.py`](examples/camphor_cu111_binding_energy.py) | `slab` | Bayesian placement search on literature Cu(111) slab |
| [`examples/bipyridine_au111_defects_saturation_raw.py`](examples/bipyridine_au111_defects_saturation_raw.py) | `slab` | HPC-scale saturation demo (also under `scripts/`) |

Omit `num_placements` in production to autotune to GPU parallel capacity. See [`examples/README.md`](examples/README.md) for run commands and notes.

## Development

```bash
pip install -e ".[mlip,dev]"
./scripts/run_all_tests.sh             # quick + cpu + gpu
ruff check . && ruff format --check . && mypy src/metalsurfer
python -m pytest tests/ -m quick \
  --cov=src/metalsurfer --cov-report=term-missing --tb=short -v
```

CI parity, coverage gates, and contributor test markers: [development guide](https://metalsurfer.readthedocs.io/en/latest/guides/development.html). Mental model: [`CORE_SYSTEM_EXPLANATION.md`](CORE_SYSTEM_EXPLANATION.md). Full mechanics: [architecture guide](https://metalsurfer.readthedocs.io/en/latest/guides/architecture.html).

---

MIT License — see [`LICENSE`](LICENSE).
