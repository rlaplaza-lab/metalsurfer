# Examples

Runnable demos from the project root (after `pip install -e ".[mlip]"`).

| Script | Description |
|--------|-------------|
| `h2_pt12_binding_energy.py` | H₂ on a Pt₁₂ nanoparticle |
| `co2_mof_binding_energy.py` | CO₂ in a MOF (porous) |
| `ethene_ru_slab_binding_energy.py` | Ethene on Ru(0001) |
| `bipyridine_au111_defects_saturation_raw.py` | Saturation on defected Au(111) with fixed substrate (`relax_top_layer=False`) |

The bipyridine workflow uses `prepare_slab` with `adatom_relaxation_mode="ionic_only"`.
During saturation, compare relaxed structures to `clean_slab_Au20_*` in the results
directory (post-adatom substrate), not `clean_slab_*` from before adatom deposition.

For large HPC runs, use the matching script under `scripts/`.

## Bipyridine benchmark dataset

Pre-synced saturation outputs (non-BO, `save_benchmark_dataset=True`) live under
`examples/results_bipyridine_au111_defects_saturation_raw/`. Detailed CSVs now
include quaternion and `z_fraction` columns from `PlacementDescriptor.to_row`;
older exports are backfilled from `ml_dataset.csv` in the same directory when
running the benchmark. If missing, sync CSVs from agustina:

```bash
mkdir -p examples/results_bipyridine_au111_defects_saturation_raw
rsync -avz \
  rlaplaza@agustina:/fs/agustina/rlaplaza/000_metalsurfer_scripts/results_bipyridine_au111_defects_saturation_raw/*.csv \
  rlaplaza@agustina:/fs/agustina/rlaplaza/000_metalsurfer_scripts/results_bipyridine_au111_defects_saturation_raw/run_metadata.json \
  examples/results_bipyridine_au111_defects_saturation_raw/
```

Offline BO benchmark (no GPU; compares surrogate/acquisition configs against the
pool oracle, with default `AdsorptionConfig` settings highlighted). Includes
tree models, ridge, a Matern GP (length scale `sqrt(n_features)`), and an
ensemble of RF + extra trees + ridge + GP:

```bash
python scripts/benchmark_bo_models.py \
  --data-dir examples/results_bipyridine_au111_defects_saturation_raw \
  --surface-type bipyridine_au111_defects_saturation_raw \
  --smiles "n1ccccc1-c2ccccn2" \
  --seeds 10 \
  --out benchmark_bo_results_bipyridine.csv
```
