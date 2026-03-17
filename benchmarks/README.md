# Benchmarks

## TorchSim optimizer/batching benchmark

`benchmark_torchsim.py` measures the impact of TorchSim settings on two
workloads that mirror a real screening run:

1. **Isolated-molecule optimisation** -- small systems (conformers).
2. **Slab+adsorbate optimisation** -- larger systems with frozen sub-surface
   constraints and ragged adsorbate sizes.

```bash
python benchmarks/benchmark_torchsim.py                                       # GPU (default)
python benchmarks/benchmark_torchsim.py --device cpu                           # CPU fallback
python benchmarks/benchmark_torchsim.py --n-conformers 16 --n-placements 20 --n-steps 100  # larger workload
```

It compares:
- **Sequential** vs **autobatched** optimisation
- **FIRE** vs **L-BFGS** optimiser
- **`autobatcher_max_memory_padding`** 0.25, 0.5, 0.8

Output: `benchmarks/torchsim_benchmark.csv` with columns for scenario, timing, convergence count, and GPU memory.

## Bayesian optimization benchmark

This section documents the BO benchmark used to compare surrogate models and acquisition functions and to inform default settings.

## Data

1. **CO2 on graphene monolayer (100 placements)**  
   Run the data collection script to generate `results_co2_graphene/adsorption_energies_detailed.csv`:

   ```bash
   conda run -n metalsurfer python scripts/co2_substituted_graphene_100placements.py
   ```

   If the GPU runs out of memory, use CPU (slower):

   ```bash
   python scripts/co2_substituted_graphene_100placements.py --device cpu
   ```

   The slab is a graphene monolayer (graphite mp-48, (0,0,1), one layer) with 10% C→N substitution so placement quality varies and the BO problem is non-trivial.

2. **Propane on Pt(111) + Ni adatoms (120 placements)**  
   Run the data collection script to generate `results_propane_pt111_ni/adsorption_energies_detailed.csv`:

   ```bash
   conda run -n metalsurfer python scripts/propane_pt111_ni_100placements.py
   ```

   If the GPU runs out of memory, use CPU (slower):

   ```bash
   python scripts/propane_pt111_ni_100placements.py --device cpu
   ```

   The slab is a 3×3 Pt(111) surface (fcc Pt mp-126) with ~10% Ni adatoms at hollow sites. Propane is a floppy C3 adsorbate (20 conformers) so placement diversity is high.

3. **Benchmark (fixed batch size 10)**  
   Run the offline BO simulation with fixed batches of 10:

   ```bash
   python scripts/benchmark_bo_models.py --data-dir results_co2_graphene --out benchmark_bo_results.csv --seeds 5
   ```

   For the propane/Pt dataset, pass `--surface-type` and `--smiles` so the feature columns are labelled correctly:

   ```bash
   python scripts/benchmark_bo_models.py --data-dir results_propane_pt111_ni --surface-type propane_pt111_ni --smiles CCC --out benchmark_bo_results_propane_pt111_ni.csv --seeds 5
   ```

   Without real data, you can test the pipeline with synthetic data:

   ```bash
   python scripts/benchmark_bo_models.py --synthetic --out benchmark_bo_results_synthetic.csv
   ```

## Defaults

Current BO defaults in `AdsorptionConfig`:

- `bo_acquisition`: `"lcb"` (alternatives: `"ei"`, `"pi"`)
- `bo_ucb_kappa`: `1.96`
- `bo_initial_random`: `20`
- `bo_batch_size`: `10`
- `bo_total_budget`: `100`

After running the benchmark on real data (CO2/graphene and/or propane/Pt), you can update these in `src/metalsurfer/config.py` based on which (surrogate, acquisition, kappa) gives the best mean final E_ads or fastest convergence across both datasets.
