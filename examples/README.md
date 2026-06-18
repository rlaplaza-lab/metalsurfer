# Examples

Runnable demos from the project root (after `pip install -e ".[mlip]"`).

| Script | Description |
|--------|-------------|
| `h2_pt12_binding_energy.py` | H₂ on a Pt₁₂ nanoparticle (`prepare_substrate`) |
| `co2_mof_binding_energy.py` | CO₂ in a MOF (porous; `prepare_substrate`) |
| `ethene_ru_slab_binding_energy.py` | Ethene on Ru(0001) (`prepare_substrate`) |
| `bipyridine_au111_defects_saturation_raw.py` | Saturation on defected Au(111) with rigid substrate (default prep freeze) |
| `camphor_cu111_binding_energy.py` | (1S)-camphor on Cu(111) vs Järvi et al. BOSS benchmark (BO, 15GB GPU) |

Demos set explicit small `num_placements` for quick runs. For production screening, omit
`num_placements` (and BO batch fields) to autotune to GPU parallel capacity via TorchSim
memory probing at workflow start.

`prepare_substrate` equilibrates substrate ionic positions by default (`slab_relaxation_mode="ionic_only"`) and freezes the entire substrate during adsorption by default (`relax_top_layer=False` prep kwarg). Placement relaxation honors the ASE `FixAtoms` on the returned substrate. Set `relax_top_layer=True` on `prepare_substrate` when the top surface layer should relax with the adsorbate.

The bipyridine workflow uses `prepare_substrate` with `adatom_relaxation_mode="ionic_only"`.
During saturation, compare relaxed structures to `clean_slab_Au20_*` in the results
directory (post-adatom substrate), not `clean_slab_*` from before adatom deposition.

### Camphor / Cu(111)

Revisits Järvi et al. ([Beilstein J. Nanotechnol. 2020](https://doi.org/10.3762/bjnano.11.140)):
eight DFT local minima from BOSS. Downloads the reference PES dataset from
[Zenodo 10.5281/zenodo.4680467](https://doi.org/10.5281/zenodo.4680467) into
gitignored `examples/camphor_cu111/`, uses the paper's 192-atom Cu(111) slab from
NOMAD, and runs a 25-batch BO placement search (~300+ MLIP evals on a 15GB GPU).
MLIP energies are compared qualitatively to the paper's DFT landscape (not absolute eV).

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python examples/camphor_cu111_binding_energy.py
python examples/camphor_cu111_binding_energy.py --compare-geometries  # RMSD vs NOMAD DFT
python examples/camphor_cu111_binding_energy.py --compare-geometries --export-overlays  # figure XYZ pairs
```

DFT reference geometries are downloaded from
[NOMAD 10.17172/NOMAD/2021.04.12-1](https://doi.org/10.17172/NOMAD/2021.04.12-1)
into `examples/camphor_cu111/nomad_references/` and compared via adsorbate RMSD
(permutation-aware Kabsch) plus Ox/Hy binding-mode labels.

For large HPC runs, use the matching script under `scripts/`. BO scripts that previously
set `bo_total_budget` as a total evaluation count should now set it to the number of
**acquisition batches** after the initial random batch
(e.g. `(300 - initial) // batch` to preserve a 300-eval budget with fixed batch sizes).
