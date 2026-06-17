# Examples

Runnable demos from the project root (after `pip install -e ".[mlip]"`).

| Script | Description |
|--------|-------------|
| `h2_pt12_binding_energy.py` | H₂ on a Pt₁₂ nanoparticle |
| `co2_mof_binding_energy.py` | CO₂ in a MOF (porous) |
| `ethene_ru_slab_binding_energy.py` | Ethene on Ru(0001) |
| `bipyridine_au111_defects_saturation_raw.py` | Saturation on defected Au(111) with fixed substrate (`relax_top_layer=False`) |
| `camphor_cu111_binding_energy.py` | (1S)-camphor on Cu(111) vs Järvi et al. BOSS benchmark (BO, 15GB GPU) |
| `c60_tio2_anatase101_binding_energy.py` | C60 on anatase TiO₂(101) vs Todorović et al. Zenodo benchmark (BO, 15GB GPU) |

Demos set explicit small `num_placements` for quick runs. For production screening, omit
`num_placements` (and BO batch fields) to autotune to GPU parallel capacity via TorchSim
memory probing at workflow start.

The bipyridine workflow uses `prepare_slab` with `adatom_relaxation_mode="ionic_only"`.
During saturation, compare relaxed structures to `clean_slab_Au20_*` in the results
directory (post-adatom substrate), not `clean_slab_*` from before adatom deposition.

### Camphor / Cu(111)

Revisits Järvi et al. ([Beilstein J. Nanotechnol. 2020](https://doi.org/10.3762/bjnano.11.140)):
eight DFT local minima from BOSS. Downloads the reference PES dataset from
[Zenodo 10.5281/zenodo.4680467](https://doi.org/10.5281/zenodo.4680467) into
gitignored `examples/camphor_cu111/`, builds Cu(111) from mp-30 with in-plane
auto-resize, and runs BO-guided placement search. MLIP energies are compared
qualitatively to the paper's DFT landscape (not absolute eV).

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python examples/camphor_cu111_binding_energy.py
python examples/camphor_cu111_binding_energy.py --setup-only  # data/slab check only
python examples/camphor_cu111_binding_energy.py --quick      # small BO budget
python examples/camphor_cu111_binding_energy.py --production      # ~300+ BO evals on 15GB GPU
python examples/camphor_cu111_binding_energy.py --compare-geometries  # RMSD vs NOMAD DFT
python examples/camphor_cu111_binding_energy.py --production --compare-geometries
python examples/camphor_cu111_binding_energy.py --dft-slab  # paper 192-atom Cu(111) from NOMAD
python examples/camphor_cu111_binding_energy.py --compare-geometries --export-overlays  # figure XYZ pairs
```

DFT reference geometries are downloaded from
[NOMAD 10.17172/NOMAD/2021.04.12-1](https://doi.org/10.17172/NOMAD/2021.04.12-1)
into `examples/camphor_cu111/nomad_references/` and compared via adsorbate RMSD
(permutation-aware Kabsch) plus Ox/Hy binding-mode labels.

### C60 / TiO₂(101)

Downloads reference slab geometry from [Zenodo 10.5281/zenodo.2565933](https://doi.org/10.5281/zenodo.2565933)
into gitignored `examples/c60_tio2_anatase101/`, then runs BO-guided placement search.
MLIP energies are compared qualitatively to the paper's DFT landscape (not absolute eV).

```bash
python examples/c60_tio2_anatase101_binding_energy.py
python examples/c60_tio2_anatase101_binding_energy.py --setup-only  # data/slab check only
python examples/c60_tio2_anatase101_binding_energy.py --quick      # small BO budget
```

For large HPC runs, use the matching script under `scripts/`. BO scripts that previously
set `bo_total_budget` as a total evaluation count should now set it to the number of
**acquisition batches** after the initial random batch
(e.g. `(300 - initial) // batch` to preserve a 300-eval budget with fixed batch sizes).
