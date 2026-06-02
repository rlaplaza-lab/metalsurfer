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
