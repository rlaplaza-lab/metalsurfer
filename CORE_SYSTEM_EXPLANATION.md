# Metalsurfer: core system

Short mental model for developers. Overlap with Sphinx guides is fine; this
file stays the one-page map of what the code does and where complexity lives.

Install / demos: [`README.md`](README.md), [`examples/`](examples/). Field knobs:
[configuration](https://metalsurfer.readthedocs.io/en/latest/guides/configuration.html).
YAML campaigns:
[yaml_campaigns](https://metalsurfer.readthedocs.io/en/latest/guides/yaml_campaigns.html).
Full mechanics (sites, placement, TorchSim, BO, validation, AdsorbML/BOSS):
[architecture](https://metalsurfer.readthedocs.io/en/latest/guides/architecture.html).

## What it does

Prep a substrate → enumerate adsorbate placements → **batch-relax** them with an
MLIP (TorchSim) → filter survivors → optionally grow **coverage** by committing
the best adsorbate and repeating.

Adsorption energy is always:

```text
E_ads = E_adslab - E_slab - E_molecule
```

## How to enter

| Want                         | Call                                                       |
|------------------------------|------------------------------------------------------------|
| Screen many molecules        | `run_adsorption` / `run_adsorption_bo`                     |
| Grow coverage on one surface | `run_saturation` / `run_saturation_bo`                     |
| YAML campaign                | `load_campaign_yaml` + `run_campaign`                      |
| Build/load the surface first | `prepare_substrate` (also `surface_prep.prepare_substrate`) |

Prefer `run_*_bo` when you want Bayesian placement selection. BO mode is chosen
by the entry point (`run_adsorption_bo` / `run_saturation_bo`) or YAML
`campaign: adsorption_bo` / `saturation_bo`; config only holds nested BO
hyperparameters (`AdsorptionConfig.bo` / `bo.transfer`). Flat `bo_*` constructor
and YAML keys are rejected.

YAML is a convenience dispatch layer (`bulk_id` / `slab_file`, inline molecules,
`skip_existing` only on `run_campaign`)—not a full substitute for `prepare_substrate`
+ `run_*` (custom ASE `slab=`, molecule CSVs, extra kwargs, post-run checks).

Inputs: `SlabContainer` or ASE `Atoms`, molecules (list or CSV),
`AdsorptionConfig`, `surface_type` (results folder label only). Prep freeze
policy is set in `prepare_substrate`, not on campaign kwargs.

## Pipeline (one molecule)

```mermaid
flowchart LR
  prep[Prep] --> ref[References]
  ref --> conf[Conformers]
  conf --> place[Place]
  place --> relax[Batch_relax]
  relax --> filter[Filter]
  filter --> out[Results]
```

Saturation repeats place→relax→filter, commits the best `E_ads < 0` structure
onto the slab, refreshes `E_slab`, and stops when the next step is endothermic
or no valid placements remain.

## Three dual models (the load-bearing complexity)

Almost every packing/saturation bug traces to one of these:

1. **Substrate vs covered slab**  
   Site catalogs and substrate distance checks use a **substrate-only** view
   (`slab_for_sites`: prefix of length `len(base_slab_for_frozen)`). Relaxation
   and adsorbate–adsorbate separation use the **full** slab (prior adsorbates
   appended as a suffix).

2. **Spec vs pose**  
   `PlacementSpec` is a discrete enumeration slot (site index, tilt, …).
   Materialization yields absolute geometry (`PlacementDescriptor` / pose). BO
   and ML features see **absolute xyz + quaternion only**, not site IDs or
   orientation labels. CSV exports default to those pose features (+ labels);
   set `export_placement_provenance=True` for `initial_*` site/orientation
   provenance (pre-relax intent, not final binding mode).

3. **Prep freeze vs adsorption freeze**  
   Prep may equilibrate the bare surface (`slab_relaxation_mode`). During
   adsorbate relaxation, frozen substrate atoms come from ASE `FixAtoms`
   attached at prep (`frozen_indices_from_constraints`). Saturation keeps the
   original substrate length so new adsorbate atoms can move.

## Module map (where to look)

| Concern                              | Package / module |
|--------------------------------------|------------------|
| Campaigns / YAML                     | `campaigns.py`, `campaign_schema.py` |
| Config / typed results               | `config.py` (`AdsorptionConfig`, nested `BOConfig` / `BOTransferConfig`), `models.py` |
| Substrate prep / freeze              | `surface_prep/` |
| Sites + placement                    | `placement/` (`generators` orchestration; `site_*`, `orientation`, `pose`, `geometry`, `policy`, `occupancy`, `dissociative`) |
| Per-molecule / saturation / BO loops | `workflow/` (`core`, `saturation`, `bayesian`, `placement_fill`, `reference`, `shared`; `MoleculeScreenOutcome`) |
| Batched MLIP relax                   | `optimization.py` |
| Post-relax filters                   | `filters.py` |
| Dataset / surrogates                 | `ml/` |
| Persistence                          | `io_results.py` |

## Design heuristics

- Sample **many** placements per GPU wave; binding energy is the best survivor
  after filters—not a single hand-picked pose.
- Clearance-aware height (slab/NP): after orientation, lift the COM so the
  closest adsorbate atom sits at the intended `z_offset` (avoids alkyl/H dig-in
  from COM-centered poses; skipped for porous).
- Saturation stops when the next `E_ads ≥ 0` (coverage proxy), not at a fixed ML.
- Under coverage, prune occupied sites before the orientation grid; clash with
  prior adsorbates is a first-class failure (`adsorbate_overlap`), with optional
  XY recovery.
- Symmetry-reduced sites until coverage breaks symmetry vs the clean reference.
- GPU-first: leave `num_placements` / `bo.initial_random` / `bo.batch_size` as
  `None` so TorchSim autotunes parallel capacity.

## Where detail lives

| Topic | Doc |
|-------|-----|
| API layers, sites, placement, TorchSim, BO, validation, materials, outputs, AdsorbML/BOSS | [Architecture](https://metalsurfer.readthedocs.io/en/latest/guides/architecture.html) |
| `AdsorptionConfig` recipes and common mistakes | [Configuration](https://metalsurfer.readthedocs.io/en/latest/guides/configuration.html) |
| YAML schema, limitations, demo files | [YAML campaigns](https://metalsurfer.readthedocs.io/en/latest/guides/yaml_campaigns.html) |
| Prep, freeze, resize, adatoms | [Surface engineering](https://metalsurfer.readthedocs.io/en/latest/guides/surface_engineering.html) |
| Field reference | [API: config](https://metalsurfer.readthedocs.io/en/latest/api/config.html), [API: campaigns](https://metalsurfer.readthedocs.io/en/latest/api/campaigns.html), [API: surface_prep](https://metalsurfer.readthedocs.io/en/latest/api/surface_prep.html) |
| Tests / CI | [Development](https://metalsurfer.readthedocs.io/en/latest/guides/development.html) |

Python **3.12+**. Core deps: `numpy`, `ase`, `pandas`, `rdkit`, `scipy`,
`scikit-learn`, `spglib`. Optional MLIP stack: `torch`, `torch-sim-atomistic`,
FairChem/UMA.
