# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.6] - 2026-10-10

### Changed

- Competitive multi-molecule n-tuplet funds each species' pure pack before
  mixed packs (largest-remainder shares); a short ``num_placements`` leaves
  mixtures unfunded instead of raising.
- ``commit_best_joint_config`` returns
  ``(pool_best, committed, ranked_group)`` so unbound steps can still log the
  Ω-best pack.
- Docs, ``CORE_SYSTEM_EXPLANATION.md``, and
  ``examples/water_oh_rutile_saturation.py`` match pure-first composition
  funding and the ``placement/pose/`` + ``ml/bayesian/`` package layout.

### Removed

- ``workflow.composite.evaluate_composite_commit`` (use
  ``evaluate_composite_batch``).

## [0.9.5] - 2026-10-02

### Changed

- Adsorbate–adsorbate disconnect, clash radii, and BO occupancy-sigma ratios
  share one covalent-radius lookup and **fail loud** when a tabulated radius
  is missing (no silent ``min_adsorbate_separation / 2`` floor).
- Ranking for single-unit and joint commits shares one helper
  (``joint_config_ranking_energy``); BO occupancy sigma inflation is shared
  between sequential and joint BO loops.
- ``scripts/oh_pt111_ntuple_saturation.py`` renamed to
  ``scripts/oh_pt111_ntuplet_saturation.py`` (matches n-tuplet naming).

### Removed

- `placement.occupancy.results_mutually_clear` (call
  ``filters.adsorbates_mutually_disconnected`` directly).

## [0.9.4] - 2026-10-02

### Changed

- **n-tuplet saturation** (`saturation_molecules_per_step > 1`) now screens
  joint configs of exactly *n* adsorbates relaxed together via TorchSim.
  There is no single-adsorbate MLIP screening pass and no partial-tuplet /
  single-winner fallback.
- Stored `energy_adsorption` for joint configs is **per molecule**
  (`E_ads_total / n`); composite totals remain on `energy_adslab` /
  `energy_adsorbate`. Stop and ranking use Ω_tuplet (equivalently Ω/n at
  default reservoir conditions).
- Workload autotune divides `num_placements`, `bo.initial_random`, and
  `bo.batch_size` by *n* (each eval is an n-body relax).
- Single-molecule BO in n-tuplet mode proposes joint configs from the
  single-site surrogate and records shared Ω/n labels per member site.
- Clash descent uses per-fixed-atom clearance scales (substrate: covalent
  radius sum; pre-adsorbed / packed adsorbates: `connectivity_multiplier` on
  the radius sum) and re-checks packs with the shared adsorbate disconnect
  gate. Exact n-tuplet packing rejects overlapping packs when clash descent
  is off.
- Competitive multi-molecule n-tuplet enumerates every species composition
  of size *n* and ranks packs by Ω_tuplet; joint BO with that combination
  raises.

### Added

- `workflow.joint_tuplet` for homogeneous and multi-molecule joint screening
  plus joint BO placement bias.
- `assemble_quota_joint_configs` / `evaluate_composite_batch` helpers in
  `workflow.composite`.
- `scripts/co_pt111_ntuplet_phases.py`: independent one-step high-*n* joint
  searches on the Gunasooriya & Saeys CO/Pt(111) coverage cells (site
  histograms and gap to the literature bridge:top ratio).

### Removed

- `scripts/co_pt111_ordered_coverages.py` (hand-seeded literature registries;
  superseded by the n-tuplet phase-discovery script above).
- `BOTransferConfig.occupancy_lengthscale` / `occupancy_floor` (BO sigma
  inflation beside occupied adsorbates now uses the shared connectivity
  covalent-sum ratio vs ``connectivity_multiplier``).
- `placement.geometry.check_adsorbate_separation` (adsorbate–adsorbate
  legality is ``filters.adsorbates_mutually_disconnected``).
