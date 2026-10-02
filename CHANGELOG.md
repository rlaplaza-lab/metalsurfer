# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
