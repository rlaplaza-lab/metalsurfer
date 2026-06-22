# Changelog

All notable user-facing changes to Metalsurfer are documented here.

## [0.3.0] - 2026-06-20

### Added

- Four high-level campaign APIs: `run_adsorption`, `run_adsorption_bo`, `run_saturation`, and `run_saturation_bo`.
- Unified workflow bootstrap for binding and saturation campaigns.
- `prepare_substrate` surface-prep entry point with ASE equilibration and `FixAtoms` defaults.
- Release QA runners: `scripts/run_all_tests.sh` and `scripts/run_all_examples.sh`.

### Changed

- Campaign results use typed dataclasses (`BindingCampaignResult`, `SaturationCampaignResult`) with format helpers.
- `skip_existing=True` (default) skips molecules already in `adsorption_energies_detailed.csv` (binding) or `saturation_summary.csv` (saturation).
- `bo_total_budget` counts acquisition batches after the initial random batch, not total MLIP evaluations.
- HPC scripts under `scripts/` are standalone copy-paste workflows aligned with `prepare_substrate`.
- Quick binding examples validate favorable adsorption energies before exit; the Pt₁₂ demo uses ethene instead of H₂ (dissociative H₂ gives unphysical positive E_ads with a molecular reference).

### Documentation

- Quickstart, surface engineering, and development guides updated for v0.3 API.
- Voronoi config defaults documented as runtime-derived when unset.
