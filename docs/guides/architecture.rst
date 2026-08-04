Architecture
============

This page is the thorough technical reference: public API layers, data flow,
site detection, placement, TorchSim batching, Bayesian screening, saturation,
validation, typed outputs, and positioning vs AdsorbML / BOSS.

For a one-page mental model (dual slab/spec/freeze rules and where to look in
the tree), see
`CORE_SYSTEM_EXPLANATION.md
<https://github.com/rlaplaza-lab/metalsurfer/blob/main/CORE_SYSTEM_EXPLANATION.md>`_
in the repository root. Config recipes:
:doc:`configuration`. Substrate prep:
:doc:`surface_engineering`.


Public API layers
-----------------

Lazy re-exports in ``metalsurfer.__init__`` load heavy modules on first access.

**1. Run-mode APIs**

- :func:`~metalsurfer.run_adsorption` — multi-molecule screening from an
  in-memory ``(smiles, name)`` list **or** CSV path; returns
  :class:`~metalsurfer.BindingCampaignResult`.
- :func:`~metalsurfer.run_adsorption_bo` — BO-guided screening; same return type.
- :func:`~metalsurfer.run_saturation` — sequential saturation (**requires**
  explicit ``molecules``); returns
  :class:`~metalsurfer.SaturationCampaignResult` (per-molecule or multi-mol
  via ``.runs``).
- :func:`~metalsurfer.run_saturation_bo` — saturation with BO placement
  selection and step-to-step transfer.

Prefer ``run_*_bo`` when you want Bayesian selection. BO mode is chosen by the
entry point (or YAML ``campaign: adsorption_bo`` / ``saturation_bo``);
:class:`~metalsurfer.AdsorptionConfig` holds ``bo_*`` hyperparameters only.

All four accept a ``SlabContainer`` (or ASE ``Atoms``), ``molecules``,
:class:`~metalsurfer.AdsorptionConfig`, and ``surface_type`` (results folder
label only).

With ``save_results=True`` (default):

- **Binding** — ``save_single_molecule_results`` per molecule,
  ``save_summary_results`` for campaign CSVs, ML rows via ``DatasetLogger``.
- **Saturation** — ``save_saturation_results``; optional flatten to
  ``adsorption_energies_detailed.csv`` when ``save_benchmark_dataset=True``.

``skip_existing=True`` (default) skips molecules already in
``adsorption_energies_detailed.csv`` (binding) or ``saturation_summary.csv``
(saturation). Official demos pass ``skip_existing=False``.

**2. Surface preparation** — :mod:`metalsurfer.surface_prep`

:func:`~metalsurfer.surface_prep.prepare_substrate` builds or loads a slab,
equilibrates ionic positions by default (``slab_relaxation_mode="ionic_only"``),
optionally alloys / deposits adatoms, and attaches ASE ``FixAtoms`` via prep
kwargs (default: freeze the entire substrate). Freeze policy is **prep-only**.
For slabs, ``relax_top_layer=True`` frees a simple height band within
``top_layer_tolerance`` of the exposed surface (not the stepped site-discovery
mask). Empty freeze sets fall back to freezing the whole substrate. Omitting
``FixAtoms`` is allowed (campaigns warn). See :doc:`../api/surface_prep` and
:doc:`surface_engineering`.

Also: ``finalize_substrate``, ``relax_substrate``,
``resize_substrate_for_molecule``, ``create_slab_from_bulk``,
``create_slab_from_atoms``, ``substitute_alloy``, ``deposit_adatoms``,
``auto_resize_substrate_for_molecule``, ``compute_minimum_supercell``.

**3. Mid-level per-molecule APIs** (custom research loops)

- ``process_molecule`` / ``process_molecule_bayesian`` — return
  :class:`~metalsurfer.workflow.MoleculeScreenOutcome` (``results``,
  ``failure_summary``, ``ml_records``, optional BO memory / transfer)
- :func:`~metalsurfer.run_saturation_screening` (pass
  ``bo_enabled=True`` for BO steps; campaign APIs set this for you)
- ``calculate_reference_energies``, ``load_molecules``

Internal helpers (``_bootstrap_screening_run``, ``_normalize_molecules_input``,
…) live in ``workflow/shared.py`` and are not part of the stable public
surface.

**4. Infrastructure**

Placement (``enumerate_placement_specs``, ``generate_placement_from_spec``,
``generate_placement_from_descriptor``, ``calculate_min_distance``,
``get_symmetry_aware_sites``, …), optimization / TorchSim helpers, filters,
I/O, ML utilities, symmetry, logging/errors — importable from the top-level
``metalsurfer`` namespace where re-exported. YAML campaigns:
:func:`~metalsurfer.load_campaign_yaml` + :func:`~metalsurfer.run_campaign`
(``campaign_schema.py`` + ``campaigns.py``).


End-to-end computational flow
-----------------------------

Physical stages across run modes:

1. **Surface preparation** — Materials Project bulk + Miller indices, or
   existing ASE ``Atoms``. Optional alloy, adatoms, supercell expansion.
   Finalize with :func:`~metalsurfer.surface_prep.prepare_substrate` before
   campaign APIs.
2. **Reference energies** —

   .. math::

      E_\mathrm{ads} = E_\mathrm{adsorbate+slab} - E_\mathrm{slab} - E_\mathrm{molecule}

   Saturation refreshes ``E_slab`` each step (``slab_energy_override``).
   ``E_molecule`` is the lowest MLIP-optimized conformer energy
   (``workflow/reference.py``). Clean-slab energy must be finite and not ~0.
3. **Conformer generation** — RDKit embed + MMFF; MLIP scoring via
   ``batch_static`` when available; dedup by RMSD/energy.
   ``conformer_sampling``: ``cycle`` (default), ``boltzmann``, or ``mixed``.
4. **Placement specification** — deterministic ``PlacementSpec`` grid over
   conformer, site, orientation, tilt, azimuth, height. Sites are
   orientation-aware (slab normal, not Cartesian ``z``). See
   `Site detection`_ and `Placement`_ below.
5. **Optimization** — TorchSim / FairChem batched MLIP relaxation
   (`TorchSim batched relaxation`_).
6. **Validation and filtering** — geometry, decomposition, desorption,
   dedup, energy caps (`Validation layers`_).
7. **Aggregation and persistence** — rank by ``E_ads``; CSV / XYZ /
   metadata; optional VASP bundles when ``write_vasp_inputs=True``.

Campaign routing:

=========== ======================== ===========================================
API         ``molecules``            Path
=========== ======================== ===========================================
``run_adsorption`` / ``_bo``  CSV or list   ``campaigns._run_binding_campaign``
``run_saturation`` / ``_bo``  CSV or list   ``run_saturation_screening`` (``bo_enabled`` from mode)
=========== ======================== ===========================================

Both share ``process_molecule`` or ``process_molecule_bayesian`` (returning
``MoleculeScreenOutcome``) where applicable.


Module layout
-------------

::

   metalsurfer/
   ├── campaigns.py          # run_adsorption, run_adsorption_bo, run_campaign, ...
   ├── campaign_schema.py    # YAML schema for load_campaign_yaml / run_campaign
   ├── config.py             # AdsorptionConfig + nested BOConfig / BOTransferConfig
   ├── conformers.py         # SMILES → conformers
   ├── filters.py            # decomposition / desorption / duplicate filtering
   ├── io_results.py         # CSV, XYZ, optional VASP I/O, metadata
   ├── models.py             # typed result dataclasses
   ├── optimization.py       # MLIP setup, batched relaxation
   ├── surface_prep/         # prepare_substrate, freeze, …
   ├── symmetry.py           # spglib-based symmetry analysis
   ├── ml/                   # BO surrogates, dataset, features (schema 3.0)
   ├── placement/            # site_* + generators / pose / policy
   └── workflow/             # orchestration by run mode
       ├── core.py           # standard per-molecule screening
       ├── bayesian.py       # BO-guided per-molecule screening
       ├── saturation.py     # sequential / multi-mol saturation
       ├── reference.py      # reference energy preparation
       └── shared.py         # bootstrap, outcomes, validation, autotune

``placement/`` internals: ``site_types``, ``site_coords``, ``site_voronoi``,
``site_classify``, ``site_enumeration``, ``site_context``, ``occupancy``,
``policy``, ``orientation``, ``pose`` (materialize + validate),
``dissociative``, ``geometry``, ``_material``; public orchestration in
``generators.py``. Site APIs are imported from ``site_enumeration`` /
``site_coords`` (also re-exported from ``metalsurfer.placement``).


Site detection
--------------

Implementation: ``placement/site_*`` (enumeration entry:
``get_unified_sites``).

Generation is **orientation-aware**: top-layer detection, Voronoi filtering,
topology candidates, and local normals use the slab normal (``a × b``) and
slab-plane projectors—not Cartesian ``z``.

Pipeline:

1. Periodic images (3×3×1 slabs, 3×3×3 porous, none for clusters) before
   Voronoi.
2. Default probe/max distances from framework covalent radii
   (``_derive_voronoi_distance_window``). Slabs use **top-layer** atoms along
   the slab normal; NP/porous use mean radii over all atoms.
3. Slabs: Voronoi on near-surface atoms only (≥4); NN filter distances still
   reference the full framework.
4. **Hybrid slab generator (default):** Delaunay topology atop / bridge /
   hollow in the slab plane, merged with Voronoi enrichment. Bridge midpoints
   from triangulation edges; hollows from triangle centroids.
5. Vertices filtered to the primary cell within
   ``[voronoi_probe_radius, voronoi_max_site_distance]``.
6. Optional ridge enrichment (``voronoi_site_enrichment``).
7. Typing: distance ratios on six nearest neighbours, or **Delaunay**
   nearest-candidate classify (precomputed atop/bridge/hollow XY KDTree) when
   ``site_classification_method`` is ``delaunay`` / ``auto`` on slabs.
   Hollows may carry ``hollow_order`` (3- or 4-fold).
8. ``Site`` records: typed dataclass objects with ``xyz``, local
   ``normal``, ``site_type``, ``slab_indices``, ``env_fingerprint``,
   ``site_source``, ``material_type`` (dict adapters only at the symmetry
   boundary).
9. Dedup via periodic ``_deduplicate_points``; clustering via
   ``_cluster_equivalent_sites`` (fingerprint + MIC).
10. Final list sorted by fractional coordinates for deterministic
    ``site_index``. Topology Delaunay is shared with classification when
    available (one triangulation per slab pass).

``_get_unique_sites_for_specs`` returns a ``SiteContext`` (``sites``,
``use_sites``, ``source``, ``raw_unclustered``). A single LRU cache keyed by
geometry fingerprint + Voronoi config (+ ``symmetry_broken`` for resolved
contexts) backs ``resolve_site_context_for_sampling``, which:

1. Reuses unique-sites context when present, then applies symmetry.
2. Uses clustered sites if symmetry is broken (typical mid-saturation).
3. Otherwise tries ``get_symmetry_aware_sites`` (reusing ``raw_unclustered``);
   falls back to clustered Voronoi on failure/empty.

``clear_site_caches()`` clears the shared cache.

Material strategies:

======= ================================================================
Type    Site strategy
======= ================================================================
slab    Top layer along normal → hybrid topology + Voronoi enrichment
nanoparticle  Full-framework Voronoi; outward normals; no PBC images
porous  3×3×3 images; pore sites when the framework spans the cell
======= ================================================================

Key knobs: ``voronoi_probe_radius``, ``voronoi_max_site_distance``,
``top_layer_tolerance``, ``symmetry_tolerance``,
``site_equivalence_tolerance``, ``site_classification_method``
(``auto`` / ``distance_ratio`` / ``delaunay``), ``voronoi_auto_widen``.

**Intentional asymmetries** (not unfinished ports): hybrid topology +
Delaunay on slabs (pure Voronoi floods GPU with weak candidates); global
``surface_ref`` along the slab normal for height; dissociative hollow pairs
on slabs (rejected for porous; NP uses outward-normal Voronoi pairs);
parallel-z floors for slab/NP aromatics (skipped for porous); no atop
injection / dissociative for porous. NP/porous molecular Cartesian-``z``
``surface_ref`` consistency remains a known follow-up.


Placement
---------

**Surface reference under coverage.**
``_build_surface_reference_slab`` prefers a **prefix** of length
``len(base_slab_for_frozen)`` (saturation appends adsorbates as a suffix).
Symbol-set stripping is only a fallback when the covered slab is shorter than
the freeze reference. Site enumeration and substrate distance checks use this
substrate-only view; the full slab is relaxed. Prefix length keeps
same-element adatoms from being treated as substrate.

**Occupancy pruning.** Under coverage, molecular enumeration builds
``available_indices`` into the full ``SiteContext.sites`` catalog (MIC
distance to existing adsorbate atoms ≥ ``min_initial_distance``) without
remapping indices—replay/BO keep stable ``site_index`` values. Dissociative
hollow filtering shares ``placement/occupancy.py``. Empty available sites
yield no specs (no random-XY fallback). Multi-molecule saturation recomputes
``estimate_molecule_complexity(..., full_slab=...)`` each step and skips
zero-capacity species in ``distribute_placement_budget``.

**Enumeration / materialization**

- ``policy.py`` — Cartesian product over conformers × sites × orientation
  knobs; **stratified** subsample by ``site_type`` to ``n_desired`` (seeded),
  with soft priors preferring milder tilt and mid ``z_fraction``.
  Topology-sourced sites are ordered first on slabs.
- ``orientation.py`` — aromatic heuristics plus ``orient_from_spec`` used by
  pose. Dissociative two-site placement uses ``place_at_sites`` in
  ``dissociative.py``. Molecular / adatom placement goes through
  ``_pose_from_spec`` + validation/descriptor build in ``pose.py``.
- ``generators.py`` — public orchestration (enumerate, materialize, replay,
  complexity/budget). Optional ``placement_filter``;
  ``adaptive_parallel_fraction`` (default on).
- Slab / nanoparticle anchor: ``site.xyz`` offset along the slab or site
  normal so the **closest adsorbate atom** (not the COM) lands at
  ``surface_ref + z_offset`` (clearance-aware lift after orientation).
  Porous frameworks skip the lift (confined pores have opposing walls).
- **Distance recovery** (default on): ``too_close`` / ``too_far`` try height
  then XY; ``adsorbate_overlap`` tries XY only
  (``placement_x/y_range``, ±0.5 Å default). VDW / contact-quality failures
  are not recovered.
- **Voronoi auto-widen** (default on): one wider probe/max retry when the
  first window finds no sites.
- **Dissociative** (``dissociative.py`` / ``place_at_sites``): homonuclear
  diatomics when ``enable_dissociative_placement=True`` (preferred). Legacy:
  ``skip_topology_check=True`` alone still enables placement with
  ``DeprecationWarning``. Keep ``skip_topology_check=True`` to disable
  post-relax connectivity checks for fragments. Descriptor COM + identity
  quaternion feed ML; ``fragment_positions`` are replay-only.
- ``_materialize_spec_placements`` — failures become
  ``PlacementFailureEvent`` (BO negatives when enabled).

**Placement retry** (``workflow/core.py``): up to
``placement_retry_max_attempts`` rounds with seed increments fill the
remaining deficit; exact failed-spec keys are excluded; site indices that
repeatedly fail with ``adsorbate_overlap`` / ``too_close`` are blocked.

**Initial geometry validation** (three layers):

1. Covalent distance — ``min_initial_distance``, ``max_initial_distance``,
   ``min_contact_ratio``.
2. VDW — ``reject_vdw_overlaps``, ``vdw_overlap_scale``.
3. Contact quality — ``strict_initial_placement``, ``max_closest_approach``,
   ``min_contact_atoms``, ``require_multiple_contact``, …

Under saturation, substrate contact uses ``exclude_slab_atoms``;
pre-adsorbed atoms use ``check_adsorbate_separation``. Typed failure
reasons include ``too_close``, ``too_far``, ``vdw_overlap``,
``adsorbate_overlap``, …


Placement materialization and ML injectivity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The surrogate sees **resolved absolute geometry**, not discrete site IDs.

=========================== ================================================
Stage                       Role
=========================== ================================================
``PlacementSpec``           Enumeration template
``generate_placement_from_spec``  → ``PlacementDescriptor`` + ``Atoms``
``PlacementDescriptor`` / ``PlacementPose``  ``x_abs``, ``y_abs``, ``z_abs``, quat, …
``PlacementRecord``         ML row (schema **3.0**); stores a
                            ``descriptor: PlacementDescriptor`` plus
                            energies/labels/context (CSV still flattens)
``extract_features``        **8 features:** x, y, z, ``conformer_index``,
                            quat_w/x/y/z
=========================== ================================================

**Not in the feature vector:** ``site_index``, ``site_type``,
``hollow_order``, ``orientation_type``, tilts/azimuths, ``z_fraction``,
``face_flip``, ``fragment_positions``. CSV lean exports omit these; rich
mode writes them as ``initial_*`` provenance columns only.

Replay paths (all tested): spec → descriptor → pose →
``PlacementRecord.to_placement_descriptor`` (returns ``record.descriptor``).
BO candidates use ``build_spec_features_geometry_aware`` (materialize →
record → ``extract_features``).

Determinism: fractional site ordering, order-independent dedup/clustering,
geometry-keyed site caches, seeded stratified subsample. Within a fixed
code version + slab, ``site_index → xyz`` is stable.


TorchSim batched relaxation
---------------------------

Many slab+adsorbate relaxations run **in parallel** on GPU
(``optimization.py``).

==================================== ========================================
Mechanism                            Role
==================================== ========================================
``optimize_adsorbate_slab_batched`` + ``InFlightAutoBatcher``  Pack N relaxations per wave
``estimate_parallel_relaxation_capacity``  Memory probe / scalers
``resolve_workload_config``          Autotune ``num_placements`` / BO batches
``resolve_saturation_step_workload_config``  Re-probe as the slab grows
``stage1_steps`` + ``stage2_steps``  Two-stage ``ts.optimize`` (FIRE default)
``saturation_reuse`` / ``saturation_autobatcher_reuse``  Amortize probes on deep coverage
==================================== ========================================

Leaving ``num_placements`` (and BO batch fields) as ``None`` is intentional:
the library sizes parallel work to GPU memory.

**Calculator / PBC:** mixed PBC normalized to full periodic for UMA;
periodic *c* ≥ 18 Å (``MIN_CALCULATOR_CELL_C_ANG``). Mixed PBC rejected on
TorchSim/UMA paths.

**Prep vs adsorption relaxation:** prep uses ASE
``slab_relaxation_mode``. Adsorption freeze masks come from ASE ``FixAtoms``
on the substrate reference. Saturation pins ``base_slab_for_frozen`` so
indices ≥ original substrate length may relax. Size in-plane during prep
(``auto_resize_substrate_for_molecule`` / ``resize_substrate_for_molecule``).


Validation layers
-----------------

1. **Per-candidate** (``_evaluate_optimized_candidate``): finite energy,
   ``min_interatomic_distance``, adsorbate force cap
   (``max_force_convergence``), desorption (``binding_distance_threshold``;
   skippable), ``max_adsorption_energy``.
2. **Batch** (``filter_results``): decomposition vs reference SMILES
   (``skip_topology_check`` disables), desorption re-check, energy/RMSD
   dedup.
3. **Saturation step** (``_filter_saturation_topology_results``): adsorbate
   pool must have expected connected-fragment count. Disabled when
   ``saturation_discard_topology_rearrangements=False`` or
   ``skip_topology_check=True``.

``fmax`` is the optimizer stop; ``max_force_convergence`` is the post-relax
reject threshold—raise both when accepting softer convergence.

``PlacementFailureEvent`` records ``placement_id``, ``stage``, and
``reason``; aggregated in logs and optionally fed to BO as penalty labels
(``bo.include_failure_negatives``).


Bayesian screening and transfer
-------------------------------

Finite ``PlacementSpec`` pool → initial batch (``bo.initial_sampling``,
default ``spread_xyz``) → geometry-aware features → surrogate → acquisition
(LCB / EI / PI; default EI) until ``bo.total_budget`` acquisition rounds
after the initial batch.

Surrogates (``bo.surrogate``): ``random_forest``, ``extra_trees``,
``gradient_boost`` (default), ``ridge``, ``gaussian_process``, ``ensemble``.
Transfer-capable models accept per-sample weights; ``gaussian_process`` does
not.

Eval budget once autotuned:
``bo.initial_random + bo.total_budget * bo.batch_size``.

Nested config: :class:`~metalsurfer.BOConfig` on ``AdsorptionConfig.bo``
(with ``bo.transfer`` = :class:`~metalsurfer.BOTransferConfig`). Legacy flat
``bo_*`` / ``bo_transfer_*`` constructor and YAML keys still fold in.

**Saturation transfer** (``run_saturation_bo``): each step emits
``BOStepMemory`` and records ``BOTransferInfo`` on
``SaturationStepResult.transfer`` (multi-mol:
``transfer_by_molecule``). Next step receives prior memory via
``_bo_transfer_memory_in``:

- ``weighted`` (default) — windowed priors
  (``bo.transfer.prior_step_window``), recency / occupancy / similarity
  weights, ``bo.transfer.weight_cap``.
- ``cumulative_refit`` — merge all prior step memories.

Trust logic can auto-disable transfer when it hurts fit. Multi-molecule
saturation keeps **per-adsorbate** memory chains (no cross-species
sharing). Pair with ``saturation_autobatcher_reuse`` for deep coverage.


Run modes
---------

**Standard screening** — :func:`~metalsurfer.run_adsorption` (YAML
``adsorption``): enumerate, relax every sampled candidate, return survivors.

**Bayesian screening** — :func:`~metalsurfer.run_adsorption_bo` (YAML
``adsorption_bo``): surrogate-guided loop over the discrete pool with
geometry-aware features (see above).

**Sequential saturation** — :func:`~metalsurfer.run_saturation` (YAML
``saturation``): screen → optional topology guard → commit best
``E_ads < 0`` → refresh slab → repeat until endothermic or no placements.
``multi_molecule_saturation=True``: all molecules compete each step;
budgets from occupancy-aware complexity; lowest ``E_ads`` wins.

**BO saturation** — :func:`~metalsurfer.run_saturation_bo` (YAML
``saturation_bo``): same saturation loop with Bayesian placement selection
and optional cross-step transfer (see above).

Stop conditions: best ``E_ads ≥ 0``; no valid placements after topology
guard; ``saturation_max_steps`` (default unlimited).

Compare structures to **post-adatom** substrate files when adatoms were
deposited during prep. Symmetry reduction is dropped once coverage breaks
symmetry vs the clean reference.


Typed data model
----------------

See :doc:`../api/models`. Highlights:
``ReferenceEnergies``, ``PlacementSpec`` / ``PlacementDescriptor``,
``ScreeningResult``, ``ScreeningRunResult``, ``SaturationStepResult``
(with embedded ``transfer: BOTransferInfo | None``),
``SaturationRunResult``, ``MultiMolSaturation*``, campaign wrappers,
``BOStepMemory``, ``MoleculeCampaignSummary``, ``TimingInfo``, plus
workflow ``MoleculeScreenOutcome``.


Configuration defaults (spot-check)
-----------------------------------

Full field docs: :doc:`../api/config` and :doc:`configuration`.
Representative defaults (verify in ``config.py`` when debugging):

- ``model_name="uma-s-1p2"``, ``num_placements=None`` (GPU autotune)
- ``placement_distance_recovery=True``, XY recovery ±0.5 Å
- ``voronoi_auto_widen=True``, ``adaptive_parallel_fraction=True``
- ``bo.surrogate="gradient_boost"``, ``bo.initial_sampling="spread_xyz"``,
  ``bo.total_budget=18``, ``bo.transfer.mode="weighted"``
- ``saturation_autobatcher_reuse=True``, ``min_pbc_image_separation=8.0`` Å


Output structure
----------------

Root: ``results_{surface_type}/``.

- ``adsorption_energies_detailed.csv`` / ``adsorption_energy_summary.csv`` —
  binding.
- ``saturation_summary.csv`` / ``saturation_details.csv`` — saturation.
- ``saturation_placements_detailed.csv`` and ``step_{NNN}_placements/`` when
  ``saturation_save_all_placements=True`` (default).
- ``ml_dataset.csv`` / ``ml_dataset_metadata.json`` — ``DatasetLogger``.
- ``xyz_structures/``, optional ``vasp_inputs/``, ``run_metadata.json``.

Result-object export helpers (``to_row``, ``to_dataframe``,
``format_completion``, …) are methods on the typed result classes so scripts
need not import internal I/O helpers.


Dataset logging and ML
----------------------

``DatasetLogger`` appends ``PlacementRecord`` rows during binding and
saturation. Feature schema: eight numeric columns (absolute **initial** xyz,
``conformer_index``, unit quaternion). CSV exports are **lean by default**
(features + energies/labels + ``context_hash``). Set
``export_placement_provenance=True`` to also write ``initial_*`` pre-relax
provenance (site, orientation, ``initial_fragment_positions``, …) and full
``ctx_*`` settings. Those provenance fields describe the placement that was
started, not the relaxed geometry (relaxed structures remain in XYZ/POSCAR;
``distance`` / energies are post-relax).

Utilities: ``extract_features``, ``train_model``, ``evaluate_model``,
``grouped_cross_validate``, ``BindingEnergyPredictor``, ``load_dataset``,
``PlacementRecord.to_placement_descriptor`` / ``to_config``. Schema versioning
in ``ml/schema.py`` (``SCHEMA_VERSION`` **3.0**). Shared numerics in
``_numeric_defaults.py``.
Loaders still accept legacy unprefixed provenance columns from schema ≤2.3.


Comparison with AdsorbML and BOSS
---------------------------------

Shared goal: low-energy adsorbate–surface configurations and
:math:`E_\mathrm{ads}`.

**AdsorbML** (Ulyssi et al., npj Comput. Mater. 2023) — ML ranks; final
energy from DFT. Heuristic + random surface sampling; GPU relax-then-rank.
Metalsurfer uses an MLIP end-to-end, orientation-aware discrete placement,
TorchSim in-flight batching, and multi-step saturation with optional BO
transfer. Prefer AdsorbML-style hybrid when you need DFT-grade publication
energies (export Metalsurfer structures for external DFT).

**BOSS** (Todorović & Rinke; continuous GP-BO on building-block DoF) —
learns a continuous PES with few expensive evaluations. Metalsurfer
enumerates a discrete ``PlacementSpec`` pool and relaxes atomistically with
batched MLIP—BOSS-inspired in spirit, not a drop-in replacement. Prefer BOSS
for bulky adsorbates with few effective DoF under a DFT budget.

Prefer Metalsurfer for high-throughput screening, MOFs/nanoparticles, and
many-step coverage on generalizable MLIPs (``run_saturation_bo``).


Design heuristics
-----------------

- Many placements, not one pose: binding energy is the best of a filtered
  sample.
- Saturation stops when the next adsorption is endothermic
  (``E_ads ≥ 0``), not at an explicit coverage fraction.
- Rigid substrate by default during adsorption (prep ``FixAtoms``);
  ``relax_top_layer=True`` is a material-aware shortcut distinct from the
  site-enumeration top-layer mask.
- Symmetry accelerates clean-slab site catalogs until coverage breaks it.
- GPU-first TorchSim + optional BO transfer for deep coverage.
- Layered topology guards; prefer ``enable_dissociative_placement=True``
  with ``skip_topology_check=True`` for fragmented H₂-like adsorbates.
- Substrate-only site view + occupancy prune under coverage.
- Geometry-only ML features: site indices label enumeration slots; the
  surrogate sees materialized absolute poses.


Dependencies
------------

Core: ``numpy``, ``ase``, ``pandas``, ``rdkit``, ``scipy``,
``scikit-learn``, ``spglib``. Optional MLIP: ``torch``,
``torch-sim-atomistic``, FairChem/UMA. Missing optional deps raise
``DependencyMissingError``. Python **3.12+**.
