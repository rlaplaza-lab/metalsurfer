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

1. Run-mode APIs
~~~~~~~~~~~~~~~~

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
:class:`~metalsurfer.AdsorptionConfig` holds ``bo`` / ``bo.transfer``
hyperparameters only (nested — flat ``bo_*`` keys are rejected).

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

2. Surface preparation
~~~~~~~~~~~~~~~~~~~~~~

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

3. Mid-level per-molecule APIs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Import from :mod:`metalsurfer.workflow` (not the package root):

- ``process_molecule`` / ``process_molecule_bayesian`` — require an
  :class:`~metalsurfer.AdsorptionConfig` and return
  :class:`~metalsurfer.workflow.MoleculeScreenOutcome` (``results``,
  optional stage-typed ``failure_summary``, ``ml_records``, optional BO
  memory / transfer). Campaign-level
  :class:`~metalsurfer.SaturationCampaignResult` stores ``failure_summary``
  keyed by molecule name (``dict[str, FailureSummary]``).
- :func:`~metalsurfer.workflow.run_saturation_screening` (pass
  ``bo_enabled=True`` for BO steps; campaign APIs set this for you)
- ``calculate_reference_energies``, ``load_molecules``

Internal helpers (``_bootstrap_screening_run``, ``_normalize_molecules_input``,
…) live in ``workflow/shared.py`` and are not part of the stable public
surface.

4. Infrastructure and YAML
~~~~~~~~~~~~~~~~~~~~~~~~~~

Root re-exports a small placement surface:
:func:`~metalsurfer.enumerate_placement_specs` and
:func:`~metalsurfer.generate_placement_from_spec`. Everything else lives in
submodules — import explicitly, for example:

- ``metalsurfer.placement`` — ``generate_placement_from_spec``,
  ``generate_placement_from_pose``, ``calculate_min_distance``,
  ``get_symmetry_aware_sites``, …
- ``metalsurfer.optimization`` — TorchSim / FairChem helpers
- ``metalsurfer.filters``, ``metalsurfer.io_results``, ``metalsurfer.ml``,
  ``metalsurfer.symmetry``, ``metalsurfer.conformers``

YAML campaigns use the root helpers
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
   ``E_molecule`` is the lowest MLIP-optimized conformer energy, or the
   UMA isolated-atom ``atom_refs`` skin when the adsorbate is a single
   atom (``workflow/reference.py``). Clean-slab energy must be finite and not ~0.
   The reference step also stores each molecule's pre-optimized conformer
   pack on :class:`~metalsurfer.ReferenceEnergies` (``conformer_packs`` /
   ``get_conformer_pack``) so later placement reuses those geometries.
3. **Conformer generation** — RDKit embed + MMFF; MLIP scoring via
   ``batch_static`` when available; dedup by RMSD/energy. The per-conformer
   energies feed the optional Boltzmann conformer prior
   (``conformer_weighting`` / ``boltzmann_temperature``), which allocates
   placement-spec slots in proportion to ``exp(-(E_i - E_min) / (k_B * T))``.
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

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - API
     - ``molecules``
     - Path
   * - ``run_adsorption`` / ``_bo``
     - CSV or list
     - ``campaigns._run_binding_campaign``
   * - ``run_saturation`` / ``_bo``
     - CSV or list
     - ``workflow.run_saturation_screening`` (``bo_enabled`` from mode)

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
   ├── result_paths.py       # canonical results_{surface_type}/ path helpers
   ├── reporting.py          # typed FailureSummary formatting helpers
   ├── models.py             # typed result dataclasses
   ├── site_plugin_ids.py    # site_generator name ↔ material_type matrix
   ├── optimization/         # MLIP setup, batched relaxation (TorchSim / FairChem)
   ├── surface_prep/         # prepare_substrate, freeze, …
   ├── symmetry.py           # spglib-based symmetry analysis
   ├── ml/                   # BO surrogates, dataset, features (schema 3.0)
   ├── placement/            # site_* + generators / pose / policy
   └── workflow/             # orchestration by run mode
       ├── core.py           # standard per-molecule screening
       ├── bayesian.py       # BO-guided per-molecule screening
       ├── saturation.py     # sequential / multi-mol saturation
       ├── composite.py      # n-tuplet winners + composite commit
       ├── placement_fill.py # one-shot oversample + optional diversity retry
       ├── reference.py      # reference energy preparation
       └── shared.py         # bootstrap, outcomes, validation, autotune

``placement/`` internals: ``site_types``, ``site_coords``, ``site_voronoi``,
``site_classify``, ``site_enumeration``, ``site_adaptive_grid``,
``site_rolling_probe``, ``site_plugins``, ``site_context``, ``occupancy``,
``policy``, ``orientation``, ``pose`` (materialize + validate), ``dissociative``,
``geometry``, ``_material``; public orchestration in ``generators.py``. Site APIs
are imported from ``site_enumeration`` / ``site_coords`` (also re-exported from
``metalsurfer.placement``).


Site detection
--------------

Implementation: ``placement/site_*`` plus ``placement/site_plugins/``
(entry point: ``get_unified_sites``). Plugin names live in
``site_plugin_ids.py``. Core algorithms include ``site_adaptive_grid`` and
``site_rolling_probe``.

Which points get proposed depends on ``site_generator``
(``auto`` / ``topology`` / ``voronoi`` / ``adaptive_grid`` / ``rolling_probe``).
Shared prep (periodicity, probe window) and post-steps (classification,
outward-normal check, catalog clustering / sort) stay in the enumerator.
Plugins emit candidate batches; topology and Voronoi own atop injection and
the slab height mask themselves. Catalog ``Site.xyz`` is the unlifted
anchor (support-plane projection for wall sites, void centre for pores).
The only adsorbate lift is the contact solve in pose.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - ``site_generator``
     - Materials
     - Behaviour
   * - ``auto`` (default)
     - all
     - slab/NP → topology; porous → voronoi
   * - ``topology``
     - slab, nanoparticle
     - Slab: atop/bridge/hollow from the top-layer mesh (Voronoi enrich when
       the surface is rough). NP: convex hull + nearest-neighbour edges
   * - ``voronoi``
     - slab, porous
     - Free-volume Voronoi vertices (+ optional ridge enrich). On slabs skips
       topology (A/B path)
   * - ``adaptive_grid``
     - all
     - Opt-in near-atom Cartesian grid (spacing in Å). **One PBC/clearance
       path** for every material: shells around framework atoms with clearance
       + exposure filters — wall-near, not pore centres. Face / exposure policy
       comes from PBC geometry (not material labels). One representative per
       support key, snapped to a lateral pocket anchor at a target clearance,
       then a modest ``merge_radius`` NMS. Does not inject atops or apply the
       slab height mask (those stay in topology / Voronoi). Sampling increment is
       ``adaptive_grid_spacing``; optional refine halvings via
       ``adaptive_grid_refine_levels``. Same classify / cluster / symmetry /
       placement path afterward. Selectable in config / YAML;
       **not** chosen by ``auto``.
   * - ``rolling_probe``
     - all
     - Opt-in Connolly / SAS wall-near contacts (probe tangent to 1/2/3
       framework spheres → atop / bridge / hollow). Geometric supports are
       retained. Covers slab faces, NP exterior, and MOF **pore walls** (not
       free-volume pore centres). Catalog density matches default
       ``adaptive_grid`` NMS. Uses ``side_policy`` (default ``positive`` remaps
       from the structure PBC mask: slab-like → vacuum face, cluster →
       ``external``, 3D-periodic → ``all``). Selectable in config / YAML;
       **not** chosen by ``auto``.

Generation follows the slab normal (``a × b``) and the surface plane — not
Cartesian ``z``.

Pipeline:

1. Resolve plugin from ``site_generator`` + ``material_type``.
2. Periodic images (3×3×1 slabs, 3×3×3 porous, none for clusters) before
   Voronoi (when the plugin runs Voronoi).
3. Default probe/max distances from framework covalent radii
   (``_derive_voronoi_distance_window``). Slabs use **top-layer** atoms along
   the slab normal; NP/porous use mean radii over all atoms.
4. Plugin ``generate`` → candidate batch. Topology slab: planar top layer
   skips Voronoi; rough slabs merge Voronoi enrichment. Topology NP: hull +
   nearest-neighbour edges. Voronoi: free-volume vertices.
5. Vertices filtered to the primary cell within
   ``[voronoi_probe_radius, voronoi_max_site_distance]``.
6. Optional ridge enrichment (``voronoi_site_enrichment``). On planar slabs
   the topology path does not use ridge enrichment; the flag matters for
   porous frameworks, rough/non-planar slabs, and explicit Voronoi.
7. Typing: distance ratios on six nearest neighbours, or top-layer mesh
   classify (``delaunay`` / ``auto`` on slabs) that labels candidates as
   atop / bridge / hollow. Hollows may carry ``hollow_order`` (3- or 4-fold).
8. ``Site`` records: ``xyz``, local ``normal``, ``kind`` (``wall`` /
   ``void``), ``site_type``, ``slab_indices``, ``env_fingerprint``,
   ``site_source``, ``material_type``. Placement switches on ``kind``;
   ``site_type`` stays for stratified sampling and CSV provenance.
9. Uniqueness in layers: (1) plugin-internal wrap / near-duplicate thinning;
   (2) enumerator vertex merge (~0.1 Å; when Voronoi is appended to topology,
   existing topology points stay frozen); (3) catalog clustering with
   ``site_equivalence_tolerance`` — sites merge when close **and** they share
   the same ``env_fingerprint`` (support-atom symbols + distance bins + side
   label; **not** ``site_source`` or classified ``site_type``); clustering
   metric follows the material PBC mask (open / slab-plane / full MIC);
   (4) optional
   spglib symmetry reduction on the clustered list for molecular sampling on a
   clean substrate.
10. Final list sorted by fractional coordinates for deterministic
    ``site_index``. Sampling rank prefers voids, then hollow / bridge / atop,
    then larger clearance; a capped list reserves room for wall sites when
    both kinds exist.

``_get_unique_sites_for_specs`` returns a ``SiteContext`` with the sampling
catalog (``sites``), the full clustered list (``clustered_sites``), and the
resolved plugin name. A small bounded cache (keyed by geometry fingerprint +
site / Voronoi settings) backs ``resolve_site_context_for_sampling``, which:

1. Reuses unique-sites context when present, then applies symmetry.
2. Uses the full clustered list if substrate symmetry is broken
   (reconstruction or ionic motion). Adsorbates are stripped before that
   check, so they do not latch ``symmetry_broken``.
3. Otherwise tries spglib symmetry reduction on the clustered list (different
   ``site_type`` values stay separate); falls back to the clustered set on
   failure/empty.

Once the full slab has an adsorbate suffix, or when
``saturation_molecules_per_step`` > 1, sampling expands to the full clustered
list even if the substrate space group is unchanged. Occupancy then drops
vertices within ``min_adsorbate_separation`` of existing adsorbate atoms.
``adaptive_grid`` multi-molecule steps share one catalog; other generators
reuse the per-geometry cache. BO features are geometry, not ``site_index``.

``SiteContext.sites`` is what molecular sampling draws from (clustered, then
optionally symmetry-reduced on a **clean** substrate; full clustered list under
coverage). ``clustered_sites`` is always the geometric clustering result used
by dissociative pairs and adatom hollow selection. Generators / pose that omit
an explicit context call ``site_context_for_sampling``, which resolves the same
path as production screening. Site uniqueness is ``site_equivalence_tolerance``
only.

Material strategies:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Type
     - Site strategy
   * - slab
     - Planar: topology atop/bridge/hollow only. Rough: topology + Voronoi
       enrichment on the top-layer band
   * - nanoparticle
     - Convex-hull + nearest-neighbour topology only (atop / bridge / 3- and
       4-fold hollow). Voronoi is skipped. Hull-facet normals lift sites on
       both symmetric and lopsided convex clusters. Cluster symmetry wraps
       transformed fractional coordinates in the padded box but does not fold
       site–site deltas through periodic images. Non-crystallographic groups
       (e.g. Ih) map to the nearest crystallographic subgroup spglib can
       return.
   * - porous
     - 3×3×3 images; pore sites when the framework spans the cell

Key knobs: ``site_generator`` (``auto`` / ``topology`` / ``voronoi`` /
``adaptive_grid`` / ``rolling_probe``),
``voronoi_probe_radius``, ``voronoi_max_site_distance``,
``top_layer_tolerance``, ``symmetry_tolerance``,
``site_equivalence_tolerance``, ``site_classification_method``
(``auto`` / ``distance_ratio`` / ``delaunay``), ``voronoi_auto_widen``,
``adaptive_grid_spacing``, ``adaptive_grid_refine_levels``,
``adaptive_grid_nms_framework_scale``, ``side_policy``, ``n_jobs``.

.. list-table:: Plugin parallelism and enrichment knobs
   :header-rows: 1
   :widths: 22 78

   * - Plugin
     - Uses ``n_jobs`` / notable knobs
   * - topology (slab)
     - Ridge enrich on **rough** slabs via ``n_jobs`` + ``voronoi_site_enrichment``;
       planar slabs skip Voronoi. Window: ``voronoi_probe_radius`` /
       ``voronoi_max_site_distance`` / ``voronoi_auto_widen``
   * - topology (NP)
     - Serial hull + NN graph (``n_jobs`` is a no-op by design). Same accessibility window
   * - voronoi
     - Ridge enrich via ``n_jobs`` when ``voronoi_site_enrichment=True``; emits
       wall-near support atoms for fingerprints (pores keep empty supports)
   * - adaptive_grid
     - Shell / refine chunks via ``n_jobs``; density via ``adaptive_grid_*`` and
       ``side_policy`` (PBC-geometry face filter). No material_type branches,
       no atop inject / height mask
   * - rolling_probe
     - Contact enumeration via ``n_jobs``; ``side_policy`` (PBC-geometry face
       filter; default remaps from the structure PBC mask). Wall-near only; no
       public spacing knob (shares default adaptive_grid NMS density)

See :doc:`configuration` for the full knob-by-plugin table.

**Intentional asymmetries** (not unfinished ports): top-layer mesh + topology
on slabs (pure Voronoi floods the batch with weak candidates); hull +
nearest-neighbour topology on nanoparticles (Voronoi voids are not adsorption
sites); distance-window auto-widen for topology / Voronoi only (adaptive grid
and rolling probe skip it); dissociative wall hollow/bridge pairs on any
material when the flag is on (void sites are never dissociation anchors).
Catalog ``Site.xyz`` is the unlifted support-plane or void-centre anchor
(plugin lift/snap is probe-only; classify projects wall anchors onto the
support plane and flips or drops inward normals). ``surface_ref`` is always
the local site frame (max support height along ``site.normal``; voids use
the void vertex). ``adaptive_grid`` and ``rolling_probe`` register in
``placement/site_plugins/`` with the same candidate-batch contract and a
**single** generation path for all materials (config-selectable, not chosen
by ``auto``). Topology / Voronoi keep system-specific heuristics; the
wall-near opt-in plugins do not.


Placement
---------

Surface reference under coverage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``_build_surface_reference_slab`` prefers a **prefix** of length
``len(base_slab_for_frozen)`` (saturation appends adsorbates as a suffix).
Symbol-set stripping is only a fallback when the covered slab is shorter than
the freeze reference. Site enumeration and substrate distance checks use this
substrate-only view; the full slab is relaxed. Prefix length keeps
same-element adatoms from being treated as substrate.

Occupancy pruning
~~~~~~~~~~~~~~~~~
``available_site_indices`` into the sampling ``SiteContext.sites`` catalog
(shortest **in-plane** MIC distance from each catalog anchor to existing
adsorbate atoms ≥ ``min_adsorbate_separation``, using each site's normal)
without remapping indices—replay/BO keep stable ``site_index`` values.
Under coverage that catalog is the full clustered list (symmetry reduction
is dropped), so unoccupied equivalent copies remain sampleable while
occupied columns are excluded. When ``occupancy_use_footprint`` is enabled,
survivors are ranked by kind, coordination, and clearance (incoming disk
scaled by ``occupancy_footprint_scale``) rather than pruned by a second reject
mask; voids and higher-coordination sites come first when clearances tie
(sampling policy, not a separate uniqueness pass). Dissociative pairing reads
``SiteContext.clustered_sites`` (wall hollow/bridge subset) and shares occupancy
helpers in ``placement/occupancy.py``. Empty available sites yield no specs (no
random-XY fallback). Multi-molecule saturation recomputes
``estimate_conformer_count`` (conformer count) each step and skips zero-capacity
species in ``distribute_placement_budget``.

Enumeration / materialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``policy.py`` — Cartesian product over conformers × sites × orientation
  knobs; **stratified** subsample by ``site_type`` to ``n_desired`` (seeded),
- Soft priors preferring milder tilt and mid ``z_fraction``.
  Sites are ranked by ``kind``, coordination, and clearance before the
  stratified draw. Soft-prior draws prefer earlier indices (voids first when
  present). Under coverage,
  a per-molecule footprint view drops ``(conformer, site)`` pairs that do
  not clear existing adsorbates without rewriting the shared
  ``SiteContext.sites`` catalog (competitive saturation reuses one frame).
- ``orientation.py`` — aromatic heuristics plus ``orient_from_spec`` used by
  pose. EN-down binder candidates are the electronegative elements (O, N, S,
  halogens) **plus marked atoms taken from the heavy-atom SMILES graph** —
  formally charged atoms (``[CH2+]``) and ``[atom:map]``-tagged atoms
  (``[C:1]``). RDKit heavy-atom order addresses conformer indices directly,
  so charged carbocations and user-tagged atoms count as contact points, and
  tagging is a sampling-conditioning knob. Dissociative two-site
  placement uses ``_place_dissociative_two_sites``
  / ``_generate_dissociative_placement_from_spec`` in ``dissociative.py``.
  Molecular / adatom placement goes through
  ``_pose_from_spec`` + validation/descriptor build in ``pose.py``.
- ``generators.py`` — public orchestration (enumerate, materialize, replay,
  complexity/budget). Optional ``placement_filter``;
  ``adaptive_parallel_fraction`` (default on).
  ``generate_placements_from_specs`` materializes a list of specs (optionally
  threaded); ``resolve_materialize_workers`` (in ``placement._parallel``,
  re-exported from ``generators``) maps joblib-style ``n_jobs`` /
  ``placement_materialize_workers`` to a concrete thread-pool size.
- After orientation, a **feasible height interval** along the site normal is
  solved once per rigid pose family (cached on the per-screen pose batch).
  Wall sites: pairwise contact-solved COM is the lower bound / nominal;
  ``z_fraction`` ≤ ``0.5`` clips to contact and values above explore toward
  the upper gate. Void sites: nominal is the free-volume centre (or nearest
  cleared height); the interval is the cleared span containing it. Empty
  intervals fail as ``infeasible_pose``;
  failed multi-contact at nominal fails as ``insufficient_contact_atoms``.
  ``surface_ref`` is always local (support plane or void vertex).
- **Distance recovery** (default on): residual ``too_close`` / ``too_far`` /
  ``contact_distance_too_large`` / ``vdw_overlap`` try one analytic height
  nudge clamped into the feasible window when the worst penetration is along
  the normal; mostly in-plane clashes skip height; huge normal penetration
  fails before Packmol. Then chemistry-scaled clash descent when
  ``placement_clash_descent`` is on (discrete XY only when clash is off).
  ``adsorbate_overlap`` skips height. Void sites (``kind == "void"``)
  invert the height nudge toward the free-volume centre — not
  ``material_type``.
- **Distance-window auto-widen** (default on): topology and Voronoi retry once
  with a wider probe/max when the first window finds no sites; adaptive grid
  and rolling probe skip that retry.
- **Dissociative** (``dissociative.py`` / ``_place_dissociative_two_sites``):
  homonuclear diatomics when ``enable_dissociative_placement=True``. Keep
  ``skip_topology_check=True`` to disable post-relax connectivity checks for
  fragments. Pairs wall hollows and bridges on any material using the
  structure PBC mask; void sites are never dissociation anchors. Fragments
  share one lift direction (averaged site normals) so pair spacing is
  preserved; each site keeps a local support-plane height.
  Descriptor COM + identity quaternion feed ML;
  ``fragment_positions`` are replay-only.
- ``_materialize_spec_placements`` — failures become
  ``PlacementFailureEvent`` (BO negatives when enabled).

Placement fill
~~~~~~~~~~~~~~
One-shot fill enumerates ``min(capacity, num_placements *
placement_retry_oversample_max)`` specs, materializes them in chunks of about
``num_placements`` (threaded via ``placement_materialize_workers``), and stops
early once the target is met. Family / overlap bans apply to the rest of the
current pool between chunks. When ``placement_retry_enabled``, the first pass
is short, and at least one spec failed materialization, one diversity round
re-enumerates excluding those exact failed-spec keys plus reason-aware bans:
``adsorbate_overlap`` bans ``(conformer_index, site_index)``;
``insufficient_contact_*`` / ``infeasible_pose`` exclude the orientation
family. Height is clipped into the feasible interval (not redrawn). BO eval
batches wrap pre-materialized cache hits (no generation backfill); the
geometry-valid pool is built once when features are extracted.

Initial geometry validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Stage
     - Role
   * - ``PlacementSpec``
     - Enumeration template
   * - ``generate_placement_from_spec``
     - → ``PlacementDescriptor`` + ``Atoms``
   * - ``PlacementDescriptor`` / ``PlacementPose``
     - ``x_abs``, ``y_abs``, ``z_abs``, quat, …
   * - ``PlacementRecord``
     - ML row (schema **3.0**); stores a ``descriptor: PlacementDescriptor``
       plus energies/labels/context (CSV still flattens)
   * - ``extract_features``
     - **8 features:** x, y, z, ``conformer_index``, quat_w/x/y/z
       (``x``/``y``/``z`` ← ``x_abs``/``y_abs``/``z_abs``). These are the
       initial-pose replay ingredients.

**Not in the feature vector:** ``site_index``, ``site_type``,
``hollow_order``, ``orientation_type``, tilts/azimuths, ``z_fraction``,
``face_flip``, ``fragment_positions``. CSV lean exports omit these; rich
mode writes them as ``initial_*`` provenance columns only.

Replay paths (all tested): spec → descriptor → pose →
``PlacementRecord.to_placement_descriptor`` (returns ``record.descriptor``);
sklearn row → ``placement_pose_from_features`` →
``generate_placement_from_pose``. BO candidates use
``build_spec_features_geometry_aware`` (materialize → record →
``extract_features``).

Determinism: fractional site ordering, order-independent dedup/clustering,
geometry-keyed site caches, seeded stratified subsample. Within a fixed
code version + slab, ``site_index → xyz`` is stable.


TorchSim batched relaxation
---------------------------

Many slab+adsorbate relaxations run **in parallel** on GPU
(``optimization/``).

On CUDA, ``optimize_adsorbate_slab_batched`` streams constrained OptimStates
into ``InFlightAutoBatcher.load_states`` (TorchSim's iterator API). Waiting
placements stay as ASE ``Atoms`` on CPU, so ``num_placements`` is not a VRAM
limit. The high-level ``ts.optimize`` path concatenates the full list first and
is used on CPU.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Mechanism
     - Role
   * - ``optimize_adsorbate_slab_batched`` + ``InFlightAutoBatcher``
     - Stream placements; pack N relaxations per inflight wave
   * - ``estimate_parallel_relaxation_capacity``
     - One GPU memory probe; returns parallel width plus ``max_memory_scaler``
   * - ``resolve_workload_config``
     - Autotune ``num_placements`` / BO batches and write the probed scaler
   * - ``resolve_saturation_step_workload_config``
     - Resolve once; later steps reuse the written-back scaler
   * - ``stage1_steps`` + ``stage2_steps``
     - Two-stage FIRE/LBFGS budget (default FIRE)
   * - ``saturation_reuse`` / ``saturation_autobatcher_reuse``
     - Reuse the batcher when the slab grows slightly (default 32 atoms / 10%)

Leaving ``num_placements`` (and BO batch fields) as ``None`` is intentional:
the library sizes parallel work to GPU memory and fills
``autobatcher_max_memory_scaler`` from that probe so later BO batches and
saturation steps skip TorchSim re-estimation. Larger adsorbed systems pack
fewer inflight relaxations against the same scaler. Large explicit
``num_placements`` values are safe on small GPUs because only the active
inflight batch holds CUDA geometry tensors.

**Calculator / PBC:** geometry, filters, MIC, and spglib site symmetry use
``material_aware_pbc(material_type)`` (slab ``[T,T,F]``, porous ``[T,T,T]``,
nanoparticle ``[F,F,F]``) — not ``atoms.get_pbc()``. At the UMA boundary,
mixed PBC is normalized to full periodic via ``calculator_pbc_for_atoms``;
periodic *c* ≥ 18 Å (``MIN_CALCULATOR_CELL_C_ANG``). Mixed PBC is rejected on
TorchSim/UMA paths. Stored ``ScreeningResult.atoms`` restore material PBC via
:func:`~metalsurfer.surface_prep.apply_material_pbc` after the calculator
boundary (same helper as prep). Post-relax desorption / decomposition / RMSD
must keep material-aware PBC so calculator TTT cannot invent vacuum wraps.

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
(with ``bo.transfer`` = :class:`~metalsurfer.BOTransferConfig`). Flat
``bo_*`` / ``bo_transfer_*`` constructor and YAML keys are rejected.

Saturation transfer
~~~~~~~~~~~~~~~~~~~
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

Standard screening
~~~~~~~~~~~~~~~~~~
``adsorption``): enumerate, relax every sampled candidate, return survivors.

Bayesian screening
~~~~~~~~~~~~~~~~~~
``adsorption_bo``): surrogate-guided loop over the discrete pool with
geometry-aware features (see above).

Sequential saturation
~~~~~~~~~~~~~~~~~~~~~
``saturation``): screen → optional topology guard → commit best
``Ω < 0`` → refresh slab → repeat until unbound or no placements.
``Ω = E_ads − k_B T ln(a_i p / p°)`` (SATP defaults recover ``E_ads``; see
:doc:`configuration`). ``multi_molecule_saturation=True``: all molecules
compete each step; lowest ``Ω`` wins. ``saturation_molecules_per_step > 1``:
greedily commit up to *n* clear winners via one composite
(``workflow/composite.py``); stop uses ``Ω_tuplet``. Demo:
``examples/water_oh_rutile_saturation.py``.

BO saturation
~~~~~~~~~~~~~
``saturation_bo``): same loop with BO placement selection and optional
transfer. Reservoir ``Ω`` ranking is on the shared commit/stop path;
``observed_y`` stays electronic ``E_ads``.

Stop conditions: empty commit (including n-tuplet ``no_binders``); committed
``Ω ≥ 0`` (``Ω_tuplet`` for multi-winner steps); no valid placements;
``saturation_max_steps``.

Compare structures to **post-adatom** substrate files when adatoms were
deposited during prep. The saturation ``symmetry_broken`` latch still follows
the *substrate prefix* space group / operation fingerprint vs the clean
reference (adsorbate atoms are stripped before that check). Independently,
once an adsorbate suffix is present, molecular sampling uses the full
clustered lattice and occupancy-prunes occupied vertices.


Typed data model
----------------

See :doc:`../api/models`. Highlights:
``ReferenceEnergies``, ``PlacementSpec`` / ``PlacementDescriptor``,
``ScreeningResult``, ``ScreeningRunResult``, ``SaturationStepResult``
(with embedded ``transfer: BOTransferInfo | None``),
``SaturationRunResult``, ``MultiMolSaturation*``, campaign wrappers,
``BOStepMemory``, ``MoleculeCampaignSummary``, and
workflow ``MoleculeScreenOutcome``.


Configuration defaults (spot-check)
-----------------------------------

Full field docs: :doc:`../api/config` and :doc:`configuration`.
Representative defaults (verify in ``config.py`` when debugging):

- ``model_name="uma-s-1p2"``, ``task_name="oc25"``, ``num_placements=None`` (GPU autotune)
- ``placement_distance_recovery=True``, ``placement_clash_descent=True``
  (Packmol-style salvage; discrete XY ±0.5 Å only when clash is off)
- ``voronoi_auto_widen=True``, ``adaptive_parallel_fraction=True``
- ``bo.surrogate="gradient_boost"``, ``bo.initial_sampling="spread_xyz"``,
  ``bo.total_budget=18``, ``bo.transfer.mode="weighted"``
- ``saturation_autobatcher_reuse=True``, ``min_pbc_image_separation=8.0`` Å


Output structure
----------------

Root: ``results_{surface_type}/`` (see :func:`~metalsurfer.results_dir_for`).

- ``adsorption_energies_detailed.csv`` / ``adsorption_energy_summary.csv`` —
  binding.
- ``saturation_summary.csv`` / ``saturation_details.csv`` — saturation.
- ``saturation_placements_detailed.csv`` and ``step_{NNN}_placements/`` when
  ``saturation_save_all_placements=True`` (default).
- ``ml_dataset.csv`` / ``ml_dataset_metadata.json`` — ``DatasetLogger``.
- ``xyz_structures/`` (including optional ``sites_plugin*.xyz`` /
  ``sites_final*.xyz`` when ``debug_write_sites=True``; saturation uses
  ``_stepNNN`` suffixes), optional ``vasp_inputs/``, ``run_metadata.json``.

Writers and CSV row builders share path layout via :mod:`metalsurfer.result_paths`
(``molecule_all_xyz_dir``, ``saturation_xyz_dir``, …). Result-object export
helpers (``to_row``, ``to_rows``, ``format_completion``, …) live on the typed
result classes so scripts need not import internal I/O helpers.


Dataset logging and ML
----------------------

``DatasetLogger`` appends ``PlacementRecord`` rows during binding and
saturation. Feature schema: eight numeric columns — the initial-pose
replay ingredients (absolute COM ``x``/``y``/``z`` from ``x_abs``/``y_abs``/
``z_abs``, ``conformer_index``, unit quaternion). CSV exports are **lean by
default** (features + energies/labels + ``context_hash``). Set
``export_placement_provenance=True`` to also write ``initial_*`` pre-relax
provenance (site, orientation, ``initial_fragment_positions``, …) and full
``ctx_*`` settings. Those provenance fields describe the placement that was
started, not the relaxed geometry (relaxed structures remain in XYZ/POSCAR;
``distance`` / energies are post-relax).

Utilities: ``extract_features``, ``placement_pose_from_features``,
``load_dataset``, ``PlacementRecord.to_placement_descriptor`` /
``to_config``. Schema versioning in ``ml/schema.py`` (``SCHEMA_VERSION``
**3.0**). Shared numerics in ``_numeric_defaults.py``.
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
- Saturation stops when a step commits nothing, when the next adsorption is
  unbound (``Ω ≥ 0``, or ``Ω_tuplet ≥ 0`` for multi-winner steps), or at
  ``saturation_max_steps`` — not at an explicit coverage fraction.
- Rigid substrate by default during adsorption (prep ``FixAtoms``);
  ``relax_top_layer=True`` is a material-aware shortcut distinct from the
  site-enumeration top-layer mask.
- Symmetry accelerates clean-slab site catalogs. After the first adsorbate,
  when n-tuplet co-adsorption will place more than one molecule, or once the
  substrate fingerprint vs the clean reference breaks, sampling uses the
  full clustered lattice and occupancy-prunes occupied sites.
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
