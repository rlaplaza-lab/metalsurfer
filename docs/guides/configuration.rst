Configuration guide
===================

:class:`~metalsurfer.AdsorptionConfig` centralizes every physical and workflow knob for
prep, screening, Bayesian search, and saturation. This guide covers common choices;
the full field reference is :doc:`../api/config`.

Choosing ``material_type``
----------------------------

Set ``material_type`` on the same ``AdsorptionConfig`` instance used for both
:func:`~metalsurfer.surface_prep.prepare_substrate` and the campaign API.

``slab``
   In-plane periodic surface with vacuum along *z*. Adsorption face at ``max(z)``.
   Use for single-crystal surfaces, adatom-decorated slabs, and literature POSCARs
   with ``slab_relaxation_mode="none"``.

``nanoparticle``
   Non-periodic cluster in a finite box (``pbc=False``). Site detection uses the
   outer shell; no in-plane image separation checks. Hand-built clusters often use
   ``slab_relaxation_mode="none"``.

``porous``
   Fully periodic framework (MOFs, zeolites). Voronoi site generation and probe radii
   dominate placement. Load from CIF and pass through ``prepare_substrate``.

Mismatch between ``material_type`` and the prepared structure's PBC/layout causes
validation errors at campaign start.

``surface_type`` on ``run_*`` is **only** the results folder label
(``results_{surface_type}/``). It does not change physics; set ``material_type``
for that.

Autotuning placements on GPU
----------------------------

Leave these at their defaults (``None``) for production GPU runs:

- ``num_placements`` — non-BO screening batch size
- ``bo.initial_random``, ``bo.batch_size`` — BO batch sizes (nested
  ``bo:`` only; flat ``bo_*`` keys are rejected)

At workflow start Metalsurfer probes TorchSim memory using ``autobatcher_*`` fields
and sets parallel capacity. Demos and CI tests set small explicit integers instead.

Tune OOM vs throughput with ``autobatcher_max_memory_padding`` (default ``0.5``):
lower values allow larger batches; higher values reserve more headroom.

Dissociative adsorption (e.g. H₂)
---------------------------------

For homonuclear diatomics that may dissociate on slabs or nanoparticles:

.. code-block:: python

   config = AdsorptionConfig(
       material_type="slab",
       enable_dissociative_placement=True,
       skip_topology_check=True,
       seed=42,
   )

- ``enable_dissociative_placement=True`` — gate for hollow-site pair
  (or nanoparticle site-pair) initial placements
- ``skip_topology_check=True`` — disables post-relaxation connectivity /
  decomposition checks so fragmented adsorbates are retained

Both flags are independent: dissociative placement requires
``enable_dissociative_placement``; topology skip only affects post-relax
filters.

Reference energy remains the **isolated molecule**; positive :math:`E_\mathrm{ads}`
is possible when the relaxed state is dissociated.

See ``examples/h2_ru_slab_binding_energy.py`` and ``scripts/campaigns/``.

Common mistakes
---------------

- Raising ``fmax`` alone does not relax the post-relax force filter — also raise
  ``max_force_convergence`` if you intend softer acceptance.
- ``bo.total_budget`` is acquisition **batches**, not total evaluations. Use
  :func:`~metalsurfer.config.resolved_bo_eval_budget` once batch sizes are resolved (or
  see the budget section below).
- BO mode is the ``run_*_bo`` entry point or YAML ``campaign: *_bo`` — not a
  config field. Unknown keys such as ``bo_enabled`` in YAML ``config:`` raise
  ``ValueError`` from the campaign schema (before ``AdsorptionConfig`` is built).
- CSV exports (``ml_dataset.csv`` and detailed result CSVs) are lean by
  default. Set ``export_placement_provenance=True`` for ``initial_*``
  placement provenance and full ``ctx_*`` computation settings.

Initial placement validation
----------------------------

Three independent layers (do not conflate):

1. **Distance** — ``min_initial_distance``, ``max_initial_distance``, ``min_contact_ratio``
2. **VDW** — ``reject_vdw_overlaps``, ``vdw_overlap_scale``
3. **Contact quality** — ``strict_initial_placement``, ``max_closest_approach``,
   ``min_contact_atoms``, ``contact_distance_threshold``, ``require_multiple_contact``

Do not confuse ``min_contact_ratio`` (default **0.8**, a unitless fraction of the
covalent-radius sum) with ``max_closest_approach`` (default **3.0** Å, the absolute
closest-approach distance used by the contact-quality layer).

Under saturation, substrate contact uses the bare-slab atom prefix while prior
adsorbates are checked with adsorbate–adsorbate separation. Generation failures
emit typed reasons (``too_close``, ``too_far``, ``vdw_overlap``,
``adsorbate_overlap``, ``distance_check_failed``, …) into
``PlacementFailureEvent`` / placement ``failure_summary``.

Placement success levers
------------------------

- **Orientation mix** — ``adaptive_parallel_fraction=True`` picks parallel vs EN-down
  from binder/ring chemistry; set ``False`` and tune
  ``flat_aromatic_parallel_fraction`` for a fixed mix.
- **Distance recovery** — ``placement_distance_recovery=True`` applies one
  analytic height nudge, then (when ``placement_clash_descent=True``) a
  chemistry-scaled Packmol-style rigid-body clash descent. When clash descent
  is off, discrete in-plane offsets within ``placement_x_range`` /
  ``placement_y_range`` are used instead. ``adsorbate_overlap`` and non-porous
  ``vdw_overlap`` skip height; porous ``vdw_overlap`` uses the same
  height-then-clash path as ``too_close``. Use ``placement_clash_descent=False``
  with ``(0.0, 0.0)`` XY ranges for height-only recovery, or
  ``placement_distance_recovery=False`` to disable.
- **Site window** — ``voronoi_auto_widen=True`` retries once with a wider Voronoi
  accessibility window when the first pass finds no sites; pair with explicit
  ``voronoi_probe_radius`` / ``voronoi_max_site_distance`` when comparing windows.
- **Fill** — one-shot oversample (``placement_retry_oversample_max``) requests
  ``min(capacity, num_placements * oversample)`` specs, materializes once, and
  keeps up to ``num_placements``. When ``placement_retry_enabled``, the first
  pass is short, and materialization recorded failed-spec keys, one diversity
  round excludes those keys. Per-spec
  materialization runs in a thread pool sized by ``placement_materialize_workers``
  (joblib-style; ``None`` inherits ``n_jobs``, which defaults to ``-2`` = all
  but one CPU). BO eval batches wrap the
  pre-materialized geometry-valid cache (no generation backfill).
- **Gates** — keep ``reject_vdw_overlaps`` and ``strict_initial_placement`` off
  unless you need stricter starts (they reduce yield).

Site classification defaults to ``site_classification_method="auto"``: Delaunay
for slabs (catalysis-style atop/bridge/hollow catalogs), hull + nearest-neighbour
topology labels for nanoparticles, and distance-ratio for porous Voronoi
vertices. Explicit ``"distance_ratio"`` on slabs is honored for A/B comparisons.

Site candidate generation defaults to ``site_generator="auto"`` (topology for
slab/NP, Voronoi for porous). Set ``site_generator="topology"``,
``"voronoi"``, or ``"adaptive_grid"`` explicitly for A/B comparisons;
incompatible ``site_generator`` / ``material_type`` pairs are rejected.
``adaptive_grid`` works on all three material types but is never selected by
``auto``. It is an opt-in near-atom Cartesian grid with spacing in Å
(``adaptive_grid_spacing``, default ``0.70``): shells around framework atoms,
clearance + exposure filters (wall-near, not pore centres), one representative
per support key snapped to a target clearance above the support centroid, and
a modest ``merge_radius`` NMS so the catalog does not oversample. Optional
refine halvings (``adaptive_grid_refine_levels``, default ``0``) densify
locally. ``adaptive_grid_nms_framework_scale`` (default ``0.25``) floors
``merge_radius`` on framework median NN. ``side_policy`` (default
``"positive"``) selects which slab face / exposure half-space adaptive-grid
keeps.

Plugin knobs: ``voronoi_*``, ``side_policy``, ``adaptive_grid_*``, and ``n_jobs``
(adaptive_grid shells and Voronoi ridge enrich). Shared post-process:
``site_classification_method``, ``site_equivalence_tolerance``,
``symmetry_tolerance``.

.. list-table:: Which site-generator knobs apply where
   :header-rows: 1
   :widths: 36 64

   * - Knob group
     - Applies to
   * - Shared window: ``voronoi_probe_radius``, ``voronoi_max_site_distance``, ``voronoi_auto_widen``
     - All plugins (accessibility window / one-shot widen retry)
   * - ``top_layer_tolerance``, ``planar_z_variance_threshold``
     - Topology slab (planarity + top-layer band); slab height mask / symmetry planar flag
   * - ``voronoi_site_enrichment``
     - Voronoi (porous / explicit slab); topology slab when the top layer is rough (planar topology and NP skip Voronoi → no-op)
   * - ``adaptive_grid_spacing``, ``adaptive_grid_refine_levels``, ``adaptive_grid_nms_framework_scale``, ``side_policy``
     - ``adaptive_grid`` only (``side_policy`` is still packed into the site-cache key for all generators)
   * - ``n_jobs``
     - ``adaptive_grid`` shells / refine; Voronoi ridge enrich (and rough topology-slab enrich). Topology NP is serial (hull + NN graph)
   * - ``site_classification_method``, ``site_equivalence_tolerance``, ``symmetry_tolerance``
     - Shared post-process after every plugin

Keep ``site_generator="auto"`` for production. Opt into ``adaptive_grid`` for
denser near-atom metal sampling (slabs / nanoparticles) when you want denser
wall-near catalogs than topology; keep Voronoi (``auto``) for MOF **pore
centres** — adaptive_grid is wall-near, not pore centres. Keep
``voronoi_site_enrichment=True``. Avoid ``adaptive_grid_spacing`` below
``0.70`` on MOFs.

Default conclusions from the site A/B + slim GPU binding demos
(``examples/compare_adaptive_grid_ab.py``; H₂/Ru, H₂/Pt₁₃, CO₂/MOF,
ethene/Ru₅₅, slim camphor/Cu(111) BO):

- Keep ``auto`` (topology for slab/NP, Voronoi for porous). Adaptive-grid
  catalogs are typically 20–200× slower to build and did not beat ``auto``
  on best E_ads for H₂/Ru, CO₂/MOF, or camphor/Cu(111).
- Keep ``adaptive_grid_spacing=0.70``, ``adaptive_grid_refine_levels=0``,
  ``adaptive_grid_nms_framework_scale=0.25``. Finer spacing or refine>0
  inflate MOF/metal wall-near counts without improving defaults; larger NMS
  floors over-merge flat metal catalogs.
- Keep ``voronoi_site_enrichment=True``: on RUBTAK01 enrich roughly doubles
  non-pore sites at nearly the same wall time while preserving pore count.
- Set ``site_generator="adaptive_grid"`` only when you explicitly want denser
  **near-atom** metal sampling (e.g. stepped/rough slabs where topology is
  sparse). Do **not** use it for MOF pore-centre screening.

Site uniqueness and sampling
----------------------------

After candidates are classified into ``Site`` records, uniqueness is shared:

- ``site_equivalence_tolerance`` (default 0.05 Å) — merge sites that are both
  spatially close and share the same local environment fingerprint
  (support-atom symbols + distance bins + side label). Ignores ``site_source``
  and classified ``site_type``, so topology / Voronoi / injected atops in the
  same pocket merge. Used by molecular placement, dissociative hollow pairs,
  and adatom hollow selection.
- ``symmetry_tolerance`` (default 0.1 Å) — optional spglib pass that keeps one
  representative of each crystallographically equivalent site on a **clean**
  substrate with a single placement per step. Once molecules are on the
  surface, or when ``saturation_molecules_per_step`` > 1, sampling uses the
  full clustered list again and drops occupied spots
  (``min_adsorbate_separation``). Dissociative / adatoms always keep the full
  clustered list.

Omit ``site_context`` on enumerate/materialize and the same
``resolve_site_context_for_sampling`` path is used as production screening.

Bayesian optimization budget
----------------------------

Total BO placement evaluations (after autotune resolves batch sizes):

.. code-block:: text

   bo.initial_random + bo.total_budget * bo.batch_size

``bo.total_budget`` counts **acquisition batches** after the initial random batch,
not total evaluations. Example: target ~300 evals with autotuned batch size 16 and
initial random 16 → set ``bo.total_budget = (300 - 16) // 16`` (integer division).
After sizes are resolved, :func:`~metalsurfer.config.resolved_bo_eval_budget` returns the
total evaluation count.

Prefer nested Python / YAML::

   from metalsurfer import AdsorptionConfig, BOConfig, BOTransferConfig

   config = AdsorptionConfig(
       bo=BOConfig(
           initial_random=16,
           batch_size=16,
           total_budget=18,
           transfer=BOTransferConfig(enabled=True),
       ),
   )

   # YAML:
   # config:
   #   bo:
   #     initial_random: 16
   #     batch_size: 16
   #     total_budget: 18
   #     transfer:
   #       enabled: true

Flat ``bo_*`` constructor kwargs and flat YAML ``bo_*`` / ``bo_transfer_*``
keys are rejected; nest under ``bo`` / ``bo.transfer``.

Use :func:`~metalsurfer.run_adsorption_bo` or :func:`~metalsurfer.run_saturation_bo`
(or YAML ``campaign: adsorption_bo`` / ``saturation_bo`` with
:func:`~metalsurfer.run_campaign`). See :doc:`yaml_campaigns` for YAML structure
and limitations, and :doc:`../api/campaigns` for the ``campaign`` mapping.

Saturation essentials
---------------------

Call :func:`~metalsurfer.run_saturation` or :func:`~metalsurfer.run_saturation_bo`.
Key fields:

- ``saturation_discard_topology_rearrangements`` (default ``True``) — connectivity
  guard on the full adsorbate pool before each step advance
- ``saturation_save_all_placements`` (default ``True``) — disk-heavy; set ``False``
  for large placement counts
- ``debug_write_sites`` — dump ``sites_plugin_stepNNN.xyz`` /
  ``sites_final_stepNNN.xyz`` once per coverage step under ``xyz_structures/``
- ``saturation_max_steps`` — hard cap on coverage steps (default unlimited);
  a step that commits nothing also stops the run (unbound final)
- ``multi_molecule_saturation`` — competitive saturation: all molecules screened each
  step; lowest ``Ω`` advances the slab
- ``saturation_molecules_per_step`` (default ``1``) — n-tuplet: commit up to this many
  clear winners per step in one composite; empty commits stop as unbound finals
- ``saturation_temperature`` / ``saturation_pressure`` / ``saturation_activities`` /
  ``saturation_omega_shift`` —
  reservoir ranking ``Ω = E_ads − k_B T ln(a_i p / p°)`` (SATP defaults
  ``298.15`` K / ``1`` bar / all ``a_i = 1``). ``saturation_omega_shift`` is a
  scalar or per-species offset in eV subtracted from ``Ω``. Not
  ``boltzmann_temperature``. If
  activities already encode ``p_i / p°``, leave pressure at 1
- ``bo.transfer.*`` — cross-step BO memory in ``run_saturation_bo`` (see
  :doc:`../api/config` — Bayesian optimization)

A runnable competitive example (water + OH⁻ on rutile TiO₂(110), both flags
combined) lives at ``examples/water_oh_rutile_saturation.py``.

MLIP model selection
--------------------

Relaxations run on FairChem UMA checkpoints through TorchSim. Two fields on
:class:`~metalsurfer.AdsorptionConfig` control them:

- ``model_name`` (default ``"uma-s-1p2"``) — checkpoint size/series.
- ``task_name`` (default ``"oc25"``) — UMA task head used for energies and forces.

Keep the pair matched: current 1p2 checkpoints are evaluated with the
``oc25`` head; older 1p1 checkpoints were trained for ``oc20``. If you set an
older checkpoint explicitly, set its matching task head too.

Prep vs campaign relaxation
---------------------------

``slab_relaxation_*`` equilibrates the substrate **before** campaigns during prep.
Freeze policy is also **prep-only** (``relax_top_layer``, ``freeze_symbols``, custom
ASE ``FixAtoms``) — not fields on :class:`~metalsurfer.AdsorptionConfig` or
``run_*`` kwargs. During adsorption relaxation, only adsorbate atoms and substrate
atoms **not** in ASE ``FixAtoms`` move.

- **Default prep** (``prepare_substrate`` / ``finalize_substrate``): freezes the
  entire substrate (``relax_top_layer=False``).
- **Partial freeze:** ``relax_top_layer=True`` on prep leaves a material-aware
  surface band free (for slabs: atoms within ``top_layer_tolerance`` of max height —
  a simple band, not the stepped site mask).
- **Deliberate no freeze:** skip ``apply_surface_constraints`` (or clear ASE
  constraints on the prepared ``Atoms``) before calling ``run_*``. Campaign APIs
  only **warn** when FixAtoms are missing; they do not auto-attach constraints, so
  a fully mobile substrate remains intentional and supported.

Details: :doc:`surface_engineering`.

Literature or pre-relaxed slabs
-------------------------------

When ionic positions must not change at prep:

.. code-block:: python

   config = AdsorptionConfig(material_type="slab", slab_relaxation_mode="none", seed=42)

Used in ``examples/co2_mof_binding_energy.py``, ``examples/camphor_cu111_binding_energy.py``,
and similar loaded-structure workflows.

Further reading
---------------

- Full parameter list: :doc:`../api/config`
- Substrate prep API: :doc:`../api/surface_prep`
- Campaign entry points: :doc:`../api/campaigns`
