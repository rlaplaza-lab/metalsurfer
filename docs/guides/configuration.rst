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
  ``TypeError`` from :class:`~metalsurfer.AdsorptionConfig`.
- Prefer ``write_settings=True`` (default) for ``run_metadata.json``.
  Set ``write_settings=False`` to suppress it.
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

Do not confuse ``min_contact_ratio`` (default **0.8**, unitless fraction of the
covalent-radius sum) with ``max_closest_approach`` (default **0.8** Å, absolute
closest-approach distance used by the contact-quality layer).

Under saturation, substrate contact uses the bare-slab atom prefix while prior
adsorbates are checked with adsorbate–adsorbate separation. Generation failures
emit typed reasons (``too_close``, ``too_far``, ``vdw_overlap``,
``adsorbate_overlap``, ``distance_check_failed``, …) into
``PlacementFailureEvent`` / placement ``failure_summary``.

Placement success levers
------------------------

Defaults aim for high accept rates with low overhead (work runs mainly on failures):

- **Orientation mix** — ``adaptive_parallel_fraction=True`` picks parallel vs EN-down
  from binder/ring chemistry; set ``False`` and tune
  ``flat_aromatic_parallel_fraction`` for a fixed mix.
- **Distance recovery** — ``placement_distance_recovery=True`` nudges height then
  small in-plane offsets (``placement_x_range`` / ``placement_y_range``, default
  ±0.5 Å) after ``too_close`` / ``too_far``. Use ``(0.0, 0.0)`` XY ranges for
  height-only recovery, or disable recovery entirely with
  ``placement_distance_recovery=False``.
- **Site window** — ``voronoi_auto_widen=True`` retries once with a wider Voronoi
  accessibility window when the first pass finds no sites; pair with explicit
  ``voronoi_probe_radius`` / ``voronoi_max_site_distance`` when comparing windows.
- **Retries** — ``placement_retry_*`` re-enumerates remaining slots with new seeds
  after generation failures. Each deficit round oversamples by estimated
  materialization yield (capped by ``placement_retry_oversample_max``) and stops
  early when the target is met or enumeration returns nothing. Per-spec
  materialization runs in a thread pool sized by ``placement_materialize_workers``
  (joblib-style; default ``-2`` = all but one CPU). BO eval batches backfill from
  the unused valid pool (also yield-oversampled) so each step still reaches its
  requested size before relaxation.
- **Gates** — keep ``reject_vdw_overlaps`` and ``strict_initial_placement`` off
  unless you need stricter starts (they reduce yield).

Site classification defaults to ``site_classification_method="auto"``: Delaunay
for slabs (catalysis-style atop/bridge/hollow catalogs) and distance-ratio for
nanoparticles and porous materials. Explicit ``"distance_ratio"`` on slabs is
honored for A/B comparisons.

Material-aware placement asymmetries (hybrid topology on slabs, parallel-z floors
for open surfaces only, no porous dissociative) are intentional for sampling
effectiveness — see :doc:`architecture`.

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
:func:`~metalsurfer.run_campaign`). Those select BO mode; ``bo`` / ``bo.transfer``
fields are hyperparameters only. See :doc:`../api/campaigns` for the YAML
``campaign`` mapping.

Saturation essentials
---------------------

Call :func:`~metalsurfer.run_saturation` or :func:`~metalsurfer.run_saturation_bo`.
Key fields:

- ``saturation_discard_topology_rearrangements`` (default ``True``) — connectivity
  guard on the full adsorbate pool before each step advance
- ``saturation_save_all_placements`` (default ``True``) — disk-heavy; set ``False``
  for large placement counts
- ``multi_molecule_saturation`` — competitive saturation when multiple SMILES are loaded
- ``bo.transfer.*`` — cross-step BO memory in ``run_saturation_bo`` (see
  :doc:`../api/config` — Bayesian optimization)

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
