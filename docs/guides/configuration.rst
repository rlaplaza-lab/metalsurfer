Configuration guide
===================

:class:`~metalsurfer.AdsorptionConfig` holds the settings for substrate prep,
screening, Bayesian search, and saturation. This page covers the choices most
people change. Every field is listed in :doc:`../api/config`.

What material?
--------------

Set ``material_type`` on the same config used for
:func:`~metalsurfer.surface_prep.prepare_substrate` and the campaign call.
It must match the structure you prepared.

``slab``
   Flat crystal surface (vacuum along *z*). Single-crystal facets, adatom
   slabs, and published POSCARs.

``nanoparticle``
   Finite cluster in a box (no periodic images). Hand-built clusters often
   keep the input geometry with ``slab_relaxation_mode="none"``.

``porous``
   Fully periodic framework (MOFs, zeolites). Load from CIF, then
   ``prepare_substrate``.

``surface_type`` on ``run_*`` is only the results folder name
(``results_{surface_type}/``). Physics come from ``material_type``.

How many placements?
--------------------

On a production GPU, leave these unset (``None``):

- ``num_placements`` — standard screening
- ``bo.initial_random``, ``bo.batch_size`` — Bayesian batches (nest under
  ``bo:``; flat ``bo_*`` keys are rejected)

Metalsurfer sizes the batch from available GPU memory. Demos and CI use small
explicit integers instead. If a run runs out of memory, raise
``autobatcher_max_memory_padding`` (default ``0.5``) to leave more headroom.

A molecule that can split (e.g. H₂)
------------------------------------

.. code-block:: python

   config = AdsorptionConfig(
       material_type="slab",
       enable_dissociative_placement=True,
       skip_topology_check=True,
       seed=42,
   )

- ``enable_dissociative_placement=True`` — place H₂ over hollow/bridge site
  pairs so atoms can separate on the surface.
- ``skip_topology_check=True`` — keep the result after relaxation even when
  the molecule has split.

The reference energy is still the isolated molecule, so :math:`E_\mathrm{ads}`
can be positive after dissociation. See
``examples/h2_ru_slab_binding_energy.py``.

Settings people mix up
----------------------

- Raising ``fmax`` alone does not soften the post-relax force filter — also
  raise ``max_force_convergence``.
- ``bo.total_budget`` counts acquisition **batches**, not total evaluations
  (see the Bayesian section below).
- Bayesian mode is chosen by calling ``run_*_bo`` (or YAML ``campaign: *_bo``),
  not by a config flag such as ``bo_enabled``.
- CSV exports are lean by default. Set ``export_placement_provenance=True``
  for full placement and settings columns.

Where molecules sit
-------------------

Leave ``site_generator="auto"``: topology sites on slabs and nanoparticles,
Voronoi sites in porous frameworks.

Set ``site_generator="adaptive_grid"`` or ``"rolling_probe"`` only when you
want near-atom sampling on stepped surfaces or MOF **pore walls**. Keep
``auto`` (Voronoi) for MOF **pore centres**. Plugin knobs and demo defaults:
:doc:`architecture`.

How a starting pose is accepted
-------------------------------

**Default (leave it).** A start is rejected when any adsorbate–surface atom
pair is closer than
``max(min_initial_distance, covalent_sum * min_contact_ratio)``
(1.5 Å and 0.8). The absolute floor limits light atoms (H) and unknown radii;
the ratio scales with atom size. Near-misses are nudged automatically
(``placement_distance_recovery=True``: height, then a small rigid move, then
in-plane shifts). Set that flag to ``False`` for accept-or-reject only.

**Naming trap.** ``min_contact_ratio`` (0.8, unitless, always on) is not
``max_closest_approach`` (3.0 Å). The second number applies only when a
stricter check below is on. ``max_initial_distance`` (default unset) is the
optional “do not start too far” ceiling on the same always-on check.

**Stricter starts (both off by default).**

- ``reject_vdw_overlaps=True`` — also reject van der Waals overlaps
  (``vdw_overlap_scale``, 1.0 = tabulated radii).
- ``strict_initial_placement`` or ``require_multiple_contact`` — require a
  good contact pattern: closest pair no farther than ``max_closest_approach``,
  at least ``min_contact_atoms`` atoms within ``contact_distance_threshold``.

These cut how many poses survive. Field details: :doc:`../api/config`.

**Orientation and fill.** Leave ``adaptive_parallel_fraction=True``. SMILES
atom-map tags (``[O:1]``) limit which atoms point at the surface. Leave the
fill/retry defaults; they only exist to reach ``num_placements``.

How long a Bayesian search runs
-------------------------------

Total placement evaluations after batch sizes are resolved::

   bo.initial_random + bo.total_budget * bo.batch_size

``bo.total_budget`` is the number of acquisition batches after the initial
random batch. Example: about 300 evals with batch size 16 and initial random
16 → ``bo.total_budget = (300 - 16) // 16``. After sizes resolve,
:func:`~metalsurfer.config.resolved_bo_eval_budget` returns the total count.

Nest BO settings under ``bo`` / ``bo.transfer`` (flat ``bo_*`` keys are
rejected)::

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

Call :func:`~metalsurfer.run_adsorption_bo` or
:func:`~metalsurfer.run_saturation_bo` (or YAML ``campaign: adsorption_bo`` /
``saturation_bo``). See :doc:`yaml_campaigns` and :doc:`../api/campaigns`.

Covering a surface
------------------

Call :func:`~metalsurfer.run_saturation` or
:func:`~metalsurfer.run_saturation_bo`. Settings people usually change:

- ``multi_molecule_saturation=True`` — several molecules compete each step;
  the lowest reservoir score advances the surface.
- ``saturation_molecules_per_step`` — commit up to this many winners per step
  (default ``1``).
- ``saturation_temperature`` / ``saturation_pressure`` /
  ``saturation_activities`` — rank and stop with
  ``Ω = E_ads − k_B T ln(a_i p / p°)`` (defaults: 298.15 K, 1 bar, all
  activities 1). Separate from ``boltzmann_temperature``.
- ``saturation_save_all_placements=False`` — less disk use on large runs.
- ``saturation_max_steps`` — hard cap on coverage steps (default unlimited).

Competitive water + OH⁻ on rutile TiO₂(110):
``examples/water_oh_rutile_saturation.py``.

Which energy model?
-------------------

Relaxations use FairChem UMA through TorchSim:

- ``model_name`` (default ``"uma-s-1p2"``) — checkpoint.
- ``task_name`` (default ``"oc25"``) — energy/force head.

Keep the pair matched: current 1p2 checkpoints use ``oc25``; older 1p1
checkpoints used ``oc20``.

Relax the surface, or keep a published structure
------------------------------------------------

Prep relaxes and freezes the substrate before the campaign. During adsorption,
only the adsorbate and any unfrozen substrate atoms move.

- **Default:** the whole substrate is frozen after prep.
- **Published / pre-relaxed structure:** set
  ``slab_relaxation_mode="none"`` so ionic positions stay fixed:

  .. code-block:: python

     config = AdsorptionConfig(
         material_type="slab",
         slab_relaxation_mode="none",
         seed=42,
     )

- **Partial surface mobility:** ``relax_top_layer=True`` on prep (see
  :doc:`surface_engineering`).

Further reading
---------------

- Full parameter list: :doc:`../api/config`
- Substrate prep: :doc:`surface_engineering` and :doc:`../api/surface_prep`
- Campaign entry points: :doc:`../api/campaigns`
- Site plugins and placement internals: :doc:`architecture`
