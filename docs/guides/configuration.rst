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

Autotuning placements on GPU
----------------------------

Leave these at their defaults (``None``) for production GPU runs:

- ``num_placements`` — non-BO screening batch size
- ``bo_initial_random``, ``bo_batch_size`` — BO batch sizes

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
       skip_topology_check=True,
       seed=42,
   )

Effects:

- Enables hollow-site pair initial placements for H₂
- Disables post-relaxation connectivity / decomposition checks
- Reference energy remains the **isolated molecule**; positive :math:`E_\mathrm{ads}`
  is possible when the relaxed state is dissociated

See ``examples/h2_ru_slab_binding_energy.py``.

Initial placement validation
----------------------------

Three independent layers (do not conflate):

1. **Distance** — ``min_initial_distance``, ``max_initial_distance``, ``min_contact_ratio``
2. **VDW** — ``reject_vdw_overlaps``, ``vdw_overlap_scale``
3. **Contact quality** — ``strict_initial_placement``, ``max_closest_approach``,
   ``min_contact_atoms``, ``contact_distance_threshold``, ``require_multiple_contact``

Under saturation, substrate contact uses the bare-slab atom prefix while prior
adsorbates are checked with adsorbate–adsorbate separation. Generation failures
emit typed reasons (``too_close``, ``vdw_overlap``, ``adsorbate_overlap``, …) into
``PlacementFailureEvent`` / placement ``failure_summary``.

Site classification defaults to ``site_classification_method="auto"``: Delaunay
for slabs (catalysis-style atop/bridge/hollow catalogs) and distance-ratio for
nanoparticles and porous materials. Explicit ``"distance_ratio"`` on slabs is
honored for A/B comparisons.

Material-aware placement asymmetries (hybrid topology on slabs, parallel-z floors
for open surfaces only, no porous dissociative) are intentional for sampling
effectiveness — see :doc:`architecture` and ``CORE_SYSTEM_EXPLANATION.md``.

Bayesian optimization budget
----------------------------

Total BO placement evaluations (after autotune resolves batch sizes):

.. code-block:: text

   bo_initial_random + bo_total_budget * bo_batch_size

``bo_total_budget`` counts **acquisition batches** after the initial random batch,
not total evaluations. Example: target ~300 evals with autotuned batch size 16 and
initial random 16 → set ``bo_total_budget = (300 - 16) // 16`` (integer division).

Use :func:`~metalsurfer.run_adsorption_bo` or :func:`~metalsurfer.run_saturation_bo`;
``bo_enabled=True`` on the config alone has no effect on non-BO entry points.

Saturation essentials
---------------------

Call :func:`~metalsurfer.run_saturation` or :func:`~metalsurfer.run_saturation_bo`.
Key fields:

- ``saturation_discard_topology_rearrangements`` (default ``True``) — connectivity
  guard on the full adsorbate pool before each step advance
- ``saturation_save_all_placements`` (default ``True``) — disk-heavy; set ``False``
  for large placement counts
- ``multi_molecule_saturation`` — competitive saturation when multiple SMILES are loaded
- ``bo_transfer_*`` — cross-step BO memory in ``run_saturation_bo`` (see
  :doc:`../api/config` — Bayesian optimization)

Prep vs campaign relaxation
---------------------------

``slab_relaxation_*`` equilibrates the substrate **before** campaigns during prep.
During adsorption relaxation, only adsorbate atoms and substrate atoms **not** in ASE
``FixAtoms`` move. Default prep freezes the entire substrate; ``relax_top_layer=True``
(on ``prepare_substrate``) leaves a material-aware surface layer free. Details:
:doc:`surface_engineering`.

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
