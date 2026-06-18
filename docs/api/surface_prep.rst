Substrate preparation
=====================

All substrate and material preparation lives in :mod:`metalsurfer.surface_prep`.
Import from this module — campaign APIs validate substrates but never align,
resize, rewrite constraints, or re-equilibrate ionic positions.

Layout conventions and worked examples: :doc:`../guides/surface_engineering`.

Two stages: prep equilibration vs adsorption freeze
---------------------------------------------------

**Prep equilibration** (:func:`~metalsurfer.surface_prep.prepare_substrate`)

By default, :func:`~metalsurfer.surface_prep.prepare_substrate` **relaxes substrate
ionic positions** with ASE/MLIP (``AdsorptionConfig.slab_relaxation_mode="ionic_only"``).
The substrate passed to ``run_adsorption`` / ``run_saturation`` should be this
**equilibrated reference** — ``E(slab)`` and ``E_ads`` are defined relative to it.
Hand-built clusters, slabs from bulk, and loaded ``Atoms`` are all relaxed unless
you set ``slab_relaxation_mode="none"`` (experimental geometries that must not move:
MOF CIF, paper DFT slabs, published graphene-oxide models, saturation snapshot XYZ).

**Adsorption / saturation (campaign APIs)**

During placement optimization, **only the adsorbate and any substrate atoms not
listed in ASE ``FixAtoms`` can move**. Campaign entry points read
``frozen_indices_from_constraints`` on the prep substrate (or saturation
``base_slab``) and log which atoms are frozen vs free.

:func:`~metalsurfer.surface_prep.prepare_substrate` and
:func:`~metalsurfer.surface_prep.finalize_substrate` attach ``FixAtoms`` via
:func:`~metalsurfer.surface_prep.apply_surface_constraints` using prep-time
keyword arguments ``relax_top_layer``, ``freeze_symbols``, and
``top_layer_tolerance``.

+--------------------------------+-----------------------------------------------+
| Kwarg / value                  | Effect during placement relaxation            |
+================================+===============================================+
| ``relax_top_layer=False``      | All substrate atoms frozen (default, rigid    |
| (default)                      | reference surface)                            |
| ``relax_top_layer=True``       | Material-aware shortcut: interior frozen;     |
|                                | exposed surface free (see table below)        |
| ``freeze_symbols`` set         | Only listed elements frozen (ignores layer    |
|                                | policy)                                       |
+--------------------------------+-----------------------------------------------+

When ``relax_top_layer=True``, which atoms stay free depends on
:attr:`~metalsurfer.AdsorptionConfig.material_type` on the *config* passed to
``prepare_substrate`` / ``finalize_substrate``. There is no separate prep
``material_type`` argument — set it on ``AdsorptionConfig`` before prep and reuse
the same config in ``run_*`` (omitting *config* defaults to ``"slab"``).

+--------------------+---------------------------------------------------------+
| ``material_type``  | Free atoms (within ``top_layer_tolerance``)             |
+====================+=========================================================+
| ``"slab"``         | Exposed layer along the slab normal                     |
| ``"nanoparticle"`` | Outermost shell (max distance from centre of mass)      |
| ``"porous"``       | Pore-wall atoms (closest neighbour per pore void site)  |
+--------------------+---------------------------------------------------------+

For custom freeze patterns, attach ASE ``FixAtoms`` yourself or call
``apply_surface_constraints`` / ``finalize_substrate`` with ``freeze_symbols``.

Example — rigid substrate (default) vs top-layer relaxation::

   config = AdsorptionConfig(material_type="slab")

   # Default: prep equilibrates ions, then entire substrate is frozen during adsorption
   slab = prepare_substrate(bulk_id="mp-33", miller_indices=(0, 0, 1), config=config)

   # Allow the top surface layer to relax with the adsorbate
   slab = prepare_substrate(
       bulk_id="mp-33",
       miller_indices=(0, 0, 1),
       config=config,
       relax_top_layer=True,
   )

   # Skip prep equilibration (experimental geometry only)
   config = AdsorptionConfig(material_type="porous", slab_relaxation_mode="none")
   slab = prepare_substrate(slab=mof_atoms, config=config, align=False)

Recommended import
------------------

.. code-block:: python

   from metalsurfer import AdsorptionConfig
   from metalsurfer.surface_prep import (
       prepare_substrate,
       resize_substrate_for_molecule,
   )

   config = AdsorptionConfig(material_type="slab", seed=42)
   slab = prepare_substrate(
       bulk_id="mp-33",
       miller_indices=(0, 0, 1),
       config=config,
       results_dir="results_demo",
   )

Public API overview
-------------------

**Orchestration**

- :func:`~metalsurfer.surface_prep.prepare_substrate` — build/load/modify/finalize
- :func:`~metalsurfer.surface_prep.finalize_substrate` — PBC + ``FixAtoms`` + validate
  after custom step-by-step edits (does **not** relax)
- :func:`~metalsurfer.surface_prep.relax_substrate` — ASE equilibration of a reference
  substrate (building block and loaded-slab path inside ``prepare_substrate``)
- :func:`~metalsurfer.surface_prep.resize_substrate_for_molecule` — in-plane expand
  after conformer generation

**Layout, PBC, constraints, validation**

- :func:`~metalsurfer.surface_prep.apply_material_pbc`
- :func:`~metalsurfer.surface_prep.ensure_slab_z_alignment`
- :func:`~metalsurfer.surface_prep.apply_surface_constraints`
- :func:`~metalsurfer.surface_prep.validate_substrate`
- :func:`~metalsurfer.surface_prep.accept_substrate_for_api`
- :func:`~metalsurfer.surface_prep.coerce_slab_container`

**Construction and modification**

- :class:`~metalsurfer.surface_prep.SlabContainer`
- :func:`~metalsurfer.surface_prep.create_slab_from_bulk`
- :func:`~metalsurfer.surface_prep.create_slab_from_atoms`
- :func:`~metalsurfer.surface_prep.substitute_alloy`
- :func:`~metalsurfer.surface_prep.deposit_adatoms`

**In-plane sizing**

- :func:`~metalsurfer.surface_prep.auto_resize_substrate_for_molecule`
- :func:`~metalsurfer.surface_prep.compute_minimum_supercell`

Relaxation
----------

Prep-time relaxation equilibrates the **reference substrate** with ASE before
campaign APIs run. It is separate from TorchSim placement relaxation
(``fmax``, ``stage1_steps``, ``stage2_steps``, ``ts_optimizer`` on
:class:`~metalsurfer.AdsorptionConfig`).

:func:`~metalsurfer.surface_prep.prepare_substrate` accepts knobs named like
``AdsorptionConfig.slab_relaxation_*``. Explicit keyword arguments override
*config* for each stage:

+--------------------------------+-----------------------------------------------+
| ``prepare_substrate`` kw       | ``AdsorptionConfig`` field                    |
+================================+===============================================+
| ``slab_relaxation_mode``       | ``slab_relaxation_mode``                      |
| ``slab_relaxation_optimizer``  | ``slab_relaxation_optimizer``                 |
| ``slab_relaxation_fmax``       | ``slab_relaxation_fmax`` (falls back to       |
|                                | ``fmax`` when unset)                          |
| ``slab_relaxation_steps``      | ``slab_relaxation_steps``                     |
| ``adatom_relaxation_*``        | defaults to ``slab_relaxation_*`` on *config* |
|                                | when unset                                    |
+--------------------------------+-----------------------------------------------+

**When each stage runs**

- **Bulk creation** (``bulk_id=...``): ``slab_relaxation_*`` is passed to
  :func:`~metalsurfer.surface_prep.create_slab_from_bulk`.
- **Loaded substrate** (``slab=...`` or ``slab_file=...``): ``slab_relaxation_*``
  runs via :func:`~metalsurfer.surface_prep.relax_substrate` before
  :func:`~metalsurfer.surface_prep.finalize_substrate`.
- **Adatom deposition**: ``adatom_relaxation_*`` is passed to
  :func:`~metalsurfer.surface_prep.deposit_adatoms`.

**Modes**

- ``"none"`` — no relaxation.
- ``"ionic_only"`` — relax atomic positions with fixed cell (default).
- ``"cell_only"`` — relax cell with ionic coordinates constrained.
- ``"full"`` — relax both ionic coordinates and cell.

Example: fully equilibrate the clean slab once, then ionic-only relaxation
after adatom deposition:

.. code-block:: python

   config = AdsorptionConfig(
       slab_relaxation_mode="full",
       slab_relaxation_steps=250,
   )
   slab = prepare_substrate(
       bulk_id="mp-81",
       miller_indices=(1, 1, 1),
       adatom_symbol="Au",
       adatom_coverage=0.20,
       config=config,
       adatom_relaxation_mode="ionic_only",
   )

Orchestration reference
-----------------------

.. autofunction:: metalsurfer.surface_prep.prepare_substrate

.. autofunction:: metalsurfer.surface_prep.finalize_substrate

.. autofunction:: metalsurfer.surface_prep.relax_substrate

.. autofunction:: metalsurfer.surface_prep.apply_material_pbc

.. autofunction:: metalsurfer.surface_prep.resize_substrate_for_molecule

Prep writes reference structures under ``results_dir`` (for example
``clean_slab.xyz`` before adatoms and ``clean_slab_Au20.xyz`` after 20\%
coverage). Saturation uses the post-prep slab as ``base_slab_for_frozen``;
compare placement-relaxed geometries to the file that matches that state, not
an earlier prep snapshot.

Building-block reference
------------------------

.. autofunction:: metalsurfer.surface_prep.ensure_slab_z_alignment

.. autofunction:: metalsurfer.surface_prep.apply_surface_constraints

.. autofunction:: metalsurfer.surface_prep.validate_substrate

.. autofunction:: metalsurfer.surface_prep.accept_substrate_for_api

.. autofunction:: metalsurfer.surface_prep.create_slab_from_bulk

.. autofunction:: metalsurfer.surface_prep.create_slab_from_atoms

.. autofunction:: metalsurfer.surface_prep.auto_resize_substrate_for_molecule

.. autoclass:: metalsurfer.surface_prep.SlabContainer
   :members:
   :undoc-members:
