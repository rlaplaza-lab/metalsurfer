Surface Engineering
===================

The surface can be prepared programmatically before any run mode.
Alloy substitution is applied first, then adatom deposition if both are
used.


Slab Geometry
-------------

For ``material_type="slab"``, metalsurfer uses a bottom-anchored layout:

- substrate atoms start at ``min(z) ≈ 0``
- the adsorption surface is at ``max(z)``
- empty space (vacuum) lies only in +z above the surface

:func:`~metalsurfer.create_slab_from_bulk`, :func:`~metalsurfer.create_slab_from_atoms`,
and :func:`~metalsurfer.prepare_slab` apply this alignment automatically.
Bare ASE ``Atoms`` passed to a run entry point are normalized when
``AdsorptionConfig(material_type="slab")`` is set.


One-Call Preparation
--------------------

:func:`~metalsurfer.prepare_slab` combines slab construction, alloy
substitution, and adatom deposition into a single convenience call:

.. code-block:: python

   from metalsurfer import prepare_slab

   slab = prepare_slab(
       bulk_id="mp-33",
       miller_indices=(0, 0, 1),
       supercell=(2, 2, 1),
       alloy_guest="Cu",
       alloy_fraction=0.25,
       enforce_top_layer_fraction=True,
       adatom_symbol="Sn",
       adatom_coverage=0.20,
       config=config,
       results_dir="results_demo",
       adatom_relaxation_mode="ionic_only",
   )


Step-by-Step Preparation
------------------------

For more control, use the individual helpers.

**Fast structural modification** (no energy ranking):

.. code-block:: python

   from metalsurfer import create_slab_from_bulk, substitute_alloy, deposit_adatoms

   slab = create_slab_from_bulk(bulk_id="mp-33", miller_indices=(0, 0, 1))

   slab = substitute_alloy(
       slab,
       host_symbol="Ru",
       guest_symbol="Cu",
       guest_fraction=0.25,
   )

   slab = deposit_adatoms(
       slab,
       adatom_symbol="Sn",
       coverage_fraction=0.20,
   )

**Energy-ranked variant selection** (recommended for realistic modified
surfaces):

.. code-block:: python

   from metalsurfer import (
       AdsorptionConfig,
       create_slab_from_bulk,
       deposit_adatoms,
       setup_single_model,
       substitute_alloy,
   )

   config = AdsorptionConfig(material_type="slab")
   slab = create_slab_from_bulk(bulk_id="mp-33", miller_indices=(0, 0, 1))
   calculator, _ = setup_single_model(config.model_name, config.device)

   slab = substitute_alloy(
       slab,
       host_symbol="Ru",
       guest_symbol="Cu",
       guest_fraction=0.25,
       calculator=calculator,
       config=config,
   )

   slab = deposit_adatoms(
       slab,
       adatom_symbol="Sn",
       coverage_fraction=0.20,
       calculator=calculator,
       config=config,
       relaxation_mode="full",  # full, ionic_only, cell_only, none
   )

Relaxation presets for slab preparation
---------------------------------------

``create_slab_from_bulk(...)`` and ``deposit_adatoms(...)`` support shared
relaxation presets:

- ``"none"``: no slab relaxation (default).
- ``"ionic_only"``: relax atomic positions with fixed cell.
- ``"cell_only"``: relax cell with ionic coordinates constrained.
- ``"full"``: relax both ionic coordinates and cell.

You can set defaults once on :class:`~metalsurfer.AdsorptionConfig`:

.. code-block:: python

   config = AdsorptionConfig(
       slab_relaxation_mode="full",
       slab_relaxation_optimizer="lbfgs",  # lbfgs, bfgs, fire
       slab_relaxation_fmax=0.03,          # optional, falls back to config.fmax
       slab_relaxation_steps=250,
   )

The ``calculator`` argument is **optional** for both
``substitute_alloy(...)`` and ``deposit_adatoms(...)``:

- Without a calculator: a valid modified slab is created (fast structural
  modification).
- With a calculator: random variants are energy-scored and the
  lowest-energy variant is selected.

Use separate relaxation presets for bulk slab creation vs adatom deposition
when you want a single full equilibration of the clean surface but only ionic
relaxation after adding adatoms:

.. code-block:: python

   slab = prepare_slab(
       bulk_id="mp-81",
       miller_indices=(1, 1, 1),
       adatom_symbol="Au",
       adatom_coverage=0.20,
       config=config,
       adatom_relaxation_mode="ionic_only",
   )

Slab freeze during adsorption and saturation
--------------------------------------------

``AdsorptionConfig.slab_relaxation_mode`` controls **prep** only (ASE).
During placement relaxation, ``relax_top_layer`` and ``base_slab_for_frozen``
control TorchSim ``FixAtoms``:

- ``relax_top_layer=False``: freeze every atom in the substrate reference
  (typical for a fixed slab during adsorption).
- Saturation stores ``base_slab`` once after ``prepare_slab``; only those
  indices stay fixed as adsorbates accumulate. Earlier adsorbate units may
  still relax in later steps.
- When adatoms are deposited, the freeze reference is the post-adatom slab
  (e.g. ``clean_slab_Au20``), not ``clean_slab`` written before deposition.
- If ``auto_resize_slab`` expands the substrate on saturation step 1, the
  freeze reference is updated to the full repeated substrate so periodic
  image tiles are not left unfrozen (standard, BO, and saturation paths;
  both ``relax_top_layer`` settings).  Multi-molecule competitive saturation
  pre-resizes once before evaluating candidates on step 1.


Material Type
-------------

:attr:`AdsorptionConfig.material_type <metalsurfer.AdsorptionConfig.material_type>`
must be set explicitly.  Valid values:

- ``"slab"`` — in-plane periodic surfaces.
- ``"nanoparticle"`` — non-periodic clusters.
- ``"porous"`` — fully periodic porous frameworks.

This choice affects site generation, adsorption validation, and distance
handling throughout the workflow.
