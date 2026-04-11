Surface Engineering
===================

The surface can be prepared programmatically before any run mode.
Alloy substitution is applied first, then adatom deposition if both are
used.


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
       adatom_symbol="Sn",
       adatom_coverage=0.20,
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
   )

The ``calculator`` argument is **optional** for both
``substitute_alloy(...)`` and ``deposit_adatoms(...)``:

- Without a calculator: a valid modified slab is created (fast structural
  modification).
- With a calculator: random variants are energy-scored and the
  lowest-energy variant is selected.


Material Type
-------------

:attr:`AdsorptionConfig.material_type <metalsurfer.AdsorptionConfig.material_type>`
must be set explicitly.  Valid values:

- ``"slab"`` — in-plane periodic surfaces.
- ``"nanoparticle"`` — non-periodic clusters.
- ``"porous"`` — fully periodic porous frameworks.

This choice affects site generation, adsorption validation, and distance
handling throughout the workflow.
