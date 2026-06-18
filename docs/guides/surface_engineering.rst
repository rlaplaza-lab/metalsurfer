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

:func:`~metalsurfer.surface_prep.create_slab_from_bulk`,
:func:`~metalsurfer.surface_prep.create_slab_from_atoms`, and
:func:`~metalsurfer.surface_prep.prepare_substrate` apply z alignment during
**prep** (not inside campaign APIs). Campaign entry points validate geometry
and PBC but do not rewrite constraints or resize the cell.


Campaign-ready substrates
-------------------------

Import all prep helpers from :mod:`metalsurfer.surface_prep` (see
:doc:`../api/surface_prep`). Before ``run_adsorption``, ``run_saturation``, or
related APIs, the substrate must have:

- **Equilibrated ionic positions** — :func:`~metalsurfer.surface_prep.prepare_substrate`
  relaxes the substrate by default (``slab_relaxation_mode="ionic_only"``).
  Campaign APIs assume this optimized reference for ``E(slab)`` and ``E_ads``.
- PBC matching ``AdsorptionConfig.material_type`` (``[T,T,F]`` for slabs,
  ``[T,T,T]`` for porous frameworks, ``[F,F,F]`` for nanoparticles)
- Bottom-anchored slab layout (``min(z) ≈ 0``) when ``material_type="slab"``
- ASE ``FixAtoms`` from :func:`~metalsurfer.surface_prep.apply_surface_constraints`
  (attached by :func:`~metalsurfer.surface_prep.prepare_substrate`; default freezes
  the entire substrate; ``relax_top_layer=True`` is a material-aware shortcut)
- Sufficient in-plane image separation for your adsorbates (use
  :func:`~metalsurfer.surface_prep.resize_substrate_for_molecule` after
  conformer generation when needed)

:func:`~metalsurfer.surface_prep.prepare_substrate` is the recommended
one-call path: equilibrate ions, apply PBC, attach constraints, validate.


One-Call Preparation
--------------------

:func:`~metalsurfer.surface_prep.prepare_substrate` combines slab construction,
alloy substitution, and adatom deposition into a single convenience call:

.. code-block:: python

   from metalsurfer import AdsorptionConfig
   from metalsurfer.surface_prep import prepare_substrate

   config = AdsorptionConfig(material_type="slab", seed=42)

   slab = prepare_substrate(
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

For more control, use the individual helpers from :mod:`metalsurfer.surface_prep`.

**Fast structural modification** (no energy ranking):

.. code-block:: python

   from metalsurfer import AdsorptionConfig
   from metalsurfer.surface_prep import (
       create_slab_from_bulk,
       deposit_adatoms,
       finalize_substrate,
       substitute_alloy,
   )

   config = AdsorptionConfig(material_type="slab")
   slab = create_slab_from_bulk(
       bulk_id="mp-33",
       miller_indices=(0, 0, 1),
       results_dir="results_demo",
   )

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

   slab = finalize_substrate(slab, config)

**Energy-ranked variant selection** (recommended for realistic modified
surfaces):

.. code-block:: python

   from metalsurfer import AdsorptionConfig, setup_single_model
   from metalsurfer.surface_prep import (
       create_slab_from_bulk,
       deposit_adatoms,
       finalize_substrate,
       relax_substrate,
       substitute_alloy,
   )

   config = AdsorptionConfig(material_type="slab")
   slab = create_slab_from_bulk(
       bulk_id="mp-33",
       miller_indices=(0, 0, 1),
       results_dir="results_demo",
   )
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

   slab = relax_substrate(slab, calculator, config, relaxation_mode="ionic_only")
   slab = finalize_substrate(slab, config)

Relaxation presets for slab preparation
---------------------------------------

:func:`~metalsurfer.surface_prep.prepare_substrate`,
:func:`~metalsurfer.surface_prep.create_slab_from_bulk`,
:func:`~metalsurfer.surface_prep.deposit_adatoms`, and
:func:`~metalsurfer.surface_prep.relax_substrate` share relaxation presets:

- ``"none"``: no slab relaxation.
- ``"ionic_only"``: relax atomic positions with fixed cell (default).
- ``"cell_only"``: relax cell with ionic coordinates constrained.
- ``"full"``: relax both ionic coordinates and cell.

Set defaults once on :class:`~metalsurfer.AdsorptionConfig` or pass explicit
``slab_relaxation_*`` / ``adatom_relaxation_*`` kwargs to
:func:`~metalsurfer.surface_prep.prepare_substrate`:

.. code-block:: python

   config = AdsorptionConfig(
       slab_relaxation_mode="full",
       slab_relaxation_optimizer="lbfgs",  # lbfgs, bfgs, fire
       slab_relaxation_fmax=0.03,          # optional, falls back to config.fmax
       slab_relaxation_steps=250,
   )

The ``calculator`` argument is **optional** for both
:func:`~metalsurfer.surface_prep.substitute_alloy` and
:func:`~metalsurfer.surface_prep.deposit_adatoms`:

- Without a calculator: a valid modified slab is created (fast structural
  modification).
- With a calculator: random variants are energy-scored and the
  lowest-energy variant is selected.

:func:`~metalsurfer.surface_prep.prepare_substrate` loads a calculator
automatically when any relaxation stage needs it.

Use separate relaxation presets for bulk slab creation vs adatom deposition
when you want a single full equilibration of the clean surface but only ionic
relaxation after adding adatoms:

.. code-block:: python

   slab = prepare_substrate(
       bulk_id="mp-81",
       miller_indices=(1, 1, 1),
       adatom_symbol="Au",
       adatom_coverage=0.20,
       config=config,
       adatom_relaxation_mode="ionic_only",
   )

For structures loaded from file or ASE ``Atoms``, choose ``slab_relaxation_mode``
explicitly:

- **``"none"``** — keep published or campaign-produced coordinates (MOF CIF,
  paper DFT slabs, graphene-oxide models, saturation intermediate XYZ). Used by
  ``examples/co2_mof_binding_energy.py``, ``examples/camphor_cu111_binding_energy.py``,
  ``scripts/furanics_go*_binding_energy.py``, and
  ``scripts/vanillin_on_h_saturated_ni111.py`` for the loaded slab.
- **``"ionic_only"``** (default) — equilibrate hand-built clusters or
  unequilibrated ``Atoms`` before campaigns (e.g. ``examples/h2_pt12_binding_energy.py``).

``relax_top_layer=True`` on ``prepare_substrate`` controls which substrate atoms
move **during adsorption**, not prep equilibration (e.g. top GO layer or
H-covered Ni surface).

Slab freeze during adsorption and saturation
--------------------------------------------

Prep equilibration and adsorption freeze are **separate stages**:

1. **Prep (``prepare_substrate``):** ``slab_relaxation_mode`` (default
   ``"ionic_only"``) equilibrates substrate **ionic positions** with ASE/MLIP.
   The returned structure is the optimized reference for ``E(slab)``.
2. **Prep (finalize):** ``relax_top_layer``, ``freeze_symbols``, and
   ``top_layer_tolerance`` prep kwargs are written to ASE ``FixAtoms`` via
   :func:`~metalsurfer.surface_prep.apply_surface_constraints`.
3. **Adsorption / saturation:** TorchSim reads those frozen indices from the
   substrate reference (``frozen_indices_from_constraints``). Campaign APIs
   log which substrate atoms are frozen vs free at workflow start.

**Default (``relax_top_layer=False``):** every substrate atom is frozen during
placement relaxation — the standard choice for rigid-surface binding energies.

**Surface relaxation shortcut (``relax_top_layer=True``):** interior atoms stay
fixed; which atoms remain free depends on
:attr:`~metalsurfer.AdsorptionConfig.material_type` and
``top_layer_tolerance``:

+--------------------+---------------------------------------------------------+
| ``material_type``  | Atoms left free during placement relaxation             |
+====================+=========================================================+
| ``"slab"``         | Exposed layer along the slab normal (within tolerance   |
|                    | of maximum height)                                      |
| ``"nanoparticle"`` | Outermost shell (within tolerance of max COM distance)  |
| ``"porous"``       | Pore-wall atoms (closest neighbour per pore void site)  |
+--------------------+---------------------------------------------------------+

Use for workflows where the surface should restructure with the adsorbate
(e.g. graphene oxide slabs, H-saturated surfaces, flexible pore mouths). For
catalyst descriptors and rigid binding energies, keep the default.

**Manual constraints:** attach your own ASE ``FixAtoms`` (or other constraints)
to the substrate before calling campaign APIs, or call lower-level helpers and
finalize with custom constraints instead of ``relax_top_layer``.

**Symbol-specific freeze (``freeze_symbols=[...]``):** only listed elements are
frozen; layer policy is ignored.

Saturation stores ``base_slab`` once after ``prepare_substrate``; indices on that
reference stay fixed as adsorbates accumulate. Earlier adsorbate units may relax
in later steps. After adatom deposition, compare structures to the post-adatom
reference (e.g. ``clean_slab_Au20``), not ``clean_slab`` from before deposition.

In-plane supercell expansion must be done during prep
(``auto_resize_substrate_for_molecule`` / ``resize_substrate_for_molecule``) before
calling campaign APIs.


Material Type
-------------

:attr:`AdsorptionConfig.material_type <metalsurfer.AdsorptionConfig.material_type>`
must be set explicitly.  Valid values:

- ``"slab"`` — in-plane periodic surfaces.
- ``"nanoparticle"`` — non-periodic clusters.
- ``"porous"`` — fully periodic porous frameworks.

This choice affects site generation, adsorption validation, and distance
handling throughout the workflow.
