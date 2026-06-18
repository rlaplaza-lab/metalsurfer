Run-Mode Functions
==================

These are the canonical high-level entry points for adsorption screening.
All are importable directly from :mod:`metalsurfer`.

Substrates must be campaign-ready before calling these functions: correct PBC
for ``AdsorptionConfig.material_type``, slab alignment when applicable, and ASE
``FixAtoms`` from prep.  Use :func:`~metalsurfer.surface_prep.prepare_substrate` or
:func:`~metalsurfer.surface_prep.accept_substrate_for_api` (validation only) as described in
:doc:`../guides/surface_engineering`.

Standard Screening
------------------

.. autofunction:: metalsurfer.run_adsorption

Bayesian Screening
------------------

.. autofunction:: metalsurfer.run_adsorption_bo

Sequential Saturation
---------------------

.. autofunction:: metalsurfer.run_saturation

Bayesian Saturation
-------------------

.. autofunction:: metalsurfer.run_saturation_bo
