Run-Mode Functions
==================

These are the canonical high-level entry points for adsorption screening.
All are importable directly from :mod:`metalsurfer`.

Substrates must be campaign-ready before calling these functions: correct PBC
for ``AdsorptionConfig.material_type``, slab alignment when applicable, and
(typically) ASE ``FixAtoms`` from prep. Freeze policy is set during substrate
preparation only — not via ``AdsorptionConfig`` or these ``run_*`` kwargs.
Omitting FixAtoms is allowed (campaigns warn; the substrate stays fully mobile).
Use :func:`~metalsurfer.surface_prep.prepare_substrate` or
:func:`~metalsurfer.surface_prep.finalize_substrate` as described in
:doc:`../guides/surface_engineering`. For advanced embedders that only need
validation, see :func:`~metalsurfer.surface_prep.accept_substrate_for_api` on
the surface-prep API page.

Prefer ``write_settings=True`` (default) to write ``run_metadata.json``.
``write_metadata`` is a deprecated alias for the same file.

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
