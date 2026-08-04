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
Set ``write_settings=False`` to suppress it.

BO vs non-BO is chosen by **which function you call** (or the YAML
``campaign`` field below)—not by a flag on
:class:`~metalsurfer.AdsorptionConfig`. Config only holds nested BO
hyperparameters (``config.bo`` / ``config.bo.transfer``). Legacy flat
``bo_*`` constructor and YAML keys still fold into those nested objects.

YAML campaigns
--------------

Load a campaign document and dispatch with the Python API
(see ``scripts/campaigns/`` and ``tests/fixtures/campaigns/``)::

   from metalsurfer import load_campaign_yaml, run_campaign

   document = load_campaign_yaml("path/to/campaign.yaml")
   result = run_campaign(document)

The top-level ``campaign`` key selects the runner:

================== ==========================================
``campaign`` value Python entry point
================== ==========================================
``adsorption``     :func:`~metalsurfer.run_adsorption`
``adsorption_bo``  :func:`~metalsurfer.run_adsorption_bo`
``saturation``     :func:`~metalsurfer.run_saturation`
``saturation_bo``  :func:`~metalsurfer.run_saturation_bo`
================== ==========================================

``config:`` maps onto :class:`~metalsurfer.AdsorptionConfig` fields. Prefer
a nested ``bo:`` block for Bayesian hyperparameters; flat ``bo_*`` keys are
still accepted and folded. Do not put a ``bo_enabled`` key there—it is not
a config field.

.. autofunction:: metalsurfer.run_campaign

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
