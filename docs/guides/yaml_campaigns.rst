YAML campaigns
==============

Metalsurfer can load a campaign document from YAML and dispatch the same four
run modes as the Python APIs. This page covers the file structure, runnable
demos under ``examples/``, and the limits of the YAML path compared with
calling ``run_*`` directly.

How to run
----------

Requires the MLIP stack (``pip install -e ".[mlip]"``). From the project root::

   from metalsurfer import load_campaign_yaml, run_campaign

   document = load_campaign_yaml("examples/ethene_ru_slab_binding_energy.yaml")
   result = run_campaign(document)

Or use the demo runner::

   python examples/run_campaign_yaml.py examples/ethene_ru_slab_binding_energy.yaml

The default for :func:`~metalsurfer.run_campaign` is ``skip_existing=True``
(skip molecules already listed in result CSVs). Pass ``skip_existing=False``
or delete the results directory to force a fresh run. The demo runner
``examples/run_campaign_yaml.py`` also locks best E_ads to a QC band.

Limitations
-----------

YAML covers the same four run modes, with these limits:

- No package CLI — load with :func:`~metalsurfer.load_campaign_yaml` and
  :func:`~metalsurfer.run_campaign`, or run
  ``examples/run_campaign_yaml.py``.
- Substrate from file only — use ``bulk_id`` or ``slab_file``. Hand-built
  clusters and custom ASE structures need the Python API.
- Molecules are an inline ``{smiles, name}`` list — no CSV path.
- Only ``skip_existing`` is available as a ``run_campaign`` option.
- The YAML API itself has no post-run validation hooks; the demo runner
  ``examples/run_campaign_yaml.py`` adds best-E_ads locks for known demos.
- Still needs the MLIP stack (and usually a GPU).

For custom ASE construction, molecule CSVs, or result checks, use
``prepare_substrate`` + ``run_*`` instead. See :doc:`quickstart` and
``examples/``.

Document structure
------------------

Top-level keys:

================== ============================================================
Key                Required
================== ============================================================
``campaign``       Yes — one of ``adsorption``, ``adsorption_bo``,
                   ``saturation``, ``saturation_bo``
``surface_type``   Yes — results folder label (``results_{surface_type}/``)
``substrate``      Yes — exactly one of ``bulk_id``, ``slab_file``, or ``slab``
``molecules``      Yes — non-empty list of ``{smiles, name}``
``config``         No — maps to :class:`~metalsurfer.AdsorptionConfig` fields
                   (including ``site_generator``, Voronoi window knobs,
                   fingerprint clustering, and reservoir / Ω-shift settings)
================== ============================================================

``campaign`` selects the runner:

================== ==========================================
``campaign`` value Python entry point
================== ==========================================
``adsorption``     :func:`~metalsurfer.run_adsorption`
``adsorption_bo``  :func:`~metalsurfer.run_adsorption_bo`
``saturation``     :func:`~metalsurfer.run_saturation`
``saturation_bo``  :func:`~metalsurfer.run_saturation_bo`
================== ==========================================

Substrate
~~~~~~~~~

Allowed keys match :func:`~metalsurfer.surface_prep.prepare_substrate` (see
:doc:`surface_engineering`): ``bulk_id``, ``slab_file``, ``slab``,
``miller_indices``, ``supercell``, alloy / adatom knobs, ``align``,
``slab_relaxation_*``, ``adatom_relaxation_*``, ``relax_top_layer``,
``freeze_symbols``, ``top_layer_tolerance``.

Unknown substrate keys raise. ``miller_indices`` and ``supercell`` must be
3-element lists (parsed as tuples). From a YAML **file**, use ``bulk_id`` or
``slab_file``; ``slab`` exists for programmatic
:func:`~metalsurfer.campaign_schema.parse_campaign_dict` use with an ASE
object, not for serializing atoms into YAML text.

Config
~~~~~~

``config:`` maps onto :class:`~metalsurfer.AdsorptionConfig`. Put Bayesian
hyperparameters under a nested ``bo:`` block (and optional ``bo.transfer:``);
flat ``bo_*`` / ``bo_transfer_*`` keys are rejected. Field recipes:
:doc:`configuration`; full reference: :doc:`../api/config`.

Demo examples
-------------

Demo-scale YAML files live under ``examples/`` (run from the project root).
They rely on library defaults and use small ``num_placements`` for speed.
Omit ``num_placements`` for production-quality GPU-autotuned budgets.
Production templates: ``scripts/campaigns/``.
Schema smoke fixtures (tiny steps, not intended as physics demos):
``tests/fixtures/campaigns/``.

Ethene on Ru(0001) (standard adsorption)::

   python examples/run_campaign_yaml.py examples/ethene_ru_slab_binding_energy.yaml

.. literalinclude:: ../../examples/ethene_ru_slab_binding_energy.yaml
   :language: yaml

H₂ on Ru(0001) with dissociative placements::

   python examples/run_campaign_yaml.py examples/h2_ru_slab_binding_energy.yaml

.. literalinclude:: ../../examples/h2_ru_slab_binding_energy.yaml
   :language: yaml

CO₂ in a MOF (``slab_file`` + porous)::

   python examples/run_campaign_yaml.py examples/co2_mof_binding_energy.yaml

.. literalinclude:: ../../examples/co2_mof_binding_energy.yaml
   :language: yaml

Water on Cu(111) with nested ``bo:`` (``adsorption_bo``)::

   python examples/run_campaign_yaml.py examples/water_cu111_adsorption_bo.yaml

.. literalinclude:: ../../examples/water_cu111_adsorption_bo.yaml
   :language: yaml

Slim ethane / Cu(111) saturation::

   python examples/run_campaign_yaml.py examples/ethane_cu_saturation.yaml

.. literalinclude:: ../../examples/ethane_cu_saturation.yaml
   :language: yaml

API reference for ``load_campaign_yaml`` / ``run_campaign``:
:doc:`../api/campaigns`.
