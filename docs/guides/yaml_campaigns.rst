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
   result = run_campaign(document, skip_existing=False)

Or use the demo runner::

   python examples/run_campaign_yaml.py examples/ethene_ru_slab_binding_energy.yaml

Official demos pass ``skip_existing=False`` so re-runs always compute. The
default for :func:`~metalsurfer.run_campaign` is ``skip_existing=True``.

Limitations
-----------

YAML is a convenience dispatch layer, not a full substitute for the Python
``run_*`` APIs.

- **No CLI.** Load and run only via the Python API
  (:func:`~metalsurfer.load_campaign_yaml` and
  :func:`~metalsurfer.run_campaign`).
- **Substrate sources from a YAML file.** Practical choices are ``bulk_id`` or
  ``slab_file``. Inline ASE ``Atoms`` (``slab=``) cannot be expressed in a
  standalone YAML file. Hand-built nanoparticles, downloaded NOMAD slabs, and
  similar cases need the Python API (for example Pt₁₂ and camphor under
  ``examples/``).
- **Molecules are inline only.** A non-empty list of ``{smiles, name}``
  entries. YAML does not accept a molecules CSV path (the Python ``run_*``
  APIs do).
- **``run_campaign`` kwargs are minimal.** Only ``skip_existing`` is exposed.
  Not available from YAML / ``run_campaign``: ``system_name``,
  ``save_results``, ``write_settings``, ``run_metadata_out``,
  ``process_kwargs``.
- **No post-run validation hooks.** Python demos can assert on E_ads or
  placement provenance; YAML runs stop at campaign results.
- **Same runtime requirements.** Still needs the MLIP stack (and typically a
  GPU). YAML does not change physics or hardware needs.

Prefer ``prepare_substrate`` + ``run_*`` when you need custom ASE
construction, CSV molecule libraries, extra campaign kwargs, or result
validation. See :doc:`quickstart` and the Python scripts under ``examples/``.

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
hyperparameters under a nested ``bo:`` block (and optional ``bo.transfer:``).
Flat ``bo_*`` / ``bo_transfer_*`` keys are rejected. There is no
``bo_enabled`` field — BO mode comes from ``campaign: adsorption_bo`` or
``saturation_bo``. Field recipes: :doc:`configuration`; full reference:
:doc:`../api/config`.

Demo examples
-------------

Demo-scale YAML files live under ``examples/`` (run from the project root).
They use short optimizer steps and ``slab_relaxation_mode: none`` so local
runs finish quickly; raise steps / restore default prep relaxation for
production-quality energies. Production templates: ``scripts/campaigns/``.
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
