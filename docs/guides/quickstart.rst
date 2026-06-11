Quick Start
===========

Installation
------------

Requires **Python 3.12 or newer**.

Core dependencies only:

.. code-block:: bash

   pip install -e .

For TorchSim/FairChem-backed relaxation and the developer toolchain:

.. code-block:: bash

   pip install -e ".[mlip,dev]"

See :doc:`development` for linting, type checking, and CI parity.

To also install the documentation build dependencies:

.. code-block:: bash

   pip install -e ".[docs]"


Runnable Examples
-----------------

Three scripts under ``examples/`` cover nanoparticle, porous, and slab workflows:

.. code-block:: bash

   python examples/h2_pt12_binding_energy.py
   python examples/co2_mof_binding_energy.py
   python examples/ethene_ru_slab_binding_energy.py


Standard Screening
------------------

Use :func:`~metalsurfer.run_adsorption` when your script already has the
molecule list in memory and you want a typed
:class:`~metalsurfer.BindingCampaignResult` back.

.. code-block:: python

   from metalsurfer import AdsorptionConfig, prepare_slab, run_adsorption

   config = AdsorptionConfig(
       material_type="slab",
       seed=42,
       num_conformers=8,
       num_placements=80,  # omit for None → autotune to GPU parallel capacity
   )

   slab = prepare_slab(
       bulk_id="mp-33",
       miller_indices=(0, 0, 1),
       config=config,
       results_dir="results_Ru001",
   )

   molecules = [
       ("CC", "ethane"),
       ("C=C", "ethene"),
       ("C#C", "acetylene"),
   ]

   result = run_adsorption(
       slab=slab,
       molecules=molecules,
       config=config,
       surface_type="Ru001",
   )

   print(result.mode)                # "non_bo"
   print(result.total_configurations)
   for summary in result.molecule_summaries:
       print(summary.molecule, summary.best_adsorption_energy)

You can also pass a CSV path instead of an in-memory list:

.. code-block:: python

   result = run_adsorption(
       slab=slab,
       molecules="molecules.csv",
       config=config,
       surface_type="Ru001",
   )

ASE ``Atoms`` objects are accepted directly — no manual wrapping needed.
Slab layout conventions are described in :doc:`surface_engineering`.

.. code-block:: python

   from ase.build import fcc111
   from metalsurfer import AdsorptionConfig, create_slab_from_atoms, run_adsorption

   slab = create_slab_from_atoms(fcc111("Ru", size=(3, 3, 3), vacuum=12.0))
   config = AdsorptionConfig(material_type="slab", seed=42)

   result = run_adsorption(
       slab=slab,
       molecules=[("O", "water")],
       config=config,
       surface_type="ru111_from_ase_atoms",
   )


Bayesian Screening
------------------

Bayesian mode keeps the same physical pipeline and output types, but
replaces exhaustive placement evaluation with surrogate-guided candidate
selection.  Use :func:`~metalsurfer.run_adsorption_bo`:

.. code-block:: python

   from metalsurfer import AdsorptionConfig, prepare_slab, run_adsorption_bo

   config = AdsorptionConfig(
       material_type="slab",
       seed=42,
       bo_enabled=True,
       # Defaults: ridge/ei, autotune batch sizes, 18 acquisition batches
   )

   slab = prepare_slab(
       bulk_id="mp-33",
       miller_indices=(0, 0, 1),
       config=config,
       results_dir="results_Ru001_bo",
   )

   result = run_adsorption_bo(
       slab=slab,
       molecules=[("O=C=O", "co2"), ("O", "water")],
       config=config,
       surface_type="Ru001_bo",
   )

Relevant BO configuration fields on :class:`~metalsurfer.AdsorptionConfig`:

- ``num_placements`` (default ``None``: autotune to GPU parallel capacity at runtime)
- ``bo_initial_random``, ``bo_batch_size`` (default ``None``: autotune to GPU parallel capacity), ``bo_total_budget`` (default ``18``: acquisition batches after the initial random batch)
- Total BO evaluations once auto fields resolve: ``bo_initial_random + bo_total_budget * bo_batch_size``
- ``bo_acquisition``: ``"ei"`` (default), ``"lcb"``, or ``"pi"``
- ``bo_surrogate``: ``"ridge"`` (default), ``"random_forest"``, ``"extra_trees"``, ``"gradient_boost"``, or ``"ensemble"``
- ``bo_include_failure_negatives`` and ``bo_failure_penalty_*`` for learning from failed placements
- ``bo_transfer_*`` for saturation transfer (default weighted mode with 2-step window, recency/occupancy decay)

Sequential Saturation
---------------------

Saturation mode repeatedly adsorbs the current best configuration onto
the evolving slab until adsorption is no longer favorable or no valid
placements remain.  Use :func:`~metalsurfer.run_saturation`:

.. code-block:: python

   from metalsurfer import AdsorptionConfig, prepare_slab, run_saturation

   config = AdsorptionConfig(
       material_type="slab",
       seed=42,
       num_conformers=6,
       num_placements=60,
   )

   slab = prepare_slab(
       bulk_id="mp-33",
       miller_indices=(0, 0, 1),
       config=config,
       results_dir="results_Ru001_sat",
   )

   campaign = run_saturation(
       slab=slab,
       molecules="molecules.csv",
       config=config,
       surface_type="Ru001_sat",
   )

   for result in campaign.runs:
       print(result.molecule, result.n_molecules_at_saturation)

``molecules`` accepts either an in-memory ``(smiles, name)`` list or a CSV path.

Important saturation behaviors:

- **Prep vs adsorption relaxation:** ``slab_relaxation_mode`` applies during
  :func:`~metalsurfer.prepare_slab`` (ASE). Placement relaxation uses
  ``relax_top_layer`` and ``base_slab_for_frozen`` (TorchSim ``FixAtoms``).
  With ``relax_top_layer=False``, indices ``0 .. len(base_slab)-1`` stay fixed;
  earlier adsorbate units on the evolving slab may still relax in later steps.
  When adatoms are deposited, compare structures to the post-adatom reference
  (e.g. ``clean_slab_Au20``), not ``clean_slab`` from before deposition.
- Auto-resize is only allowed on the first adsorption step; if the substrate
  is repeated in-plane, the freeze reference is updated to the full resized slab.
- When ``bo_enabled=True``, the saturation loop can reuse prior-step BO
  observations through the ``bo_transfer_*`` settings.
- When ``multi_molecule_saturation=True`` and multiple molecules are provided
  (in-memory list or CSV), the workflow switches to competitive saturation.
- By default, ``saturation_discard_topology_rearrangements=True`` validates the
  full adsorbate pool on each step candidate using a connectivity-only
  fragment-count check before choosing the best slab for the next step, so
  inter-adsorbate coupling or unexpected splitting is not propagated. Set
  ``False`` to rank by ``E_ads`` only; the guard is skipped when
  ``skip_topology_check=True``.
- By default, runs persist XYZ structures and CSV tables only. Set
  ``write_vasp_inputs=True`` to also write ``vasp_inputs/`` placement bundles
  and reference-slab POSCAR files during surface prep.

A full defected-surface saturation example (fixed substrate, adatom prep) lives
under ``examples/bipyridine_au111_defects_saturation_raw.py``; see also
:doc:`surface_engineering`.
