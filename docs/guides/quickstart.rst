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

Five scripts under ``examples/`` cover nanoparticle, porous, slab, saturation,
and Bayesian workflows:

.. code-block:: bash

   python examples/h2_pt12_binding_energy.py
   python examples/co2_mof_binding_energy.py
   python examples/ethene_ru_slab_binding_energy.py
   python examples/bipyridine_au111_defects_saturation_raw.py
   python examples/camphor_cu111_binding_energy.py


Standard Screening
------------------

Use :func:`~metalsurfer.run_adsorption` when your script already has the
molecule list in memory and you want a typed
:class:`~metalsurfer.BindingCampaignResult` back.

.. code-block:: python

   from metalsurfer import AdsorptionConfig, run_adsorption
   from metalsurfer.surface_prep import prepare_substrate

   config = AdsorptionConfig(
       material_type="slab",
       seed=42,
       num_conformers=8,
       num_placements=80,  # or omit to autotune to GPU parallel capacity
   )

   slab = prepare_substrate(
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
   from metalsurfer import AdsorptionConfig, run_adsorption
   from metalsurfer.surface_prep import create_slab_from_atoms, finalize_substrate

   slab = finalize_substrate(
       create_slab_from_atoms(fcc111("Ru", size=(3, 3, 3), vacuum=12.0)),
       AdsorptionConfig(material_type="slab", seed=42),
   )
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

   from metalsurfer import AdsorptionConfig, run_adsorption_bo
   from metalsurfer.surface_prep import prepare_substrate

   config = AdsorptionConfig(
       material_type="slab",
       seed=42,
       bo_enabled=True,
       # Defaults: ridge surrogate, EI acquisition, autotuned batch sizes
   )

   slab = prepare_substrate(
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

   from metalsurfer import AdsorptionConfig, run_saturation
   from metalsurfer.surface_prep import prepare_substrate

   config = AdsorptionConfig(
       material_type="slab",
       seed=42,
       num_conformers=6,
       num_placements=60,
   )

   slab = prepare_substrate(
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

- **Prep equilibration:** ``slab_relaxation_mode`` (default ``ionic_only``) relaxes
  substrate ionic positions during
  :func:`~metalsurfer.surface_prep.prepare_substrate`. The returned substrate is
  the optimized reference for ``E(slab)``.
- **Adsorption freeze:** prep kwargs write ASE ``FixAtoms`` during
  :func:`~metalsurfer.surface_prep.prepare_substrate` (default: entire substrate
  frozen). Placement relaxation reads those constraints only. See
  :doc:`surface_engineering` for ``relax_top_layer`` / ``freeze_symbols``.
  Saturation pins ``base_slab`` at campaign start. Compare structures to the
  matching prep snapshot (e.g. ``clean_slab_Au20`` after adatoms).
- In-plane supercell expansion must be done during prep
  (``auto_resize_substrate_for_molecule`` / ``resize_substrate_for_molecule``) before
  calling campaign APIs.
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
