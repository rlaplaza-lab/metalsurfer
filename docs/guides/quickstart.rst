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

To also install the documentation build dependencies:

.. code-block:: bash

   pip install -e ".[docs]"


Standard Screening
------------------

Use :func:`~metalsurfer.run_adsorption` when your script already has the
molecule list in memory and you want a typed
:class:`~metalsurfer.BindingCampaignResult` back.

.. code-block:: python

   from metalsurfer import AdsorptionConfig, create_slab_from_bulk, run_adsorption

   config = AdsorptionConfig(
       material_type="slab",
       seed=42,
       num_conformers=8,
       num_placements=80,
   )

   slab = create_slab_from_bulk(
       bulk_id="mp-33",
       miller_indices=(0, 0, 1),
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

ASE Atoms objects are accepted directly — no manual wrapping needed:

.. code-block:: python

   from ase.build import fcc111
   from metalsurfer import AdsorptionConfig, run_adsorption

   slab_atoms = fcc111("Ru", size=(3, 3, 3), vacuum=12.0)
   config = AdsorptionConfig(material_type="slab", seed=42)

   result = run_adsorption(
       slab=slab_atoms,
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

   from metalsurfer import AdsorptionConfig, create_slab_from_bulk, run_adsorption_bo

   config = AdsorptionConfig(
       material_type="slab",
       seed=42,
       bo_enabled=True,
       bo_initial_random=20,
       bo_batch_size=10,
       bo_total_budget=60,
       bo_acquisition="lcb",
       bo_surrogate="random_forest",
   )

   slab = create_slab_from_bulk(bulk_id="mp-33", miller_indices=(0, 0, 1))

   result = run_adsorption_bo(
       slab=slab,
       molecules=[("O=C=O", "co2"), ("O", "water")],
       config=config,
       surface_type="Ru001_bo",
   )

Relevant BO configuration fields on :class:`~metalsurfer.AdsorptionConfig`:

- ``bo_initial_random``, ``bo_batch_size``, ``bo_total_budget``
- ``bo_acquisition``: ``"lcb"``, ``"ei"``, or ``"pi"``
- ``bo_surrogate``: ``"random_forest"``, ``"extra_trees"``, ``"gradient_boost"``, or ``"ridge"``
- ``bo_include_failure_negatives`` and ``bo_failure_penalty_*`` for learning from failed placements
- ``bo_transfer_*`` for transfer-enabled saturation runs


Sequential Saturation
---------------------

Saturation mode repeatedly adsorbs the current best configuration onto
the evolving slab until adsorption is no longer favorable or no valid
placements remain.  Use :func:`~metalsurfer.run_saturation`:

.. code-block:: python

   from metalsurfer import AdsorptionConfig, create_slab_from_bulk, run_saturation
   from metalsurfer.io_results import save_saturation_results

   config = AdsorptionConfig(
       material_type="slab",
       seed=42,
       num_conformers=6,
       num_placements=60,
   )

   slab = create_slab_from_bulk(bulk_id="mp-33", miller_indices=(0, 0, 1))

   saturation_results = run_saturation(
       slab=slab,
       molecules="molecules.csv",
       config=config,
       surface_type="Ru001_sat",
   )

   save_saturation_results(saturation_results, surface_type="Ru001_sat", config=config)

   for result in saturation_results:
       print(result.molecule, result.n_molecules_at_saturation)

Important saturation behaviors:

- Auto-resize is only allowed on the first adsorption step.
- When ``bo_enabled=True``, the saturation loop can reuse prior-step BO
  observations through the ``bo_transfer_*`` settings.
- When ``multi_molecule_saturation=True`` and the CSV contains multiple
  molecules, the workflow switches to competitive saturation.
