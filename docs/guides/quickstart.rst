Quick Start
===========

Core idea
---------

Give Metalsurfer a material structure (slab, nanoparticle, or porous
framework) and one or more molecules as SMILES. Prep equilibrates and freezes
the substrate; the library places conformers, relaxes them with an MLIP, and
ranks by adsorption energy.

Four campaign functions cover the usual workflows:

- :func:`~metalsurfer.run_adsorption` — standard screening
- :func:`~metalsurfer.run_adsorption_bo` — Bayesian placement search
- :func:`~metalsurfer.run_saturation` — sequential coverage
- :func:`~metalsurfer.run_saturation_bo` — Bayesian saturation

Set ``material_type`` on :class:`~metalsurfer.AdsorptionConfig` to match the
substrate (``slab``, ``nanoparticle``, or ``porous``).
``surface_type`` on ``run_*`` is only the results folder name
(``results_{surface_type}/``).


Installation
------------

Requires **Python 3.12 or newer**.

From PyPI (core library; MLIP stack for campaigns)::

   pip install metalsurfer
   pip install "metalsurfer[mlip]"

Editable install from a clone (core dependencies only — library import and
CPU-only workflow tests):

.. code-block:: bash

   pip install -e .

**Running examples, scripts, or any ``run_*`` campaign requires the MLIP stack:**

.. code-block:: bash

   pip install -e ".[mlip]"

For TorchSim/FairChem-backed relaxation plus the developer toolchain:

.. code-block:: bash

   pip install -e ".[mlip,dev]"

See :doc:`development` for linting, type checking, and CI parity.

To also install the documentation build dependencies:

.. code-block:: bash

   pip install -e ".[docs]"

Then build HTML locally with::

   cd docs && make html

Runnable Examples
-----------------

Demos under ``examples/`` cover nanoparticle (including Ru₅₅), porous, slab,
dissociative H₂ (slab and cluster), competitive saturation, and Bayesian
workflows:

.. code-block:: bash

   python examples/ethene_pt12_binding_energy.py
   python examples/ethene_ru55_binding_energy.py
   python examples/h2_pt13_binding_energy.py
   python examples/co2_mof_binding_energy.py
   python examples/ethene_ru_slab_binding_energy.py
   python examples/h2_ru_slab_binding_energy.py
   python examples/water_oh_rutile_saturation.py
   python examples/camphor_cu111_binding_energy.py   # BO benchmark; ~15 GB GPU

An additional HPC-scale saturation script
(``bipyridine_au111_defects_saturation_raw.py``) lives under ``examples/`` and
``scripts/`` but is not a quick demo.


Standard Screening
------------------

Use :func:`~metalsurfer.run_adsorption` when your script already has the
molecule list in memory and you want a typed
:class:`~metalsurfer.BindingCampaignResult` back.

.. code-block:: python

   from metalsurfer import AdsorptionConfig, prepare_substrate, run_adsorption

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
       results_dir="results_Ru0001",
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
       surface_type="Ru0001",
   )

   print(result.mode)                # "non_bo"
   print(result.total_configurations)
   for summary in result.molecule_summaries:
       print(summary.molecule, summary.best_adsorption_energy)

You can also pass a CSV path instead of an in-memory list (same outputs; optional ``smiles,molecule`` header row supported):

.. code-block:: python

   result = run_adsorption(
       slab=slab,
       molecules="molecules.csv",
       config=config,
       surface_type="Ru0001",
   )

   print(result.format_summary(
       title="Binding summary",
       results_dir="results_Ru0001",
   ))

By default, ``skip_existing=True`` skips molecules already listed in
``adsorption_energies_detailed.csv`` (in-memory lists and CSV paths). Delete
the results directory or pass ``skip_existing=False`` to force a fresh run.

Campaign APIs accept plain ASE ``Atoms`` or
:class:`~metalsurfer.surface_prep.SlabContainer`. Prepare the structure first
with :func:`~metalsurfer.surface_prep.prepare_substrate` (or
``slab_relaxation_mode="none"`` for a published geometry). Layout conventions:
:doc:`surface_engineering`.

Prefer ``write_settings=True`` (default) so campaigns write ``run_metadata.json``.

YAML campaigns: :doc:`yaml_campaigns`.

Slab
~~~~~

:func:`~metalsurfer.surface_prep.prepare_substrate` equilibrates ions
by default (``slab_relaxation_mode="ionic_only"``), applies bottom-anchored
z-layout, PBC, freeze constraints, and validation:

.. code-block:: python

   from ase.build import fcc111
   from metalsurfer import AdsorptionConfig, prepare_substrate, run_adsorption

   config = AdsorptionConfig(material_type="slab", seed=42)

   slab_atoms = fcc111("Ru", size=(3, 3, 3), vacuum=12.0)
   slab = prepare_substrate(
       slab=slab_atoms,
       config=config,
       results_dir="results_ru111_from_ase",
   )

   result = run_adsorption(
       slab=slab,
       molecules=[("O", "water")],
       config=config,
       surface_type="ru111_from_ase_atoms",
   )

Nanoparticle
~~~~~~~~~~~~~

Minimal Pt₄ snippet below; for dissociative H₂ on a periodic slab see
``examples/h2_ru_slab_binding_energy.py`` (Ru(0001),
``enable_dissociative_placement=True`` + ``skip_topology_check=True``).
The runnable ``examples/ethene_pt12_binding_energy.py`` uses the same workflow with a
12-atom Pt cluster and molecular ethene adsorption:

.. code-block:: python

   from ase import Atoms
   from metalsurfer import AdsorptionConfig, prepare_substrate, run_adsorption

   config = AdsorptionConfig(
       material_type="nanoparticle",
       seed=42,
       slab_relaxation_mode="none",  # hand-built clusters: keep input geometry
   )

   cluster_atoms = Atoms(
       "Pt4",
       positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.2, 0], [3.75, 2.2, 0]],
       cell=[20, 20, 20],
       pbc=False,
   )
   slab = prepare_substrate(
       slab=cluster_atoms,
       config=config,
       results_dir="results_pt4_nanoparticle",
   )

   result = run_adsorption(
       slab=slab,
       molecules=[("C=C", "ethene")],
       config=config,
       surface_type="pt4_nanoparticle",
   )

Dissociative H₂ on a slab
~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``enable_dissociative_placement=True`` for
wall hollow/bridge pair placements and ``skip_topology_check=True`` to skip post-relax
connectivity checks; E_ads still uses molecular E(H₂):

.. code-block:: python

   from metalsurfer import AdsorptionConfig, prepare_substrate, run_adsorption

   config = AdsorptionConfig(
       material_type="slab",
       seed=42,
       enable_dissociative_placement=True,
       skip_topology_check=True,
   )
   slab = prepare_substrate(
       bulk_id="mp-33",
       miller_indices=(0, 0, 1),
       supercell=(2, 2, 1),
       config=config,
       results_dir="results_h2_ru_slab",
   )
   result = run_adsorption(
       slab=slab,
       molecules=[("[H][H]", "H2")],
       config=config,
       surface_type="h2_ru_slab",
   )

Already equilibrated?
~~~~~~~~~~~~~~~~~~~~~

When ionic positions must not change, set
``slab_relaxation_mode="none"`` on *config* and use
:func:`~metalsurfer.surface_prep.finalize_substrate` instead of the full
:func:`~metalsurfer.surface_prep.prepare_substrate` call. For
``material_type="slab"``, this applies bottom-anchored z-layout, PBC,
``FixAtoms``, and validation only (no MLIP relaxation). For nanoparticles and
porous frameworks, z-alignment is skipped; PBC and constraints still apply:

.. code-block:: python

   from metalsurfer import AdsorptionConfig, finalize_substrate

   config = AdsorptionConfig(material_type="slab", slab_relaxation_mode="none", seed=42)
   slab = finalize_substrate(slab_atoms, config)

For step-by-step bulk, alloy, and adatom workflows see :doc:`surface_engineering`.


Bayesian Screening
------------------

Same physical pipeline as standard screening, but placements are chosen by a
surrogate. Use :func:`~metalsurfer.run_adsorption_bo`:

.. code-block:: python

   from metalsurfer import AdsorptionConfig, prepare_substrate, run_adsorption_bo

   config = AdsorptionConfig(
       material_type="slab",
       seed=42,
       # Defaults: gradient_boost surrogate, EI acquisition, autotuned batch sizes
   )

   slab = prepare_substrate(
       bulk_id="mp-33",
       miller_indices=(0, 0, 1),
       config=config,
       results_dir="results_Ru0001_bo",
   )

   result = run_adsorption_bo(
       slab=slab,
       molecules=[("O=C=O", "co2"), ("O", "water")],
       config=config,
       surface_type="Ru0001_bo",
   )

BO knobs live under ``config.bo`` / ``config.bo.transfer``. Budget math and
recipes: :doc:`configuration`. Field reference: :doc:`../api/config`.

Sequential Saturation
---------------------

Saturation mode repeatedly adsorbs the current best configuration onto
the evolving slab until adsorption is no longer favorable or no valid
placements remain.  Use :func:`~metalsurfer.run_saturation`:

.. code-block:: python

   from metalsurfer import (
       AdsorptionConfig,
       MultiMolSaturationRunResult,
       prepare_substrate,
       run_saturation,
   )

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
       results_dir="results_Ru0001_sat",
   )

   campaign = run_saturation(
       slab=slab,
       molecules="molecules.csv",
       config=config,
       surface_type="Ru0001_sat",
   )

   for entry in campaign.runs:
       if isinstance(entry, MultiMolSaturationRunResult):
           print(entry.molecules, entry.n_molecules_at_saturation)
       else:
           print(entry.molecule, entry.n_molecules_at_saturation)

``molecules`` accepts either an in-memory ``(smiles, name)`` list or a CSV path
(there is no default file). With ``skip_existing=True`` (default), molecules
already listed in ``saturation_summary.csv`` are skipped.

Running two molecules at once
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default molecules saturate the surface
one after another (sequential mode). Set ``multi_molecule_saturation=True``
and supply several molecules to make them compete at every step; optionally
combine it with ``saturation_molecules_per_step > 1`` so each step can commit
several placements simultaneously (n-tuplet mode):

.. code-block:: python

   config = AdsorptionConfig(
       material_type="slab",
       seed=42,
       multi_molecule_saturation=True,
       saturation_molecules_per_step=2,
       saturation_temperature=423.15,
       saturation_pressure=1.0,
       saturation_activities=(1.0, 1.0e-3),
   )
   campaign = run_saturation(
       slab=slab,
       molecules=[("O", "water"), ("[OH-]", "hydroxide")],
       config=config,
       surface_type="water_oh_rutile_saturation",
   )

Optional reservoir fields rank/stop on ``Ω = E_ads − k_B T ln(a_i p / p°)``
(SATP defaults when omitted); same knobs for n-tuplet and
:func:`~metalsurfer.run_saturation_bo`. Full demo:
``examples/water_oh_rutile_saturation.py``.

A CPU-friendly undergraduate tutorial (Pt₄ tetrahedron; water and OH⁻
competing from pH 7–14 via ``saturation_activities``) lives at
``scripts/tutorials/water_oh_pt4_saturation.py``::

   python scripts/tutorials/water_oh_pt4_saturation.py

Saturation settings and BO transfer: :doc:`configuration` and
:doc:`../api/config`. Resize the in-plane cell during prep when the adsorbate
is large. BO-guided coverage:
:func:`~metalsurfer.run_saturation_bo` or YAML ``campaign: saturation_bo``.

When printing completion summaries, pass
``write_vasp_inputs=config.write_vasp_inputs`` to
:meth:`~metalsurfer.SaturationCampaignResult.format_completion`.

Defected-surface example:
``examples/bipyridine_au111_defects_saturation_raw.py`` (see also
:doc:`surface_engineering`).
