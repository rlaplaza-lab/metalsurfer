Quick Start
===========

Core idea
---------

Metalsurfer is substrate-agnostic: pass any ASE ``Atoms`` object—periodic slab,
fully periodic porous framework, or non-periodic cluster—after optional prep
with :func:`~metalsurfer.surface_prep.prepare_substrate` (equilibration, PBC,
ASE ``FixAtoms``). Supply adsorbates as SMILES; the library builds conformers,
finds adsorption sites (orientation-aware Voronoi/topology hybrid, material-aware via
:attr:`~metalsurfer.AdsorptionConfig.material_type`), deposits candidates with
orientation/height sampling, relaxes with an MLIP, validates geometry, and
ranks by adsorption energy. The four ``run_*`` campaign APIs orchestrate
screening, Bayesian placement search, or sequential saturation on that pipeline.


Installation
------------

Requires **Python 3.12 or newer**.

Core dependencies only (library import and CPU-only workflow tests):

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


Runnable Examples
-----------------

Five quick demos under ``examples/`` cover nanoparticle, porous, slab, dissociative
H₂, and Bayesian workflows. An additional HPC-scale saturation script
(``bipyridine_au111_defects_saturation_raw.py``) lives under ``examples/`` and
``scripts/`` but is not a quick demo:

.. code-block:: bash

   python examples/ethene_pt12_binding_energy.py
   python examples/co2_mof_binding_energy.py
   python examples/ethene_ru_slab_binding_energy.py
   python examples/h2_ru_slab_binding_energy.py
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
``adsorption_energies_detailed.csv`` (in-memory lists and CSV paths). Official
demos pass ``skip_existing=False`` so re-runs always compute.

``surface_type`` is only the output folder name (``results_{surface_type}/``);
physics come from ``AdsorptionConfig.material_type`` and the prepared substrate.

Use :func:`~metalsurfer.run_adsorption_bo` for Bayesian placement search; setting
``bo_enabled=True`` on :class:`~metalsurfer.AdsorptionConfig` with
:func:`~metalsurfer.run_adsorption` emits a warning and has no effect.
Prefer the ``run_*_bo`` entry points — do not toggle ``bo_enabled`` yourself.

Campaign APIs accept plain ASE ``Atoms`` or :class:`~metalsurfer.surface_prep.SlabContainer`,
but the structure must be **campaign-ready** before the call: **equilibrated ionic
positions** (from prep unless ``slab_relaxation_mode="none"``), PBC matching
``AdsorptionConfig.material_type``, adequate cell/vacuum, and (typically) ASE
``FixAtoms`` from prep. Define :class:`~metalsurfer.AdsorptionConfig` first,
build the substrate with ASE, then pass it to
:func:`~metalsurfer.surface_prep.prepare_substrate` via ``slab=``. Layout
conventions are described in :doc:`surface_engineering`.

Prefer ``write_settings=True`` (default) so campaigns write ``run_metadata.json``.
``write_metadata`` is a deprecated alias for the same file.

**Slab** — :func:`~metalsurfer.surface_prep.prepare_substrate` equilibrates ions
by default (``slab_relaxation_mode="ionic_only"``), applies bottom-anchored
z-layout, PBC, freeze constraints, and validation:

.. code-block:: python

   from ase.build import fcc111
   from metalsurfer import AdsorptionConfig, run_adsorption
   from metalsurfer.surface_prep import prepare_substrate

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

**Nanoparticle** — minimal Pt₄ snippet below; for dissociative H₂ on a periodic slab see
``examples/h2_ru_slab_binding_energy.py`` (Ru(0001), ``skip_topology_check=True``).
The runnable ``examples/ethene_pt12_binding_energy.py`` uses the same workflow with a
12-atom Pt cluster and molecular ethene adsorption:

.. code-block:: python

   from ase import Atoms
   from metalsurfer import AdsorptionConfig, run_adsorption
   from metalsurfer.surface_prep import prepare_substrate

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

**Dissociative H₂ on a slab** — ``skip_topology_check=True`` enables hollow-site pair
placements **and** skips post-relax connectivity checks; E_ads still uses molecular E(H₂):

.. code-block:: python

   from metalsurfer import AdsorptionConfig, run_adsorption
   from metalsurfer.surface_prep import prepare_substrate

   config = AdsorptionConfig(
       material_type="slab",
       seed=42,
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

**Already equilibrated?** When ionic positions must not change, set
``slab_relaxation_mode="none"`` on *config* and use
:func:`~metalsurfer.surface_prep.finalize_substrate` instead of the full
:func:`~metalsurfer.surface_prep.prepare_substrate` call. For
``material_type="slab"``, this applies bottom-anchored z-layout, PBC,
``FixAtoms``, and validation only (no MLIP relaxation). For nanoparticles and
porous frameworks, z-alignment is skipped; PBC and constraints still apply:

.. code-block:: python

   from metalsurfer.surface_prep import finalize_substrate

   config = AdsorptionConfig(material_type="slab", slab_relaxation_mode="none", seed=42)
   slab = finalize_substrate(slab_atoms, config)

For step-by-step bulk, alloy, and adatom workflows see :doc:`surface_engineering`.


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

BO knobs live on :class:`~metalsurfer.AdsorptionConfig`; see
:doc:`../guides/configuration` (budget math and recipes) and
:doc:`../api/config` (full field reference — Bayesian optimization).
Remember ``bo_total_budget`` is acquisition batches; after sizes resolve, call
:func:`~metalsurfer.resolved_bo_eval_budget` for the total evaluation count.

Sequential Saturation
---------------------

Saturation mode repeatedly adsorbs the current best configuration onto
the evolving slab until adsorption is no longer favorable or no valid
placements remain.  Use :func:`~metalsurfer.run_saturation`:

.. code-block:: python

   from metalsurfer import (
       AdsorptionConfig,
       MultiMolSaturationRunResult,
       run_saturation,
   )
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

Important saturation behaviors:

- Prep equilibrates the substrate before campaigns; adsorption respects ASE
  ``FixAtoms`` from prep. See :doc:`surface_engineering` and
  :doc:`configuration`.
- Resize in-plane supercells during prep
  (``auto_resize_substrate_for_molecule``) before calling campaign APIs.
- Use :func:`~metalsurfer.run_saturation_bo` for BO-guided saturation.
- Saturation-specific config fields (``saturation_*``, ``multi_molecule_saturation``,
  ``bo_transfer_*``): :doc:`configuration` and :doc:`../api/config`.
- When printing completion summaries, pass
  ``write_vasp_inputs=config.write_vasp_inputs`` to
  :meth:`~metalsurfer.SaturationCampaignResult.format_completion`.

A full defected-surface saturation example (fixed substrate, adatom prep) lives
under ``examples/bipyridine_au111_defects_saturation_raw.py``; see also
:doc:`surface_engineering`.
