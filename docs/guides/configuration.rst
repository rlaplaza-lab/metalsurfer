Configuration guide
===================

:class:`~metalsurfer.AdsorptionConfig` holds the settings for substrate prep,
screening, Bayesian search, and saturation. This page covers the choices most
people change. Every field is listed in :doc:`../api/config`.

What material?
--------------

Set ``material_type`` on the same config used for
:func:`~metalsurfer.surface_prep.prepare_substrate` and the campaign call.
It must match the structure you prepared.

``slab``
   Flat crystal surface (vacuum along *z*). Single-crystal facets, adatom
   slabs, and published POSCARs.

``nanoparticle``
   Finite cluster in a box (no periodic images). Hand-built clusters often
   keep the input geometry with ``slab_relaxation_mode="none"``.

``porous``
   Fully periodic framework (MOFs, zeolites). Load from CIF, then
   ``prepare_substrate``.

``surface_type`` on ``run_*`` is only the results folder name
(``results_{surface_type}/``). Physics come from ``material_type``.

How many placements?
--------------------

On a production GPU, leave these unset (``None``):

- ``num_placements`` — standard screening
- ``bo.initial_random``, ``bo.batch_size`` — Bayesian batches (nest under
  ``bo:``; flat ``bo_*`` keys are rejected)

Metalsurfer sizes the batch from available GPU memory. Demos and CI use small
explicit integers instead. If a run runs out of memory, raise
``autobatcher_max_memory_padding`` (default ``0.5``) to leave more headroom.

A molecule that can split (e.g. H₂)
------------------------------------

.. code-block:: python

   config = AdsorptionConfig(
       material_type="slab",
       enable_dissociative_placement=True,
       skip_topology_check=True,
       seed=42,
   )

- ``enable_dissociative_placement=True`` — place H₂ over hollow/bridge site
  pairs so atoms can separate on the surface.
- ``skip_topology_check=True`` — keep the result after relaxation even when
  the molecule has split.

The reference energy is still the isolated molecule, so :math:`E_\mathrm{ads}`
can be positive after dissociation. See
``examples/h2_ru_slab_binding_energy.py``.

Settings people mix up
----------------------

- Raising ``fmax`` alone does not soften the post-relax force filter — also
  raise ``max_force_convergence``.
- ``bo.total_budget`` counts acquisition **batches**, not total evaluations
  (see the Bayesian section below).
- Bayesian mode is chosen by calling ``run_*_bo`` (or YAML ``campaign: *_bo``),
  not by a config flag such as ``bo_enabled``.
- CSV exports are lean by default. Set ``export_placement_provenance=True``
  for full placement and settings columns. Reloading a lean row requires
  pose, energies, and identity columns; a partial ``ctx_*`` /
  ``initial_*`` set raises.

Where molecules sit
-------------------

Leave ``site_generator="auto"``: topology sites on slabs and nanoparticles,
Voronoi sites in porous frameworks.

Set ``site_generator="adaptive_grid"`` or ``"rolling_probe"`` only when you
want near-atom sampling on stepped surfaces or MOF **pore walls**. Keep
``auto`` (Voronoi) for MOF **pore centres**. Plugin knobs and demo defaults:
:doc:`architecture`.

How a starting pose is accepted
-------------------------------

A candidate is placed over a site and kept when every adsorbate–surface pair
clears a covalent floor: the larger of ``min_initial_distance`` (1.5 Å) and
``covalent_sum * min_contact_ratio`` (default ratio 0.8). The 1.5 Å floor
is what limits hydrogen and atoms with an unknown radius. The ratio grows
with the atoms involved, so it is the limit for typical C, N, O, and metal
pairs. A start that only just misses that window is moved into it
(``placement_distance_recovery=True``): first along the surface normal, then
by a small rigid shift.

Knobs worth changing:

- **Packing.** Raise ``min_contact_ratio`` or ``min_initial_distance`` when
  starts are still overlapping. Lower them when bulky adsorbates are thrown
  out before they can relax.
- **Starts that sit too high.** Set ``max_initial_distance`` (Å) to reject
  poses that begin farther from the surface than that.
- **Van der Waals clashes.** ``reject_vdw_overlaps=True`` adds a second
  reject on tabulated van der Waals radii. ``vdw_overlap_scale`` (default
  1.0) multiplies those radii; values above 1 leave a larger gap.
- **How the molecule touches.** ``strict_initial_placement=True`` requires
  the closest pair within ``max_closest_approach`` (3.0 Å) and at least
  ``min_contact_atoms`` atoms inside ``contact_distance_threshold`` (2.5 Å).
  ``require_multiple_contact=True`` asks for two or more of those contacts
  at similar distances, which suits flat or chelating adsorbates.
- **Drawn pose, unchanged.** ``placement_distance_recovery=False`` keeps or
  drops the pose exactly as sampled.
- **Which atoms face the surface.** Leave
  ``adaptive_parallel_fraction=True`` so flat aromatics mix parallel poses
  with binder-down poses. A SMILES atom-map tag (``[O:1]``) restricts the
  binder-down atoms to the tagged set.

Field details: :doc:`../api/config`.

How long a Bayesian search runs
-------------------------------

Total placement evaluations after batch sizes are resolved::

   bo.initial_random + bo.total_budget * bo.batch_size

``bo.total_budget`` is the number of acquisition batches after the initial
random batch. Example: about 300 evals with batch size 16 and initial random
16 → ``bo.total_budget = (300 - 16) // 16``. After sizes resolve,
:func:`~metalsurfer.config.resolved_bo_eval_budget` returns the total count.

Nest BO settings under ``bo`` / ``bo.transfer`` (flat ``bo_*`` keys are
rejected)::

   from metalsurfer import AdsorptionConfig, BOConfig, BOTransferConfig

   config = AdsorptionConfig(
       bo=BOConfig(
           initial_random=16,
           batch_size=16,
           total_budget=18,
           transfer=BOTransferConfig(enabled=True),
       ),
   )

   # YAML:
   # config:
   #   bo:
   #     initial_random: 16
   #     batch_size: 16
   #     total_budget: 18
   #     transfer:
   #       enabled: true

Call :func:`~metalsurfer.run_adsorption_bo` or
:func:`~metalsurfer.run_saturation_bo` (or YAML ``campaign: adsorption_bo`` /
``saturation_bo``). A saturation run repeats that budget on every coverage
step. How those steps reuse earlier placements is below. See
:doc:`yaml_campaigns` and :doc:`../api/campaigns`.

Covering a surface
------------------

Call :func:`~metalsurfer.run_saturation` or
:func:`~metalsurfer.run_saturation_bo`. Each step screens the current slab,
commits the placements that still bind, appends them to the slab, and
repeats. Ranking and the stop use the reservoir score

.. math::

   \Omega = E_\mathrm{ads} - k_B T \ln(a_i p / p^\circ)

with defaults 298.15 K, 1 bar, and activity 1, so :math:`\Omega` matches
:math:`E_\mathrm{ads}` until the reservoir changes. A step commits while
:math:`\Omega < 0`. It stops when the best remaining placement is at or above
zero, the pool is empty, or ``saturation_max_steps`` is reached. Written
energies stay :math:`E_\mathrm{ads}`.

**One adsorbate after another.** With ``multi_molecule_saturation=False``
(the default), each molecule in the list covers the surface on its own, in
order.

**Molecules competing.** ``multi_molecule_saturation=True`` screens every
adsorbate on the same slab each step and advances the one with the lowest
:math:`\Omega`. With one molecule in the list the flag is ignored and the
single-molecule loop runs.

**n-tuplet mode.** ``saturation_molecules_per_step`` (default ``1``) is how
many adsorbates one step places together. Above 1, each trial is one initial
structure with exactly that many adsorbates: CPU placement builds clash-free
packs (``num_placements`` joint configs), TorchSim relaxes every pack with all
members together, and the step commits the best binding config. There is no
single-adsorbate MLIP screen and no partial-tuplet or single-winner fallback.
Overlapping packs are rejected when ``placement_clash_descent`` is off;
otherwise near-misses are rescued by clash descent (substrate contacts use
radius sums; packed adsorbates use ``connectivity_multiplier`` on the radius
sum, then re-checked with the shared disconnect gate). Stored
``energy_adsorption`` is **per molecule**
(:math:`E_\mathrm{ads,total}/n`); composite totals stay on
``energy_adslab`` / ``energy_adsorbate``. The stop uses
:math:`\Omega_\mathrm{tuplet}` (equivalently :math:`\Omega/n` at default
reservoir conditions). Once ``num_placements`` is known it is divided by the
tuplet size, as are ``bo.initial_random`` and ``bo.batch_size`` (each eval is
an *n*-body relax). With BO on a single adsorbate, acquisition still scores
single-site features but labels use :math:`\Omega/n` from each joint eval.
Competitive multi-molecule n-tuplet enumerates every species composition of
size *n*, funds each species' pure pack first then mixed packs
(largest-remainder shares; a short ``num_placements`` can leave mixtures
unfunded), and ranks packs by :math:`\Omega_\mathrm{tuplet}`; joint BO is
single-species only and raises if combined with ``multi_molecule_saturation``.

**How Bayesian search uses earlier placements.**
:func:`~metalsurfer.run_saturation_bo` ranks with :math:`\Omega`. Sequential
saturation trains the surrogate on electronic :math:`E_\mathrm{ads}`; joint
n-tuplet BO (single adsorbate) labels with :math:`\Omega/n`. Failed poses enter
as penalties when ``bo.include_failure_negatives`` is on (the default).
``bo.transfer`` is on by default, so the next coverage step starts from
evaluations already made for that molecule:

- The last ``bo.transfer.prior_step_window`` steps are reused (default 2).
  Set it to ``None`` to keep the full history. Older steps inside that
  window count less (``bo.transfer.recency_lengthscale``).
- Candidates near a molecule already on the slab get larger predictive
  ``sigma`` from the same ``connectivity_multiplier`` clash cutoff used by
  placement and n-tuplet packing, so EI/LCB sample the legal interaction
  shell without evaluating doomed overlaps.
- Each molecule keeps its own history. When the transferred model fits the
  current step worse than a fit on that step alone, transfer turns off for
  the rest of the step.

Leave ``bo.transfer.mode="weighted"``. ``"cumulative_refit"`` retrains on the
pooled history when every retained step should enter the fit.
``bo.transfer.enabled=False`` fits each step on the poses evaluated in that
step. New poses must stay disconnected from adsorbates already on the slab
(``connectivity_multiplier``, default 1.3 — lenient enough for lateral
contacts outside the covalent shell; in-plane site clearance uses
``min_adsorbate_separation``, default 1.5 Å).

**Moving the stop line.** Coverage ends at :math:`\Omega \ge 0`.
``saturation_omega_shift`` (:math:`\delta`, eV) ranks and stops on
:math:`\Omega' = \Omega - \delta` and leaves the stored
:math:`E_\mathrm{ads}` unchanged.

- A positive :math:`\delta` continues the run past the default cutoff. With
  ``saturation_omega_shift=0.15`` a placement still commits while
  :math:`\Omega` is below +0.15 eV.
- A negative :math:`\delta` stops the run while adsorption is still
  favorable. With ``saturation_omega_shift=-0.10`` the step stops once the
  best :math:`\Omega` is −0.10 eV or higher.
- One number applies to every molecule. A sequence follows the molecule
  list, so one species can keep adsorbing after another has stopped.

Change ``saturation_temperature``, ``saturation_pressure``, and
``saturation_activities`` when the reservoir itself is different (higher
temperature, pressure, or activity makes :math:`\Omega` more negative). Use
the shift when you want a fixed energy offset on an otherwise unchanged
reservoir. ``None`` or ``0`` leaves the zero threshold in place.

**Disk.** ``saturation_save_all_placements=False`` keeps the committed
structure each step and skips the full placement dump.

Competitive water + OH⁻ on rutile TiO₂(110), including activities:
``examples/water_oh_rutile_saturation.py``.

Which energy model?
-------------------

Relaxations use FairChem UMA through TorchSim:

- ``model_name`` (default ``"uma-s-1p2"``) — checkpoint.
- ``task_name`` (default ``"oc25"``) — energy/force head.

Keep the pair matched: current 1p2 checkpoints use ``oc25``; older 1p1
checkpoints used ``oc20``.

Relax the surface, or keep a published structure
------------------------------------------------

Prep relaxes and freezes the substrate before the campaign. During adsorption,
only the adsorbate and any unfrozen substrate atoms move.

- **Default:** the whole substrate is frozen after prep.
- **Published / pre-relaxed structure:** set
  ``slab_relaxation_mode="none"`` so ionic positions stay fixed:

  .. code-block:: python

     config = AdsorptionConfig(
         material_type="slab",
         slab_relaxation_mode="none",
         seed=42,
     )

- **Partial surface mobility:** ``relax_top_layer=True`` on prep (see
  :doc:`surface_engineering`).

Further reading
---------------

- Full parameter list: :doc:`../api/config`
- Substrate prep: :doc:`surface_engineering` and :doc:`../api/surface_prep`
- Campaign entry points: :doc:`../api/campaigns`
- Site plugins and placement internals: :doc:`architecture`
