Configuration
=============

For workflow context see :doc:`../guides/quickstart`, :doc:`../guides/configuration`,
and :doc:`../guides/surface_engineering`.

.. py:module:: metalsurfer

.. py:class:: AdsorptionConfig

   Configuration for adsorption screening, Bayesian search, and saturation.

   Primary knobs: ``model_name``, ``num_conformers``, ``num_placements``, and
   ``material_type``. For dissociative adsorption (e.g. H₂ → 2H), set
   ``enable_dissociative_placement=True`` and usually ``skip_topology_check=True``
   so connectivity filters allow fragmented adsorbates. Use ``run_*_bo`` (or
   YAML ``campaign: adsorption_bo`` / ``saturation_bo``) for Bayesian placement
   selection; nested ``bo`` (:class:`~metalsurfer.BOConfig`) hyperparameters
   only. Flat ``bo_*`` constructor / YAML keys are rejected—use nested
   ``bo:`` / ``bo.transfer:``.
   Reference energies remain isolated-molecule
   energies; positive :math:`E_\mathrm{ads}` can result when the relaxed adsorbate
   dissociates.

   Source: :mod:`metalsurfer.config`

Field reference
---------------

Every attribute on :class:`~metalsurfer.AdsorptionConfig` is documented once below,
grouped by pipeline stage. Defaults match the installed package version; when in
doubt, inspect ``metalsurfer.config.AdsorptionConfig`` in source.

Material and substrate
~~~~~~~~~~~~~~~~~~~~~~

``material_type``
   **Type:** ``Literal["slab", "nanoparticle", "porous"]`` · **Default:** ``"slab"``

   Selects geometry conventions for site detection, placement validation, and PBC
   handling. ``"slab"`` expects in-plane periodicity with vacuum along *z*;
   ``"nanoparticle"`` is non-periodic; ``"porous"`` is fully periodic (MOFs,
   zeolites). Must match the prepared substrate. See :doc:`../guides/configuration`.

``top_layer_tolerance``
   **Type:** ``float`` · **Default:** ``0.5`` (Å)

   Thickness along the surface normal used to (1) identify top-layer atoms for
   Voronoi / site generation on slabs and (2), when ``relax_top_layer=True`` on
   prep, define the **simple height band** of substrate atoms left free during
   adsorption (distinct from the stepped site-discovery mask). Larger values
   (e.g. ``≈2.1`` Å) free multiple Cu(111) layers on thin multi-layer slabs.

``rough_slab_local_z``
   **Type:** ``bool`` · **Default:** ``True``

   When ``True``, rough (non-planar) slabs use a per-site local *z* reference instead
   of global ``max(z)`` when depositing adsorbates. Improves height sampling on
   stepped or adatom-decorated surfaces.

``symmetry_tolerance``
   **Type:** ``float`` · **Default:** ``0.1`` (Å)

   Cartesian tolerance for symmetry detection when deduplicating symmetrically
   equivalent adsorption sites.

``planar_z_variance_threshold``
   **Type:** ``float`` · **Default:** ``0.01`` (Å²)

   Maximum *z* variance of top-layer atoms for classifying a slab surface as planar.
   Surfaces above this threshold trigger rough-slab placement behavior.

``min_pbc_image_separation``
   **Type:** ``float`` · **Default:** ``8.0`` (Å)

   Minimum in-plane separation between periodic images. Used by
   :func:`~metalsurfer.surface_prep.auto_resize_substrate_for_molecule` and
   :func:`~metalsurfer.surface_prep.resize_substrate_for_molecule` during prep,
   and by :func:`~metalsurfer.surface_prep.validate_substrate` during campaign
   molecule preamble checks (nanoparticle vacuum margins and in-plane supercell
   sizing once conformer diameters are known).

``vacuum_box_size``
   **Type:** ``float`` · **Default:** ``20.0`` (Å)

   Edge length of the cubic simulation cell for isolated conformer generation and
   gas-phase reference energy calculations.

``slab_relaxation_mode``
   **Type:** ``Literal["none", "ionic_only", "cell_only", "full"]`` · **Default:** ``"ionic_only"``

   Controls ASE/MLIP relaxation during :func:`~metalsurfer.surface_prep.prepare_substrate`.
   ``"ionic_only"`` equilibrates substrate ionic positions before campaigns (default);
   ``"none"`` skips prep relaxation (literature slabs, hand-built clusters);
   ``"cell_only"`` / ``"full"`` allow cell degrees of freedom. See
   :doc:`../api/surface_prep`.

``slab_relaxation_optimizer``
   **Type:** ``Literal["lbfgs", "bfgs", "fire"]`` · **Default:** ``"lbfgs"``

   Optimizer for prep-time slab relaxation.

``slab_relaxation_fmax``
   **Type:** ``float | None`` · **Default:** ``None``

   Force convergence threshold for prep relaxation. When ``None``, falls back to
   ``fmax``.

``slab_relaxation_steps``
   **Type:** ``int`` · **Default:** ``200``

   Maximum optimizer steps for prep-time slab relaxation.

Conformers
~~~~~~~~~~

``num_conformers``
   **Type:** ``int`` · **Default:** ``10``

   Number of distinct 3D conformers generated per SMILES before placement
   enumeration. More conformers increase coverage of flexible molecules at higher
   compute cost.

``conformer_sampling``
   **Type:** ``Literal["boltzmann", "cycle", "mixed"]`` · **Default:** ``"cycle"``

   Strategy for selecting which conformers enter the placement loop.
   ``"cycle"`` rotates through conformers deterministically; ``"boltzmann"`` weights
   by relative energy at ``boltzmann_temperature``; ``"mixed"`` combines both.

``boltzmann_temperature``
   **Type:** ``float`` · **Default:** ``300.0`` (K)

   Temperature for Boltzmann-weighted conformer selection when
   ``conformer_sampling`` is ``"boltzmann"`` or ``"mixed"``.

Site detection
~~~~~~~~~~~~~~

``voronoi_probe_radius``
   **Type:** ``float | None`` · **Default:** ``None`` (Å)

   Minimum distance from a framework atom to an accepted Voronoi site. When
   ``None``, derived from covalent radii at runtime. Increase to exclude sites too
   close to pore walls in tight frameworks.

``voronoi_max_site_distance``
   **Type:** ``float | None`` · **Default:** ``None`` (Å)

   Maximum distance from framework atoms for an accessible Voronoi site. When
   ``None``, derived at runtime. Must exceed ``voronoi_probe_radius`` when both are
   set.

``voronoi_site_enrichment``
   **Type:** ``bool`` · **Default:** ``True``

   Enable geodesic ridge subdivision to add denser candidate sites on irregular
   surfaces (especially porous materials and rough slabs).

``voronoi_auto_widen``
   **Type:** ``bool`` · **Default:** ``True``

   When the first Voronoi accessibility window finds no sites, retry detection
   **once** with a wider window (probe × 0.8, max × 1.25). Set ``False`` for strict
   A/B comparisons of explicit ``voronoi_probe_radius`` /
   ``voronoi_max_site_distance`` values.

``site_classification_method``
   **Type:** ``Literal["auto", "distance_ratio", "delaunay"]`` · **Default:** ``"auto"``

   Algorithm for labeling sites as atop, bridge, or hollow.
   ``"auto"`` uses Delaunay triangulation of the slab top layer (recommended for
   catalysis-style sampling) and distance-ratio labeling for nanoparticles and
   porous materials. ``"distance_ratio"`` always uses six-neighbour distance
   ratios. ``"delaunay"`` triangulates the slab top layer (slabs only; falls back
   for other material types).

``site_equivalence_tolerance``
   **Type:** ``float`` · **Default:** ``0.05`` (Å)

   Cartesian tolerance for merging symmetrically or geometrically equivalent sites
   after initial detection.

``hollow_site_dedup_tolerance``
   **Type:** ``float`` · **Default:** ``0.1`` (Å)

   Distance tolerance for deduplicating hollow sites that map to the same
   three-fold coordination pocket.

Placement generation
~~~~~~~~~~~~~~~~~~~~

``num_placements``
   **Type:** ``int | None`` · **Default:** ``None``

   Target number of placement candidates evaluated per molecule (non-BO) or the
   upper bound for BO candidate pools. When ``None``, autotunes at runtime from GPU
   memory probing via TorchSim autobatcher settings. Set explicitly for reproducible
   small demos or fixed budgets on CPU.

``placement_x_range``, ``placement_y_range``
   **Type:** ``tuple[float, float]`` · **Default:** ``(-0.5, 0.5)`` (Å)

   In-plane search radius used only by **distance recovery** after a
   ``too_close`` / ``too_far`` failure (not applied to every successful site-centered
   pose). Equal bounds such as ``(0.0, 0.0)`` disable lateral recovery while leaving
   height recovery on. Widen for bulky adsorbates; keep small for reproducibility.

``placement_z_range``
   **Type:** ``tuple[float, float]`` · **Default:** ``(0.7, 1.25)``

   Lower and upper bounds for initial adsorbate height. When
   ``placement_z_scale_by_covalent_radius`` is ``True``, values are scale factors
   on ``(r_adsorbate + r_surface)``; when ``False``, literal Å offsets above the
   surface reference.

``placement_z_scale_by_covalent_radius``
   **Type:** ``bool`` · **Default:** ``True``

   Derive initial *z* offsets from adsorbate and surface covalent radii (all
   placement paths). Set ``False`` to interpret ``placement_z_range`` as absolute Å.

``placement_distance_recovery``
   **Type:** ``bool`` · **Default:** ``True``

   After a covalent distance failure (``too_close`` / ``too_far``), nudge height
   within the placement *z* window, then try a few deterministic in-plane offsets
   within ``placement_x_range`` / ``placement_y_range``, and revalidate. Does not
   apply to VDW, contact-quality, or adsorbate–adsorbate failures. Set ``False`` to
   keep binary accept/reject only.

``flat_aromatic_parallel_fraction``
   **Type:** ``float`` · **Default:** ``0.5``

   Fraction of flat-aromatic placements oriented parallel (π-stacking) versus
   electronegative-atom-down when ``adaptive_parallel_fraction`` is ``False``.
   ``0.5`` explores both equally.

``adaptive_parallel_fraction``
   **Type:** ``bool`` · **Default:** ``True``

   When ``True`` (default), overrides ``flat_aromatic_parallel_fraction`` with a
   molecule-aware estimate (high for pure aromatics, low for strong EN-down binders).

``placement_filter``
   **Type:** ``Callable[[PlacementSpec], bool] | None`` · **Default:** ``None``

   Optional callback to reject placement specifications before materialization.
   Receives a :class:`~metalsurfer.models.PlacementSpec`; return ``False`` to skip.

``placement_retry_enabled``
   **Type:** ``bool`` · **Default:** ``True``

   Retry failed placement generation with perturbed seeds until
   ``num_placements`` valid specs are found or deficit rounds are exhausted.

``placement_retry_max_attempts``
   **Type:** ``int`` · **Default:** ``8``

   Maximum deficit rounds when ``placement_retry_enabled`` is ``True``. Each
   round requests more specs than the remaining count (yield-aware oversampling)
   rather than one retry per placement slot.

``placement_retry_diversity_seed_increment``
   **Type:** ``int`` · **Default:** ``1000``

   Added to the RNG seed on each placement retry for diversity.

``placement_retry_oversample_max``
   **Type:** ``float`` · **Default:** ``6.0`` · **Valid range:** ``>= 1.0``

   Cap on specs requested per deficit round as a multiple of the remaining
   placement count. Combined with the observed materialization yield so rounds
   stay short while still filling ``num_placements``.

``placement_materialize_workers``
   **Type:** ``int`` · **Default:** ``-2``

   Thread-pool size for per-spec placement materialization (joblib-style
   ``n_jobs``). ``1`` is serial, positive values use that many workers,
   ``-1`` uses all CPUs, and ``-2`` uses all but one CPU. Must not be ``0``.
   Advanced callers that materialize specs directly use
   :func:`~metalsurfer.placement.generators.generate_placements_from_specs`
   (pool size via
   :func:`~metalsurfer.placement.generators.resolve_materialize_workers`).

Initial placement validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``min_initial_distance``
   **Type:** ``float`` · **Default:** ``1.5`` (Å)

   Minimum adsorbate–surface separation at placement time. Rejects structures
   starting too close before relaxation.

``min_contact_ratio``
   **Type:** ``float`` · **Default:** ``0.8`` · **Valid range:** ``[0.5, 1.2]``

   Lower bound on initial contact distance as a fraction of
   ``(r_molecule + r_surface)``. Prevents covalent-overlap starts while allowing
   reasonable approach distances.

``max_initial_distance``
   **Type:** ``float | None`` · **Default:** ``None`` (Å)

   Optional upper bound on initial adsorbate–surface distance. When set, rejects
   placements starting too far from the surface.

``strict_initial_placement``
   **Type:** ``bool`` · **Default:** ``False``

   Enable contact-quality pre-relaxation checks (closest approach, contacting-atom
   count). This is independent of van der Waals overlap rejection; use
   ``reject_vdw_overlaps`` for VDW.

``reject_vdw_overlaps``
   **Type:** ``bool`` · **Default:** ``False``

   Reject placements with van der Waals overlaps (stricter than covalent-radius
   checks). Independent of ``strict_initial_placement``.

``vdw_overlap_scale``
   **Type:** ``float`` · **Default:** ``1.0``

   Scale factor applied to summed VDW radii when testing overlaps. Values ``> 1``
   are stricter; ``< 1`` more lenient.

``max_closest_approach``
   **Type:** ``float`` · **Default:** ``0.8`` (Å)

   Maximum allowed closest-approach distance (Å) between the adsorbate and the
   substrate when ``strict_initial_placement`` or ``require_multiple_contact`` is
   enabled. Rejects placements whose nearest contact is farther than this
   threshold. Distinct from ``contact_distance_threshold``, which only counts
   contacting atoms.

``min_contact_atoms``
   **Type:** ``int`` · **Default:** ``1``

   Minimum number of molecule atoms within ``contact_distance_threshold`` of the
   surface required to accept an initial placement under strict contact checks.

``contact_distance_threshold``
   **Type:** ``float`` · **Default:** ``2.5`` (Å)

   Distance cutoff for counting molecule atoms as surface-contacting during initial
   validation.

``require_multiple_contact``
   **Type:** ``bool`` · **Default:** ``False``

   Require at least ``max(2, min_contact_atoms)`` contacting atoms and reject
   high contact-distance variance among those contacts. Useful for bidentate or
   flat adsorbates.

Relaxation and MLIP
~~~~~~~~~~~~~~~~~~~

``model_name``
   **Type:** ``str`` · **Default:** ``"uma-s-1p2"``

   FairChem/UMA model identifier passed to TorchSim for energy and force
   evaluations during relaxation and reference calculations.

``device``
   **Type:** ``Literal["cuda", "cpu"]`` · **Default:** ``"cuda"``

   Compute device for MLIP calculations. Use ``"cpu"`` when CUDA is unavailable;
   autotuning still applies but batch sizes will be smaller.

``fmax``
   **Type:** ``float`` · **Default:** ``0.05`` (eV/Å)

   Maximum force magnitude for adsorbate–slab and reference-molecule optimization
   convergence. Raising ``fmax`` does **not** relax the post-relaxation force
   reject threshold; set ``max_force_convergence`` as well if you want looser
   acceptance after optimization.

``stage1_steps``, ``stage2_steps``
   **Type:** ``int`` · **Default:** ``50``, ``150``

   Two-stage TorchSim relaxation: coarse then fine optimization of adsorbate–slab
   complexes.

``reference_optimization_steps``
   **Type:** ``int`` · **Default:** ``100``

   Optimizer steps for isolated gas-phase reference molecule calculations used in
   :math:`E_\mathrm{ads} = E_\mathrm{adslab} - E_\mathrm{slab} - E_\mathrm{molecule}`.

``optimize_isolated_sequentially``
   **Type:** ``bool`` · **Default:** ``False``

   Optimize isolated reference molecules one at a time instead of batched. Reduces
   peak GPU memory at the cost of throughput.

``ts_optimizer``
   **Type:** ``Literal["fire", "lbfgs", "bfgs"]`` · **Default:** ``"fire"``

   TorchSim optimizer used during MLIP relaxation.

``steps_between_swaps``
   **Type:** ``int`` · **Default:** ``5``

   Interval for optimizer swap steps inside TorchSim's multi-stage relaxation.

``autobatcher_max_memory_padding``
   **Type:** ``float`` · **Default:** ``0.5`` · **Valid range:** ``[0.1, 1.0]``

   Fraction of GPU memory reserved as headroom when probing parallel batch capacity
   for placement relaxation. Lower values allow larger batches; higher values reduce
   OOM risk.

``autobatcher_max_memory_scaler``
   **Type:** ``float | None`` · **Default:** ``None``

   Optional override for TorchSim memory scaler. When set, also influences
   autotuned ``num_placements`` and BO batch sizes.

``autobatcher_max_atoms_to_try``
   **Type:** ``int | None`` · **Default:** ``None``

   Cap on system size used during TorchSim memory-estimation probes. When ``None``,
   Metalsurfer computes a conservative per-call cap from the current workload.

``saturation_autobatcher_reuse``
   **Type:** ``bool`` · **Default:** ``True``

   In saturation campaigns, reuse a prior step's autobatcher estimate when the slab
   grows only slightly (avoids repeated probing).

``saturation_autobatcher_reuse_growth_atoms``
   **Type:** ``int`` · **Default:** ``32``

   Maximum atom-count increase for which saturation reuses a prior autobatcher
   estimate.

``saturation_autobatcher_reuse_growth_fraction``
   **Type:** ``float`` · **Default:** ``0.1``

   Maximum fractional atom-count increase for autobatcher reuse in saturation.

Post-relaxation validation
~~~~~~~~~~~~~~~~~~~~~~~~~~

``min_interatomic_distance``
   **Type:** ``float`` · **Default:** ``0.5`` (Å)

   Reject relaxed structures with any pair of atoms closer than this distance
   (clash filter after optimization).

``max_force_convergence``
   **Type:** ``float`` · **Default:** ``0.05`` (eV/Å)

   Reject structures whose maximum force remains above this threshold after
   relaxation (failed optimization filter). Independent of ``fmax`` (optimizer
   stop criterion); raise both when intentionally accepting softer convergence.

``binding_distance_threshold``
   **Type:** ``float`` · **Default:** ``4.0`` (Å)

   Post-optimization desorption check: reject if the adsorbate centroid is farther
   than this from the surface. Skipped when ``skip_desorption_check=True``.

``skip_desorption_check``
   **Type:** ``bool`` · **Default:** ``False``

   Disable the post-relaxation adsorbate–surface distance validation. Use when
   legitimate bound states sit at unusually large distances or for debugging.

``enable_dissociative_placement``
   **Type:** ``bool`` · **Default:** ``False``

   Preferred gate for dissociative hollow/site-pair initial placements of
   homonuclear diatomics on slabs and nanoparticles (e.g. H₂ → 2H). Pair with
   ``skip_topology_check=True`` when fragmented post-relax states must pass
   connectivity filters. Descriptor ``fragment_positions`` support replay but
   are omitted from BO feature vectors.

``skip_topology_check``
   **Type:** ``bool`` · **Default:** ``False``

   Disables post-relaxation molecular connectivity / decomposition checks so
   fragmented adsorbates can be retained. Does **not** enable dissociative
   placement—set ``enable_dissociative_placement=True`` for hollow/site-pair
   initial placements. Reference energies remain the isolated molecule;
   positive :math:`E_\mathrm{ads}` can result after dissociation.

``connectivity_multipliers``
   **Type:** ``list[float]`` · **Default:** ``[1.2, 1.3]``

   Covalent-radius multipliers used in connectivity analysis. The workflow tries
   each multiplier in order when testing whether the adsorbate remains intact.

``max_adsorption_energy``
   **Type:** ``float`` · **Default:** ``5.0`` (eV)

   Reject configurations with adsorption energy above this cap (unphysical or
   poorly converged states).

Deduplication
~~~~~~~~~~~~~

``energy_dedup_threshold``
   **Type:** ``float`` · **Default:** ``0.05`` (eV)

   Treat two surviving configurations as duplicates when their adsorption energies
   differ by less than this value.

``rmsd_dedup_threshold``
   **Type:** ``float`` · **Default:** ``0.1`` (Å)

   Additional RMSD threshold for structural deduplication among energy-degenerate
   placements.

Bayesian optimization
~~~~~~~~~~~~~~~~~~~~~

Used by :func:`~metalsurfer.run_adsorption_bo` and
:func:`~metalsurfer.run_saturation_bo` (and YAML ``campaign: adsorption_bo`` /
``saturation_bo``). Those entry points select BO mode; nested ``bo`` /
``bo.transfer`` fields below are hyperparameters only. Flat ``bo_*`` /
``bo_transfer_*`` constructor and YAML keys are rejected.

``bo``
   **Type:** :class:`~metalsurfer.BOConfig` · **Default:** ``BOConfig()``

   Nested Bayesian hyperparameters. Use ``config.bo.*`` in Python; YAML must
   use a nested ``bo:`` / ``bo.transfer:`` block.

``bo.initial_random``
   **Type:** ``int | None`` · **Default:** ``None``

   Number of placements evaluated in the initial random batch before surrogate-guided
   acquisition. When ``None``, autotunes to GPU parallel capacity.

``bo.initial_sampling``
   **Type:** ``Literal["random", "spread", "spread_xyz", "stratified"]`` · **Default:** ``"spread_xyz"``

   Strategy for selecting the initial random batch. ``"spread_xyz"`` uses
   farthest-point sampling in absolute (*x*, *y*, *z*) feature space.

``bo.batch_size``
   **Type:** ``int | None`` · **Default:** ``None``

   Placements per acquisition batch after the initial random phase. When ``None``,
   autotunes to GPU capacity.

``bo.total_budget``
   **Type:** ``int`` · **Default:** ``18``

   Number of **acquisition batches** after the initial random batch—not total
   evaluations. Total BO evaluations (once autotune resolves) is
   ``bo.initial_random + bo.total_budget * bo.batch_size`` (see
   :func:`~metalsurfer.config.resolved_bo_eval_budget`).

``bo.ucb_kappa``
   **Type:** ``float`` · **Default:** ``1.96``

   Exploration parameter for **LCB** acquisition only (``bo.acquisition="lcb"``).
   Ignored for the default ``"ei"`` and for ``"pi"``.

``bo.acquisition``
   **Type:** ``Literal["lcb", "ei", "pi"]`` · **Default:** ``"ei"``

   Acquisition function: lower confidence bound, expected improvement, or probability
   of improvement.

``bo.surrogate``
   **Type:** ``Literal["random_forest", "extra_trees", "gradient_boost", "ridge", "gaussian_process", "ensemble"]`` · **Default:** ``"gradient_boost"``

   Surrogate regressor mapping placement geometry features to adsorption energy.
   ``"gaussian_process"`` does not support sample weights and is incompatible
   with ``bo.transfer.enabled=True``. Transfer-capable surrogates are
   ``random_forest``, ``extra_trees``, ``gradient_boost``, ``ridge``, and
   ``ensemble``.

``bo.candidate_pool_size``
   **Type:** ``int | None`` · **Default:** ``None``

   Optional cap on the number of unexecuted placement specs considered during each
   acquisition step.

``bo.include_failure_negatives``
   **Type:** ``bool`` · **Default:** ``True``

   Train the surrogate on failed placements (generation, optimization, validation)
   using penalty energies so the model learns to avoid bad regions.

``bo.failure_penalty_default``
   **Type:** ``float`` · **Default:** ``10.0`` (eV)

   Penalty energy assigned to failed placements when ``bo.include_failure_negatives``
   is ``True``.

``bo.failure_penalty_overrides``
   **Type:** ``dict[str, float]`` · **Default:** per failure-type map

   Override penalty energies by failure stage (``"generation"``, ``"optimization"``,
   ``"validation"``, ``"energy_cap"``, ``"filter"``) or by generation reason token
   (e.g. ``"too_close"``, ``"vdw_overlap"``, ``"distance_check_failed"``).

``bo.transfer.enabled``
   **Type:** ``bool`` · **Default:** ``True``

   Reuse observations from prior saturation steps when running
   ``run_saturation_bo``. Requires a surrogate that supports sample weights.

``bo.transfer.mode``
   **Type:** ``Literal["weighted", "cumulative_refit"]`` · **Default:** ``"weighted"``

   ``"weighted"`` downweights prior-step rows; ``"cumulative_refit"`` refits on the
   union of prior and current observations.

``bo.transfer.min_step_observations``
   **Type:** ``int`` · **Default:** ``5``

   Minimum current-step observations before prior-step transfer weights apply.

``bo.transfer.weight_cap``
   **Type:** ``float`` · **Default:** ``0.35``

   Maximum total weight contributed by transferred prior-step observations.

``bo.transfer.similarity_lengthscale``
   **Type:** ``float`` · **Default:** ``4.0``

   Length scale for gating prior rows by feature-space similarity to current
   candidates.

``bo.transfer.min_similarity``
   **Type:** ``float`` · **Default:** ``0.05``

   Minimum similarity score for a prior observation to receive non-zero transfer
   weight.

``bo.transfer.trust_patience``
   **Type:** ``int`` · **Default:** ``2``

   Steps to wait before trusting transfer weights when surrogate error is high.

``bo.transfer.mae_tolerance``
   **Type:** ``float`` · **Default:** ``0.0`` (eV)

   Allow transfer when surrogate MAE on held-out current-step points is below this
   tolerance.

``bo.transfer.exploration_fraction``
   **Type:** ``float`` · **Default:** ``0.2``

   Fraction of each BO batch reserved for exploration (random or spread picks) rather
   than pure acquisition.

``bo.transfer.proximity_lengthscale``
   **Type:** ``float`` · **Default:** ``1.0``

   Feature-space decay length for downweighting prior rows near already-executed
   placements in the current step.

``bo.transfer.proximity_floor``
   **Type:** ``float`` · **Default:** ``0.0``

   Minimum sample weight for prior rows after proximity decay.

``bo.transfer.prior_step_window``
   **Type:** ``int | None`` · **Default:** ``2``

   Number of most recent prior saturation steps whose BO memories are eligible for
   transfer. ``None`` uses all prior steps.

``bo.transfer.recency_lengthscale``
   **Type:** ``float`` · **Default:** ``4.0``

   Exponential decay vs step age within the transfer window (``0`` = most recent
   prior step).

``bo.transfer.occupancy_lengthscale``
   **Type:** ``float`` · **Default:** ``1.0``

   Downweight prior rows near the previous step's winning placement site to reduce
   redundant re-exploration.

``bo.transfer.occupancy_floor``
   **Type:** ``float`` · **Default:** ``0.0``

   Minimum transfer modifier at the executed placement site after occupancy decay.

Saturation
~~~~~~~~~~

Saturation behavior is enabled by calling :func:`~metalsurfer.run_saturation` or
:func:`~metalsurfer.run_saturation_bo`. Fields prefixed with ``saturation_`` tune
loop behavior and I/O only.

``multi_molecule_saturation``
   **Type:** ``bool`` · **Default:** ``False``

   When ``True`` and multiple molecules are loaded, each saturation step runs
   competitive adsorption: all adsorbates compete and the best overall
   :math:`E_\mathrm{ads}` wins the step.

``saturation_save_all_placements``
   **Type:** ``bool`` · **Default:** ``True``

   Write every validated placement per step under ``step_{NNN}_placements/`` and
   ``saturation_placements_detailed.csv``. Set ``False`` on large runs to persist
   only per-step best structures.

``save_benchmark_dataset``
   **Type:** ``bool`` · **Default:** ``False``

   Flatten all saturation-step placements into ``adsorption_energies_detailed.csv``
   for benchmarking or ML dataset export.

``export_placement_provenance``
   **Type:** ``bool`` · **Default:** ``False``

   Control richness of ``ml_dataset.csv`` and detailed result CSVs. Default lean
   rows keep identity, the eight ML feature columns (absolute initial pose +
   ``conformer_index`` + quaternion), energies, and ``context_hash``. Set
   ``True`` to also write ``initial_*`` pre-relax placement provenance (site,
   orientation, fragment positions, …) and full ``ctx_*`` computation settings.
   These provenance fields describe the **initial** placement, not the relaxed
   geometry (relaxed structures remain in XYZ/POSCAR).

``saturation_discard_topology_rearrangements``
   **Type:** ``bool`` · **Default:** ``True``

   Before advancing to the next step, discard candidates whose full adsorbate pool
   fails a connectivity-only fragment-count check (inter-adsorbate coupling or
   unexpected splitting). Set ``False`` to rank by :math:`E_\mathrm{ads}` only. Also
   skipped when ``skip_topology_check=True``.

``saturation_max_steps``
   **Type:** ``int | None`` · **Default:** ``None``

   Optional hard cap on saturation loop depth. ``None`` runs until adsorption is
   unfavorable or no valid placements remain.

Reproducibility, strictness, and I/O
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``seed``
   **Type:** ``int`` · **Default:** ``42``

   Master random seed for conformer generation, placement sampling, and BO stochastic
   choices. Thread through the same config object used for prep and campaigns.

``fail_on_missing_reference``
   **Type:** ``bool`` · **Default:** ``False``

   Raise instead of skipping a molecule when an isolated reference energy calculation
   fails.

``fail_on_conformer_failure``
   **Type:** ``bool`` · **Default:** ``False``

   Raise instead of skipping a molecule when conformer generation fails.

``debug_write_initial_placements``
   **Type:** ``bool`` · **Default:** ``False``

   Write ``initial_*.xyz`` files of pre-relaxation placements alongside optimized
   structures under ``xyz_structures/``.

``write_vasp_inputs``
   **Type:** ``bool`` · **Default:** ``False``

   Write POSCAR/INCAR/KPOINTS placement bundles and reference-slab POSCAR files.
   XYZ and CSV outputs are always written regardless of this flag.

``vasp_encut``
   **Type:** ``int`` · **Default:** ``400`` (eV)

   VASP ``ENCUT`` when ``write_vasp_inputs=True``.

``vasp_ediff``
   **Type:** ``float`` · **Default:** ``1e-6`` (eV)

   VASP ``EDIFF`` when ``write_vasp_inputs=True``.

``vasp_ediffg``
   **Type:** ``float`` · **Default:** ``-0.02`` (eV/Å)

   VASP ``EDIFFG`` when ``write_vasp_inputs=True``.

``vasp_nsw``
   **Type:** ``int`` · **Default:** ``100``

   VASP ``NSW`` when ``write_vasp_inputs=True``.

``vasp_kpoints``
   **Type:** ``tuple[int, int, int]`` · **Default:** ``(4, 4, 1)``

   Monkhorst-Pack k-point grid written to KPOINTS when ``write_vasp_inputs=True``.

Helper functions
----------------

.. autofunction:: metalsurfer.config.resolved_bo_eval_budget

.. autofunction:: metalsurfer.config.bo_eval_schedule

.. autofunction:: metalsurfer.config.fold_bo_config

Nested BO types
---------------

.. autoclass:: metalsurfer.BOConfig
   :members:
   :undoc-members:

.. autoclass:: metalsurfer.BOTransferConfig
   :members:
   :undoc-members:
