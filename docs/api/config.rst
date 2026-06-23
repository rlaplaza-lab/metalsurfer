Configuration
=============

.. autoclass:: metalsurfer.AdsorptionConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
   :exclude-members: __post_init__, save_benchmark_dataset

Key Attributes
--------------

.. csv-table::
   :header: "Attribute", "Type", "Default", "Description"
   :widths: 20, 15, 15, 50

   "model_name", "str", "'uma-s-1p1'", "Name of the MLIP model to use for energy calculations"
   "num_conformers", "int", "10", "Number of conformers to generate for each molecule"
   "num_placements", "int | None", "None", "Total placement attempts; None autotunes to GPU parallel capacity at runtime"
   "device", "str", "'cuda'", "Device to use for MLIP calculations ('cuda' or 'cpu')"
   "fmax", "float", "0.05", "Maximum force threshold for optimization convergence (eV/Å)"
   "stage1_steps", "int", "50", "Number of optimization steps in stage 1 (coarse optimization)"
   "stage2_steps", "int", "150", "Number of optimization steps in stage 2 (fine optimization)"
   "reference_optimization_steps", "int", "100", "Number of optimization steps for reference calculations"
   "placement_x_range", "tuple[float, float]", "(-4.0, 4.0)", "Range for x-coordinate placement (Å)"
   "placement_y_range", "tuple[float, float]", "(-4.0, 4.0)", "Range for y-coordinate placement (Å)"
   "placement_z_range", "tuple[float, float]", "(0.7, 1.25)", "Lower/upper scale factors on (r_adsorbate + r_surface) for z placement when placement_z_scale_by_covalent_radius is True; literal Å offsets when False"
   "placement_z_scale_by_covalent_radius", "bool", "True", "Derive z-placement offsets from adsorbate and surface covalent radii (all placement paths)"
   "material_type", "Literal['slab', 'nanoparticle', 'porous']", "'slab'", "Type of material ('slab', 'nanoparticle', or 'porous')"
   "voronoi_probe_radius", "float", "None", "Minimum distance from framework atoms to Voronoi sites (Å); derived from covalent radii when unset"
   "voronoi_max_site_distance", "float", "None", "Maximum distance for accessible Voronoi sites (Å); derived at runtime when unset"
   "voronoi_site_enrichment", "bool", "True", "Enable geodesic ridge subdivision for denser sites"
   "top_layer_tolerance", "float", "0.5", "Slab top-layer thickness along the slab normal (Å); used for Voronoi input and topology sites"
   "site_equivalence_tolerance", "float", "0.05", "Cartesian tolerance (Å) for merging equivalent adsorption sites after clustering"
   "site_classification_method", "Literal['distance_ratio', 'delaunay']", "'distance_ratio'", "Site typing: six-neighbour distance ratios, or Delaunay triangulation of the slab top layer (slabs only; topology bridges use Delaunay edges by default)"
   "conformer_sampling", "Literal['boltzmann', 'cycle', 'mixed']", "'cycle'", "Method for conformer sampling ('boltzmann', 'cycle', or 'mixed')"
   "slab_relaxation_mode", "Literal['none', 'ionic_only', 'cell_only', 'full']", "'ionic_only'", "ASE relaxation during slab prep; see :doc:`surface_prep`"
   "slab_relaxation_optimizer", "Literal['lbfgs', 'bfgs', 'fire']", "'lbfgs'", "Optimizer for prep-time slab relaxation"
   "slab_relaxation_fmax", "float | None", "None", "Force tolerance for prep-time slab relaxation; falls back to ``fmax`` when unset"
   "slab_relaxation_steps", "int", "200", "Maximum ASE steps for prep-time slab relaxation"
   "min_pbc_image_separation", "float", "8.0", "Minimum in-plane image separation (Å) for :func:`~metalsurfer.surface_prep.auto_resize_substrate_for_molecule` during prep"
   "saturation_save_all_placements", "bool", "True", "Write every validated placement per saturation step under step_*_placements/ and saturation_placements_detailed.csv"
   "saturation_discard_topology_rearrangements", "bool", "True", "Connectivity-only guard before per-step best-slab selection in saturation"
   "saturation_max_steps", "int | None", "None", "Optional cap on saturation loop depth"
   "skip_topology_check", "bool", "False", "Disable decomposition checks; when True, homonuclear diatomics on slabs/nanoparticles get dissociative initial placements"
   "skip_desorption_check", "bool", "False", "Disable post-optimization desorption distance validation"
   "multi_molecule_saturation", "bool", "False", "Competitive multi-molecule saturation when multiple adsorbates are loaded"
   "save_benchmark_dataset", "bool", "False", "Flatten saturation step placements to adsorption_energies_detailed.csv for benchmarking"
   "autobatcher_max_memory_padding", "float", "0.5", "TorchSim autobatcher headroom fraction; used during GPU capacity probing when autotuning placements"
   "autobatcher_max_memory_scaler", "float | None", "None", "Optional TorchSim memory scaler override; when set, also drives autotuned placement counts"
   "bo_enabled", "bool", "False", "Internal BO flag; use :func:`~metalsurfer.run_adsorption_bo` / :func:`~metalsurfer.run_saturation_bo` (ignored with warning on non-BO entry points)"
   "bo_initial_random", "int | None", "None", "Initial random BO batch size; None autotunes to GPU parallel capacity"
   "bo_batch_size", "int | None", "None", "Surrogate-guided BO batch size; None autotunes to GPU parallel capacity"
   "bo_total_budget", "int", "18", "Number of acquisition batches after the initial random batch (not total evaluations)"
   "bo_acquisition", "Literal['lcb', 'ei', 'pi']", "'ei'", "BO acquisition function"
   "bo_surrogate", "Literal['random_forest', 'extra_trees', 'gradient_boost', 'ridge', 'ensemble']", "'ridge'", "BO surrogate model"
   "write_vasp_inputs", "bool", "False", "Write POSCAR/INCAR/KPOINTS placement bundles and reference-slab POSCAR files (XYZ/CSV remain default)"
   "vasp_encut", "int", "400", "VASP ENCUT parameter (eV) when write_vasp_inputs is enabled"
   "vasp_ediff", "float", "1e-6", "VASP EDIFF parameter (eV) when write_vasp_inputs is enabled"
   "vasp_ediffg", "float", "-0.02", "VASP EDIFFG parameter (eV/Å) when write_vasp_inputs is enabled"
   "vasp_nsw", "int", "100", "VASP NSW parameter when write_vasp_inputs is enabled"
   "vasp_kpoints", "tuple[int, int, int]", "(4, 4, 1)", "VASP k-points grid when write_vasp_inputs is enabled"

Saturation coverage is started with :func:`~metalsurfer.run_saturation` or
:func:`~metalsurfer.run_saturation_bo`. Fields prefixed with ``saturation_`` tune
loop behavior and I/O; they do not enable saturation on their own.
