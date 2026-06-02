Configuration
=============

.. autoclass:: metalsurfer.AdsorptionConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
   :exclude-members: __post_init__

Key Attributes
--------------

.. csv-table::
   :header: "Attribute", "Type", "Default", "Description"
   :widths: 20, 15, 15, 50

   "model_name", "str", "'uma-s-1p1'", "Name of the MLIP model to use for energy calculations"
   "num_conformers", "int", "10", "Number of conformers to generate for each molecule"
   "num_placements", "int", "100", "Number of placement attempts per conformer"
   "device", "str", "'cuda'", "Device to use for MLIP calculations ('cuda' or 'cpu')"
   "fmax", "float", "0.05", "Maximum force threshold for optimization convergence (eV/Å)"
   "stage1_steps", "int", "50", "Number of optimization steps in stage 1 (coarse optimization)"
   "stage2_steps", "int", "150", "Number of optimization steps in stage 2 (fine optimization)"
   "reference_optimization_steps", "int", "100", "Number of optimization steps for reference calculations"
   "placement_x_range", "tuple[float, float]", "(-4.0, 4.0)", "Range for x-coordinate placement (Å)"
   "placement_y_range", "tuple[float, float]", "(-4.0, 4.0)", "Range for y-coordinate placement (Å)"
   "placement_z_range", "tuple[float, float]", "(2.0, 3.0)", "Range for z-coordinate placement (Å)"
   "placement_z_scale_by_covalent_radius", "bool", "True", "Scale z-placement by adsorbate covalent radius"
   "material_type", "Literal['slab', 'nanoparticle', 'porous']", "'slab'", "Type of material ('slab', 'nanoparticle', or 'porous')"
   "voronoi_probe_radius", "float", "1.2", "Minimum distance from framework atoms to Voronoi sites (Å)"
   "voronoi_max_site_distance", "float", "4.0", "Maximum distance for accessible Voronoi sites (Å)"
   "voronoi_site_enrichment", "bool", "True", "Enable geodesic ridge subdivision for denser sites"
   "site_classification_method", "Literal['distance_ratio', 'delaunay']", "'distance_ratio'", "Method for classifying adsorption sites ('distance_ratio' or 'delaunay')"
   "conformer_sampling", "Literal['boltzmann', 'cycle', 'mixed']", "'cycle'", "Method for conformer sampling ('boltzmann', 'cycle', or 'mixed')"
   "relax_top_layer", "bool", "True", "If True, only non-top-layer slab atoms are frozen during placement relaxation; if False, the entire freeze reference is fixed"
   "slab_relaxation_mode", "Literal['none', 'ionic_only', 'cell_only', 'full']", "'none'", "ASE relaxation during slab prep (``create_slab_from_bulk``, ``deposit_adatoms``); not used during TorchSim placement relaxation"
   "slab_relaxation_optimizer", "Literal['lbfgs', 'bfgs', 'fire']", "'lbfgs'", "Optimizer for prep-time slab relaxation"
   "slab_relaxation_steps", "int", "250", "Maximum ASE steps for prep-time slab relaxation"
   "auto_resize_slab", "bool", "True", "Repeat substrate in-plane on saturation step 1 when needed for PBC image separation"
