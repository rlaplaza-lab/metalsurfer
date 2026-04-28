
.. autoclass:: metalsurfer.AdsorptionConfig
   :members:
   :undoc-members:
   :show-inheritance:
=======
Configuration
=============

.. autoclass:: metalsurfer.AdsorptionConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

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
=============

.. autoclass:: metalsurfer.AdsorptionConfig
   :members:
   :undoc-members:
   :show-inheritance:
