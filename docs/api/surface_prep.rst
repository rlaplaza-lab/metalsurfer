Surface Preparation
===================

Slab layout conventions: :doc:`../guides/surface_engineering`.

.. autofunction:: metalsurfer.prepare_slab

Prep writes reference structures under ``results_dir`` (for example
``clean_slab.xyz`` before adatoms and ``clean_slab_Au20.xyz`` after 20\%
coverage).  Saturation uses the post-prep slab as ``base_slab_for_frozen``;
compare placement-relaxed geometries to the file that matches that state, not
an earlier prep snapshot.

Optional arguments ``create_relaxation_mode`` / ``adatom_relaxation_mode`` (and
matching optimizer, ``fmax``, ``steps``) override ``AdsorptionConfig.slab_relaxation_*``
for each stage so you can fully equilibrate the clean slab once and use
``ionic_only`` relaxation after depositing adatoms.  See :doc:`../guides/surface_engineering`.
