.. raw:: html

    <div style="text-align: center; margin: 30px 0 20px 0;">
        <img src="_static/logo_metalsurfer.svg" alt="Metalsurfer" style="width: 200px;">
    </div>

Adsorption on arbitrary materials
==================================

Pass any ASE ``Atoms`` structure (slab, nanoparticle, or porous framework),
prepare it with optional equilibration and freeze constraints, supply SMILES
adsorbates, and run screening or saturation via the ``run_*`` campaign APIs.
See :doc:`guides/quickstart` for install steps and runnable examples, and
:doc:`guides/configuration` for ``AdsorptionConfig`` recipes.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   guides/quickstart
   guides/configuration
   guides/development
   guides/architecture
   guides/surface_engineering

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/campaigns
   api/surface_prep
   api/config
   api/models