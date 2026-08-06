.. raw:: html

    <div style="text-align: center; margin: 30px 0 20px 0;">
        <img src="_static/logo_metalsurfer.svg" alt="Metalsurfer" style="width: 200px;">
    </div>
    <div style="text-align: center; margin: 0 0 24px 0;">
        <a href="https://www.python.org/downloads/">
            <img src="https://img.shields.io/badge/python-3.12%2B-blue.svg" alt="Python 3.12+">
        </a>
        <a href="https://pypi.org/project/metalsurfer/">
            <img src="https://img.shields.io/pypi/v/metalsurfer.svg" alt="PyPI">
        </a>
        <a href="https://opensource.org/licenses/MIT">
            <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
        </a>
    </div>

Adsorption on arbitrary materials
==================================

Pass any ASE ``Atoms`` structure (slab, nanoparticle, or porous framework),
prepare it with optional equilibration and freeze constraints, supply SMILES
adsorbates, and run screening or saturation via the ``run_*`` campaign APIs
or YAML documents loaded with :func:`~metalsurfer.load_campaign_yaml` and
:func:`~metalsurfer.run_campaign`. See :doc:`guides/quickstart` for install
steps and runnable examples, :doc:`guides/yaml_campaigns` for YAML structure
and limitations, and :doc:`guides/configuration` for ``AdsorptionConfig``
recipes.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   guides/quickstart
   guides/yaml_campaigns
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