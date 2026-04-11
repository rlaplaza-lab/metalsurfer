"""Sphinx configuration for metalsurfer documentation."""

project = "metalsurfer"
copyright = "2026, metalsurfer contributors"
author = "metalsurfer contributors"
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

# -- Autodoc ------------------------------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_class_signature = "separated"

# Mock heavy optional dependencies so docs build without torch/fairchem/GPU.
autodoc_mock_imports = [
    "torch",
    "torch_sim",
    "fairchem",
    "fairchem.core",
    "fairchem_core",
    "fairchem.data",
    "fairchem_data_oc",
]

# -- Napoleon (NumPy-style docstrings) ----------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Intersphinx --------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "ase": ("https://ase-lib.org/", None),
}

# -- Theme ---------------------------------------------------------------------

html_theme = "furo"
html_title = "metalsurfer"

# -- General -------------------------------------------------------------------

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
