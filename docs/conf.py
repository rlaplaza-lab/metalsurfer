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
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

# -- Autodoc ------------------------------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_class_signature = "separated"
autodoc_docstring_signature = True

# Enable extraction of inline comments for dataclass fields
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": False,
    "special-members": False,
    "inherited-members": True,
    "show-inheritance": True,
}

# Better dataclass field documentation
autodoc_docstring_signature = True

# Improve dataclass documentation - hide the inline comments to avoid duplication with CSV table
autodoc_class_signature = "separated"
autodoc_typehints = "description"
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
napoleon_use_ivar = False  # Hide dataclass field docstrings since we have the CSV table

# -- Intersphinx --------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "ase": ("https://ase-lib.org/", None),
}

# -- Theme ---------------------------------------------------------------------

html_theme = "furo"
html_title = "metalsurfer"
html_logo = "_static/logo_metalsurfer.svg"
html_favicon = "_static/logo_metalsurfer.svg"

# Furo theme specific settings
html_theme_options = {
    "sidebar_hide_name": True,
}

# -- General -------------------------------------------------------------------

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
