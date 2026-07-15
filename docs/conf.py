"""Sphinx configuration for metalsurfer documentation."""

project = "metalsurfer"
copyright = "2026, metalsurfer contributors"
author = "metalsurfer contributors"

try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("metalsurfer")
except Exception:
    from metalsurfer import __version__

    release = __version__

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
napoleon_use_ivar = False

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": False,
    "special-members": False,
    "inherited-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
    "exclude-members": "__post_init__",
}

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
