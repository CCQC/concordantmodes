# Sphinx configuration for the Concordant Modes manual.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)

# -- Project -------------------------------------------------------------------
project = "Concordant Modes"
author = "M. E. Lahm, N. L. Kitzmiller, W. D. Allen and co-workers"
copyright = "2022-2026, the Concordant Modes developers"

try:
    from importlib.metadata import version as _version

    release = _version("concordantmodes")
except Exception:  # building from a source tree that is not installed
    release = "dev"
version = release

# -- General -------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # NumPy-style docstrings
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
]

templates_path = ["_templates"]
exclude_patterns = []
language = "en"
nitpicky = False
todo_include_todos = False

# Equations are numbered per page and can be referenced with :eq:.
math_numfig = True
numfig = True

# -- API documentation ---------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": False,
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"
# MolSym is optional; the docs build without it.
autodoc_mock_imports = ["molsym"]
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False

# -- HTML ----------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "Concordant Modes"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "github_url": "https://github.com/CCQC/ConcordantModes",
    "navbar_align": "left",
    "show_toc_level": 2,
    "navigation_depth": 3,
    "use_edit_page_button": False,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
}
html_context = {"default_mode": "auto"}
html_show_sourcelink = True


# -- Generated pages -----------------------------------------------------------
# The options reference is generated from concordantmodes/options.py on every
# build, so it always matches the code.
def _write_options_page(app):
    from generate_options import write_options_rst

    write_options_rst(os.path.join(app.srcdir, "reference", "options.rst"))


def setup(app):
    app.connect("builder-inited", _write_options_page)
