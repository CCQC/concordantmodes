# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "concordantmodes"
copyright = "2026, Mitchell Lahm"
author = "M.E. Lahm and N. L. Kitzmiller"

try:
    from importlib.metadata import version as _version

    release = _version("concordantmodes")
except Exception:
    release = "unknown"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = []
language = "en"

html_theme = "alabaster"
html_static_path = []


# The options reference is generated from concordantmodes/options.py so it
# always matches the code.
def _write_options_page(app):
    from generate_options import write_options_rst

    write_options_rst(os.path.join(app.srcdir, "options.rst"))


def setup(app):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    app.connect("builder-inited", _write_options_page)
