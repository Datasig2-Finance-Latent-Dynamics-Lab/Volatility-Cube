import os
import sys

# Make project packages importable by autodoc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

project = 'Volatility Extrapolation'
copyright = '2026, Alvaro Abad'
author = 'Alvaro Abad'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
]

# Napoleon: use Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# autodoc defaults
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}
autodoc_member_order = 'bysource'

templates_path = ['_templates']
exclude_patterns = ['_build']
suppress_warnings = ['ref.duplicate']

html_theme = 'alabaster'
html_static_path = ['_static']
