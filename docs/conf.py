import os
import sys

# sys.path.insert(0, os.path.abspath("../../src"))
sys.path.insert(0, os.path.abspath("../src"))  # adjust path as needed
# sys.path.append("../..")  # Adjust this path as needed
import micromet

# -- Project information -----------------------------------------------------
project = "micromet"
copyright = "2025, Paul Inkenbrandt & Kathryn Ladig"
author = "Paul Inkenbrandt & Kathryn Ladig"
release = "0.1.35"

master_doc = "index"  # The name of the master document (without the .rst extension)
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = [
    "tests_micromet/*",
    "_build/*",
    "docs/_build/*",
]  # Exclude the tests_micromet directory and _build directory

napoleon_google_docstring = False  # You can still use Google-style if True
napoleon_numpy_docstring = True  # Set this to True for NumPy-style

napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
