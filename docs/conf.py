# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "S2BALL"
copyright = "2024, Matthew Price and Jason McEwen"
author = "Matthew Price, Jason McEwen"

# The short X.Y version
version = "0.0.1"
# The full version, including alpha/beta/rc tags
release = "0.0.1"


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_copybutton",
    "nbsphinx_link",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx_rtd_theme",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_tabs.tabs",
    "sphinx_git",
    "sphinxcontrib.texfigure",
    "sphinx.ext.autosectionlabel",
    "sphinxemoji.sphinxemoji",
]

nbsphinx_execute = "never"
napoleon_google_docstring = True
napoleon_include_init_with_doc = True
napoleon_numpy_docstring = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
source_suffix = [".rst", ".ipynb"]

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None
sphinx_tabs_disable_css_loading = True

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"

html_logo = "assets/sax_logo.png"
html_theme_options = {
    "footer_items": "copyright.html",
    "logo_only": True,
    "display_version": False,
    "navbar_align": "left",
    "announcement": "s2ball is currently in an open alpha, please provide feedback on GitHub",
    "show_toc_level": 2,
    "show_nav_level": 1,
    "header_links_before_dropdown": 5,
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "icon_links": [
        {
            "name": "ArXiv",
            "url": "https://arxiv.org/abs/xxxx.xxxxx",
            "icon": "_static/arxiv-logomark-small.png",
            "type": "local",
        },
        # {
        #     "name": "YouTube",
        #     "url": "https://www.youtube.com/channel/UCrCOQsyQOJhOUaIYzmbkKQQ",
        #     "icon": "fa-brands fa-youtube fa-2x",
        #     "type": "fontawesome",
        # },
        # {
        #     "name": "PyPi",
        #     "url": "https://github.com/astro-informatics/s2fft/",
        #     "icon": "_static/pypi.png",
        #     "type": "local",
        # },
        {
            "name": "GitHub",
            "url": "https://github.com/astro-informatics/s2ball/",
            "icon": "fa-brands fa-github fa-2x",
            "type": "fontawesome",
        },
        {
            "name": "CodeCov",
            "url": "https://app.codecov.io/gh/astro-informatics/s2ball",
            "icon": "_static/codecov.png",
            "type": "local",
        },
        {
            "name": "Licence",
            "url": "https://opensource.org/licenses/MIT",
            "icon": "_static/MIT_Licence.png",
            "type": "local",
        },
    ],
}

html_sidebars = {
    # "tutorials/*": [
    #     "indices.html",
    #     "navbar-nav.html",
    # ],
    "user_guide/*": [
        "indices.html",
        "navbar-nav.html",
    ],
    # "background/*": [
    #     "indices.html",
    #     "navbar-nav.html",
    # ],
}

html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
    "css/custom_tabs.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/fontawesome.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/solid.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/brands.min.css",
]
panels_add_bootstrap_css = False

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "S2BALLdoc"


# -- Options for LaTeX output ------------------------------------------------

latex_documents = [
    (
        master_doc,
        "S2BALL.tex",
        "S2BALL Documentation",
        "Matthew Price, Jason McEwen",
        "manual",
    ),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "s2ball", "S2BALL Documentation", [author], 1)]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]

suppress_warnings = ["autosectionlabel.*", "autodoc", "autodoc.import_object"]

# -- Extension configuration -------------------------------------------------
