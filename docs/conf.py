from __future__ import annotations

import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "FlexPredict"
author = "NewMrPotato"
release = version("flexpredict")
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

source_suffix = {".md": "markdown"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

autodoc_typehints = "description"
autodoc_class_signature = "separated"
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

html_theme = "furo"
html_title = f"FlexPredict {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["language-switcher.js"]
html_theme_options = {
    "source_repository": "https://github.com/NewMrPotato/ml-flex-predict/",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#3157c8",
        "color-brand-content": "#3157c8",
    },
    "dark_css_variables": {
        "color-brand-primary": "#91a7ff",
        "color-brand-content": "#91a7ff",
    },
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

linkcheck_ignore = [
    r"https://github\.com/NewMrPotato/ml-flex-predict/(issues|discussions)/new.*",
]
