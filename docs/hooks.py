"""MkDocs hooks of the documentation site (``hooks:`` in mkdocs.yml).

mike publishes the site twice (.github/workflows/docs.yml): ``stable``, built
from the latest GitHub release, and ``nightly``, built from every commit on
main. The pages, README.md included, are written for stable and say
``pip install torch-mojo-backend``, which fetches the release from PyPI. The
nightly build rewrites that to an install from the repository, so each
version tells its reader how to get the code it documents.
"""

import os
import re

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page

GIT_URL = "git+https://github.com/gabrieldemarmiesse/torch-mojo-backend"
PYPI_INSTALL = re.compile(r"\b(pip install|uv add) torch-mojo-backend\b")


def on_page_markdown(
    markdown: str, *, page: Page, config: MkDocsConfig, files: Files
) -> str:
    # mike exports the version it is building. A plain `mkdocs build` or
    # `mkdocs serve` builds a checkout of main, so it previews nightly.
    if os.environ.get("MIKE_DOCS_VERSION", "nightly") != "nightly":
        return markdown
    return PYPI_INSTALL.sub(rf"\1 {GIT_URL}", markdown)
