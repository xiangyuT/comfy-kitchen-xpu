"""Resolve the package version from its build metadata."""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path


def _source_tree_version() -> str | None:
    """Read the PEP 621 version when importing directly from a checkout."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.is_file():
        return None
    match = re.search(
        r'^version\s*=\s*["\']([^"\']+)["\']\s*$',
        pyproject.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    return match.group(1) if match else None


def get_version() -> str:
    """Return one identity for source imports and installed wheels."""
    source_version = _source_tree_version()
    if source_version is not None:
        return source_version
    try:
        return distribution_version("comfy-kitchen")
    except PackageNotFoundError:
        return "0+unknown"


__version__ = get_version()
