"""Docs drift guards beyond config.rst field tables.

Covers: ``[docs]`` extras vs ``docs/requirements.txt``, public ``__all__``
symbols vs API autodoc pages, and YAML campaign root keys vs the guide.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

import metalsurfer
from metalsurfer import campaign_schema

_ROOT = Path(__file__).resolve().parents[1]
_API_DIR = _ROOT / "docs" / "api"
_YAML_GUIDE = _ROOT / "docs" / "guides" / "yaml_campaigns.rst"
_DOCS_REQS = _ROOT / "docs" / "requirements.txt"
_PYPROJECT = _ROOT / "pyproject.toml"

_AUTODOC_TARGET = re.compile(
    r"^\.\. auto(?:function|class|exception|data|attribute|method|module)"
    r"::\s+(\S+)",
    re.MULTILINE,
)
_PY_CLASS = re.compile(r"^\.\. py:class::\s+(\S+)", re.MULTILINE)


def _api_rst_text() -> str:
    parts = [p.read_text(encoding="utf-8") for p in sorted(_API_DIR.glob("*.rst"))]
    return "\n".join(parts)


def _documented_api_symbol_names() -> set[str]:
    """Leaf names targeted by autodoc / hand-written py:class in docs/api/."""
    text = _api_rst_text()
    names: set[str] = set()
    for match in _AUTODOC_TARGET.finditer(text):
        names.add(match.group(1).rsplit(".", 1)[-1])
    for match in _PY_CLASS.finditer(text):
        names.add(match.group(1).rsplit(".", 1)[-1])
    return names


def _docs_extra_requirements() -> list[str]:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    return list(data["project"]["optional-dependencies"]["docs"])


def _requirements_txt_lines() -> list[str]:
    lines: list[str] = []
    for raw in _DOCS_REQS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def _documented_yaml_root_keys() -> set[str]:
    text = _YAML_GUIDE.read_text(encoding="utf-8")
    start = text.index("Top-level keys:")
    # Simple table: top rule, header, mid rule, rows, bottom rule.
    after = text[start:]
    seps = list(re.finditer(r"^={10,} .*$", after, re.MULTILINE))
    assert len(seps) >= 3, (
        "expected top-level keys table separators in yaml_campaigns.rst"
    )
    body = after[seps[1].end() : seps[2].start()]
    keys: set[str] = set()
    for line in body.splitlines():
        match = re.match(r"^``([a-z_]+)``\s+", line.strip())
        if match:
            keys.add(match.group(1))
    return keys


@pytest.mark.docs
def test_docs_requirements_match_pyproject_extra():
    """RTD requirements file must stay in lockstep with the [docs] extra."""
    from_extra = _docs_extra_requirements()
    from_file = _requirements_txt_lines()
    assert from_file == from_extra, (
        "docs/requirements.txt must match pyproject.toml [project.optional-dependencies].docs "
        f"exactly.\nrequirements.txt: {from_file}\npyproject: {from_extra}"
    )


@pytest.mark.docs
def test_public_all_symbols_have_api_docs():
    """Every public export (except ``__version__``) must have an API autodoc target."""
    documented = _documented_api_symbol_names()
    missing = [
        name
        for name in metalsurfer.__all__
        if name != "__version__" and name not in documented
    ]
    assert not missing, (
        f"Public metalsurfer.__all__ symbols missing from docs/api/: {missing}"
    )


@pytest.mark.docs
def test_yaml_campaign_root_keys_match_schema():
    """Guide top-level YAML keys must match campaign_schema._ROOT_KEYS."""
    documented = _documented_yaml_root_keys()
    schema_keys = set(campaign_schema._ROOT_KEYS)
    assert documented == schema_keys, (
        f"YAML guide keys {sorted(documented)} != schema {sorted(schema_keys)}"
    )


@pytest.mark.docs
def test_package_version_matches_pyproject():
    """pyproject.toml version and metalsurfer.__version__ must stay in sync."""
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    pyproject_version = data["project"]["version"]
    assert metalsurfer.__version__ == pyproject_version, (
        f"__version__={metalsurfer.__version__!r} != "
        f"pyproject.toml version={pyproject_version!r}"
    )
