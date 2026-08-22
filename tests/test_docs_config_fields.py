"""Guard against documentation drift in ``docs/api/config.rst``.

Parses the hand-written field reference and asserts that every dataclass field of
``AdsorptionConfig`` / ``BOConfig`` / ``BOTransferConfig`` is documented. This catches
the class of bugs where a new config field is added (or a default changes) but the
API docs are never updated.
"""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path

import pytest

from metalsurfer.config import AdsorptionConfig, BOConfig, BOTransferConfig

_CONFIG_RST = Path(__file__).resolve().parents[1] / "docs" / "api" / "config.rst"


def _documented_field_names() -> set[str]:
    text = _CONFIG_RST.read_text(encoding="utf-8").splitlines()
    documented: set[str] = set()
    for i, line in enumerate(text):
        if "**Type:**" not in line:
            continue
        prev = ""
        for j in range(i - 1, -1, -1):
            candidate = text[j].strip()
            if candidate:
                prev = text[j]
                break
        if not prev:
            continue
        for token in re.finditer(r"``([^`]+)``", prev):
            documented.add(token.group(1))
    return documented


@pytest.mark.docs
def test_all_config_fields_documented():
    documented = _documented_field_names()

    ads_missing = {f.name for f in fields(AdsorptionConfig)} - documented
    assert not ads_missing, (
        f"Undocumented AdsorptionConfig fields: {sorted(ads_missing)}"
    )

    bo_missing = {f"bo.{f.name}" for f in fields(BOConfig)} - documented
    assert not bo_missing, f"Undocumented BOConfig fields: {sorted(bo_missing)}"

    transfer_missing = {
        f"bo.transfer.{f.name}" for f in fields(BOTransferConfig)
    } - documented
    assert not transfer_missing, (
        f"Undocumented BOTransferConfig fields: {sorted(transfer_missing)}"
    )


@pytest.mark.docs
def test_no_stale_documented_fields():
    """Docs must not list fields that no longer exist on the dataclasses."""
    documented = _documented_field_names()
    ads = {f.name for f in fields(AdsorptionConfig)}
    bo = {f"bo.{f.name}" for f in fields(BOConfig)}
    transfer = {f"bo.transfer.{f.name}" for f in fields(BOTransferConfig)}
    known = ads | bo | transfer
    # Only treat tokens that match known naming patterns as field docs.
    fieldish = {n for n in documented if n in ads or n.startswith("bo.")}
    stale = fieldish - known
    assert not stale, f"Stale documented fields: {sorted(stale)}"


@pytest.mark.docs
def test_documented_defaults_match_simple_scalars():
    """When a Default line has a single simple backticked scalar, it must match code."""
    from dataclasses import MISSING

    text = _CONFIG_RST.read_text(encoding="utf-8").splitlines()
    ads = {f.name: f for f in fields(AdsorptionConfig)}
    bad: list[tuple[str, object, list[str]]] = []
    for i, line in enumerate(text):
        if "**Default:**" not in line:
            continue
        tick_vals = re.findall(r"``([^`]+)``", line)
        # Only check unambiguous single-token defaults (skip prose / multi-token).
        if len(tick_vals) != 1:
            continue
        doc_val = tick_vals[0]
        # Walk upward to the field name preceding the Type line.
        name = None
        for j in range(i - 1, max(-1, i - 12), -1):
            if "**Type:**" in text[j]:
                for k in range(j - 1, max(-1, j - 6), -1):
                    if text[k].strip():
                        names = re.findall(r"``([^`]+)``", text[k])
                        if len(names) == 1 and names[0] in ads:
                            name = names[0]
                        break
                break
        if name is None:
            continue
        field = ads[name]
        if field.default is MISSING:
            continue
        val = field.default
        if not isinstance(val, (bool, int, float, str)):
            continue
        ok = doc_val in {str(val), repr(val)}
        if isinstance(val, str):
            ok = ok or doc_val.strip("\"'") == val
        if not ok:
            bad.append((name, val, tick_vals))
    assert not bad, f"Documented defaults disagree with code: {bad[:10]}"


def _field_doc_block(field_name: str) -> tuple[str, str] | None:
    """Return (type_line, default_line) for a documented AdsorptionConfig field."""
    text = _CONFIG_RST.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(text):
        if f"``{field_name}``" not in line or "**Type:**" in line:
            continue
        type_line = default_line = ""
        for j in range(i + 1, min(len(text), i + 8)):
            if "**Type:**" in text[j]:
                type_line = text[j]
            if "**Default:**" in text[j]:
                default_line = text[j]
                break
        if type_line and default_line:
            return type_line, default_line
    return None


@pytest.mark.docs
def test_connectivity_multiplier_documentation_matches_code():
    """Catch list-vs-scalar doc drift that multi-token Default lines hide."""
    from dataclasses import fields

    field = next(
        f for f in fields(AdsorptionConfig) if f.name == "connectivity_multiplier"
    )
    block = _field_doc_block("connectivity_multiplier")
    assert block is not None, "connectivity_multiplier section missing from config.rst"
    type_line, default_line = block
    assert "float" in type_line, type_line
    assert "list" not in type_line.lower(), type_line
    assert str(field.default) in default_line, default_line
