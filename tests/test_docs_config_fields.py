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
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if "**Type:**" not in nxt:
            continue
        # A field-name line may list several comma-separated names
        # (e.g. ``placement_x_range``, ``placement_y_range``).
        for token in re.finditer(r"``([^`]+)``", line):
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
