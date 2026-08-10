"""Tests for the io_results merge helper and related result-writing behaviour."""

import json
import warnings

import pandas as pd
from ase.build import fcc111
from ase.io import read

from metalsurfer.io_results import (
    _merge_preserving_existing_molecules,
    _write_clean_xyz,
)


def test_merge_preserves_molecules_absent_from_the_new_run(tmp_path):
    """Regression: ``skip_existing`` read the same CSV that was truncated.

    Run 1 wrote {A, B}; run 2 skipped A and B (already present) and then
    overwrote the file with only {C}, permanently losing A and B.
    """

    path = tmp_path / "adsorption_energies_detailed.csv"
    pd.DataFrame(
        [{"molecule": "A", "E_ads": -1.0}, {"molecule": "B", "E_ads": -2.0}]
    ).to_csv(path, index=False)

    merged = _merge_preserving_existing_molecules(
        path, pd.DataFrame([{"molecule": "C", "E_ads": -3.0}])
    )
    assert set(merged["molecule"]) == {"A", "B", "C"}


def test_merge_replaces_rows_for_recomputed_molecules(tmp_path):
    path = tmp_path / "summary.csv"
    pd.DataFrame(
        [{"molecule": "A", "E_ads": -1.0}, {"molecule": "B", "E_ads": -2.0}]
    ).to_csv(path, index=False)

    merged = _merge_preserving_existing_molecules(
        path, pd.DataFrame([{"molecule": "A", "E_ads": -9.0}])
    )
    assert set(merged["molecule"]) == {"A", "B"}
    assert merged.loc[merged["molecule"] == "A", "E_ads"].tolist() == [-9.0]
    assert merged.loc[merged["molecule"] == "B", "E_ads"].tolist() == [-2.0]


def test_merge_falls_back_when_existing_file_is_corrupt(tmp_path):
    path = tmp_path / "summary.csv"
    path.write_text("not,a,valid\ncsv\x00row\n")
    new = pd.DataFrame([{"molecule": "C", "E_ads": -3.0}])
    merged = _merge_preserving_existing_molecules(path, new)
    assert set(merged["molecule"]) == {"C"}


def test_merge_unions_columns_across_schema_versions(tmp_path):
    path = tmp_path / "detailed.csv"
    pd.DataFrame([{"molecule": "A", "E_ads": -1.0}]).to_csv(path, index=False)

    merged = _merge_preserving_existing_molecules(
        path,
        pd.DataFrame([{"molecule": "B", "E_ads": -2.0, "new_column": 1.0}]),
    )
    assert set(merged.columns) == {"molecule", "E_ads", "new_column"}
    assert set(merged["molecule"]) == {"A", "B"}
    # The pre-existing row must not have acquired a bogus value.
    assert pd.isna(merged.loc[merged["molecule"] == "A", "new_column"]).all()
    json.dumps(merged.to_dict(orient="records"), default=str)


def test_write_clean_xyz_drops_stale_adsorbate_info(tmp_path):
    """Slabs built via ase.build carry stale ``adsorbate_info``; writing must not warn.

    ``ase.io.extxyz`` warns and drops an unhashable ``adsorbate_info`` on write, so
    ``_write_clean_xyz`` pops it first.
    """
    slab = fcc111("Cu", size=(2, 2, 3), a=3.6, vacuum=5.0)
    assert "adsorbate_info" in slab.info

    out = tmp_path / "slab.xyz"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _write_clean_xyz(slab, str(out))

    adsorbate_warnings = [
        w
        for w in caught
        if issubclass(w.category, UserWarning) and "adsorbate_info" in str(w.message)
    ]
    assert not adsorbate_warnings

    assert out.exists()
    read_back = read(str(out))
    assert len(read_back) == len(slab)
