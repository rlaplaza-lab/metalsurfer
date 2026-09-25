"""Adsorbate orientation strategies and parallel-fraction estimation."""

import numpy as np
import pytest

from metalsurfer.config import AdsorptionConfig
from metalsurfer.conformers import create_conformers_from_smiles
from metalsurfer.placement import (
    enumerate_placement_specs,
)
from metalsurfer.placement.orientation import (
    _adsorbate_binder_indices,
    _estimate_parallel_fraction,
    _exclusive_marked_binders,
    _is_flat_aromatic_with_en,
    _marked_binder_indices,
    _smiles_tag_charge_indices,
)

from ..conftest import (
    make_slab,
)


def test_flat_aromatic_detection_requires_ring_and_en_atoms():
    assert _is_flat_aromatic_with_en("c1(C=O)cc(OC)c(O)cc1") is True
    assert _is_flat_aromatic_with_en("c1ccccc1") is False
    assert _is_flat_aromatic_with_en("CCO") is False


def test_flat_aromatic_specs_include_parallel_and_en_down_when_applicable():
    pytest.importorskip("rdkit", reason="RDKit required for conformer generation")
    slab = make_slab()
    config = AdsorptionConfig(
        material_type="slab",
        num_placements=24,
        placement_z_range=(2.0, 3.0),
        flat_aromatic_parallel_fraction=0.5,
    )
    result = create_conformers_from_smiles(
        "c1(C=O)cc(OC)c(O)cc1",
        config=AdsorptionConfig(num_conformers=3),
    )
    assert result is not None
    conformers, _ = result

    specs = enumerate_placement_specs(
        conformers,
        slab,
        config,
        "c1(C=O)cc(OC)c(O)cc1",
        n_desired=24,
    )
    kinds = {spec.orientation_type for spec in specs}
    assert "parallel" in kinds
    assert "EN-down" in kinds


@pytest.mark.parametrize(
    "symbols, smiles, expected",
    [
        (["C"] * 6 + ["H"] * 6, "c1ccccc1", 0.8),
        (["C"] * 5 + ["N"] + ["H"] * 5, "c1ccncc1", 0.3),
        (["C"] * 6 + ["O"] + ["H"] * 6, "c1ccccc1O", 0.3),
        (["C"] * 6 + ["N", "O"] + ["H"] * 6, "c1ccc(O)c(N)c1", 0.5),
        (["C"] * 6 + ["H"] * 6, None, 0.8),
        (["C"] * 5 + ["N"] + ["H"] * 5, None, 0.3),
        (["C"] * 4 + ["N", "O"] + ["H"] * 4, None, 0.3),
        (["C"] * 8 + ["N", "O"] + ["H"] * 8, None, 0.5),
    ],
)
def test_estimate_parallel_fraction(symbols, smiles, expected):
    frac = _estimate_parallel_fraction(symbols, smiles=smiles)
    assert frac == expected


def test_marked_binder_indices_from_smiles():
    """Charged atoms (no tags) and tags populate the marked-index channel."""
    assert _marked_binder_indices(None) == ()
    # Benzyl cation: charge on the benzylic carbon (heavy idx 5).
    assert _marked_binder_indices("c1ccc(C[CH2+])cc1") == (5,)
    # Neutral molecule: no marked binders.
    assert _marked_binder_indices("c1ccccc1O") == ()
    # Failed parse falls back to no marked binders.
    assert _marked_binder_indices("not a smiles") == ()


def test_atom_map_tags_are_exclusive_binders():
    """``[atom:map]`` tags override EN/charge binders for EN-down sampling."""
    # Toluene with the methyl carbon tagged: map 1 on heavy idx 5.
    assert _marked_binder_indices("c1ccc(C[CH3:1])cc1") == (5,)
    assert _exclusive_marked_binders("c1ccc(C[CH3:1])cc1") is True
    # Tag present → exclusive: charge at idx 3 is ignored; only the tag.
    assert _smiles_tag_charge_indices("O[CH2:1]C[CH2+]") == ((1,), (3,))
    assert _marked_binder_indices("O[CH2:1]C[CH2+]") == (1,)
    assert _adsorbate_binder_indices(["O", "C", "C", "C"], "O[CH2:1]C[CH2+]") == [1]
    # Without tags, charge merges with EN elements.
    assert _adsorbate_binder_indices(["O", "C", "C", "C"], "OCC[CH2+]") == [0, 3]
    # Toluene without a tag: element-only (no binders).
    assert _marked_binder_indices("c1ccc(C)cc1") == ()
    assert _exclusive_marked_binders("c1ccc(C)cc1") is False


def test_tagged_phenol_excludes_oxygen_from_en_down_pool():
    """Tagging the methyl of *p*-cresol drops the phenol O from EN-down."""
    # Oc1ccc(C)cc1 with methyl tagged — heavy: O=0, … methyl C ≈ 6
    smiles = "Oc1ccc(C[CH3:1])cc1"
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    binders = _adsorbate_binder_indices(symbols, smiles)
    assert binders == [6]  # tagged methyl only
    assert "O" not in [symbols[i] for i in binders]


def test_charged_carbocation_counts_as_binder_for_parallel_fraction():
    """Charged atoms raise the binder count for π-stacking estimation.

    Tropylium ([CH+]1C=CC=CC=C1) has one formal charge and no EN element:
    element-only binders would score 0.8 (no binders); with the SMILES
    charge the single binder scores 0.3. Symbols include Hs (as conformers
    do) so the SMILES-derived count is the only source of binders.
    """
    symbols = ["C"] * 7 + ["H"] * 6
    assert _estimate_parallel_fraction(symbols, None) == 0.8
    assert _estimate_parallel_fraction(symbols, "[CH+]1C=CC=CC=C1") == 0.3


def test_atom_map_tag_counts_as_binder_for_parallel_fraction():
    """A ``[C:1]`` tag acts like a binder for π-stacking estimation.

    Tagged toluene has one binder (the methyl carbon, heavy idx 5) and no
    EN element; without the tag the same molecule has no binders.
    """
    symbols = ["C"] * 7 + ["H"] * 8
    assert _estimate_parallel_fraction(symbols, "Cc1ccccc1") == 0.8
    assert _estimate_parallel_fraction(symbols, "[CH3:1]c1ccccc1") == 0.3


def test_flat_aromatic_charged_ring_counts_as_en_for_parallel_detection():
    """A charged aromatic ring heteroatom satisfies the EN gate (pyrylium)."""
    # Pyrylium: aromatic O+ ring — flat_aromatic even without a neutral binder.
    assert _is_flat_aromatic_with_en("c1cc[o+]cc1") is True
    # Neutral benzene stays excluded.
    assert _is_flat_aromatic_with_en("c1ccccc1") is False


def test_marked_indices_address_conformer_atoms():
    """Heavy-atom SMILES indices address conformer atom lists after AddHs."""
    pytest.importorskip("rdkit", reason="RDKit required for conformer generation")
    smiles = "c1ccc(C[CH2+])cc1"
    result = create_conformers_from_smiles(
        smiles, config=AdsorptionConfig(num_conformers=2, seed=0)
    )
    assert result is not None
    conformers, _ = result
    tagged, charged = _smiles_tag_charge_indices(smiles)
    assert tagged == ()
    assert charged == (5,)
    for conformer in conformers:
        symbols = list(conformer.get_chemical_symbols())
        for idx in charged:
            assert 0 <= idx < len(symbols)
            assert symbols[idx] == "C"


def test_tagged_indices_address_conformer_atoms():
    """``[atom:map]`` tag indices survive conformer generation (AddHs)."""
    pytest.importorskip("rdkit", reason="RDKit required for conformer generation")
    smiles = "c1ccc(C[CH3:1])cc1"
    result = create_conformers_from_smiles(
        smiles, config=AdsorptionConfig(num_conformers=2, seed=0)
    )
    assert result is not None
    conformers, _ = result
    tagged, charged = _smiles_tag_charge_indices(smiles)
    assert tagged == (5,)
    assert charged == ()
    for conformer in conformers:
        symbols = list(conformer.get_chemical_symbols())
        for idx in tagged:
            assert 0 <= idx < len(symbols)
            assert symbols[idx] == "C"


def test_en_down_specs_cover_charged_binder_end_to_end():
    """EN-down enumeration treats the charged atom as a contact point.

    Hydroxybenzyl cation has two merged binders: the phenol O (heavy idx 0)
    and the charged benzylic C (heavy idx 6). Policy emits ``en_atom_index``
    over the merged binder list, so binder-list index 1 is the charged
    carbon; materialization orients that binder toward the surface.
    """
    _run_merged_binder_end_to_end(
        smiles="Oc1ccc(C[CH2+])cc1",
        marked_expected=(6,),
        binder_map={0: 0, 1: 6},
        contact_atom_idx=6,
    )


def test_en_down_specs_cover_tagged_binder_end_to_end():
    """EN-down enumeration treats ``[C:1]``-tagged atoms as contact points.

    Toluene has no EN element; tagging the methyl carbon (heavy idx 5) makes
    it the sole binder, so EN-down specs orient that tagged atom toward the
    surface — a sampling-conditioning knob for user-designated binders.
    """
    _run_merged_binder_end_to_end(
        smiles="c1ccc(C[CH3:1])cc1",
        marked_expected=(5,),
        binder_map={0: 5},
        contact_atom_idx=5,
    )


def _run_merged_binder_end_to_end(
    *,
    smiles: str,
    marked_expected: tuple[int, ...],
    binder_map: dict[int, int],
    contact_atom_idx: int,
) -> None:
    pytest.importorskip("rdkit", reason="RDKit required for conformer generation")
    from metalsurfer.placement.generators import _spec_grid_info
    from metalsurfer.placement.geometry import _surface_aligned_rotation
    from metalsurfer.placement.pose import _contact_atom_index, _pose_from_spec
    from metalsurfer.placement.site_context import _get_unique_sites_for_specs

    slab = make_slab()
    config = AdsorptionConfig(
        material_type="slab",
        num_placements=24,
        placement_z_range=(2.0, 3.0),
        flat_aromatic_parallel_fraction=0.5,
    )
    result = create_conformers_from_smiles(
        smiles, config=AdsorptionConfig(num_conformers=2, seed=0)
    )
    assert result is not None
    conformers, _ = result
    symbols = list(conformers[0].get_chemical_symbols())
    marked = _marked_binder_indices(smiles)
    assert marked == marked_expected

    ctx = _get_unique_sites_for_specs(slab, config)
    assert ctx.use_sites and ctx.sites

    # The binder pool merges the EN element and the SMILES marks.
    info = _spec_grid_info(conformers, slab, config, smiles, ctx)
    assert info.n_binders == len(binder_map)

    specs = enumerate_placement_specs(
        conformers, slab, config, smiles, n_desired=24, site_context=ctx
    )
    en_specs = [s for s in specs if s.orientation_type == "EN-down"]
    assert en_specs, "marked molecule must enumerate EN-down specs"
    # Policy emits binder-list indices only when there is more than one
    # binder (None otherwise); sampled indices stay within the merged pool.
    multi_binder = len(binder_map) > 1
    sampled = {s.en_atom_index for s in en_specs}
    assert sampled <= (set(binder_map) if multi_binder else {None})

    # Every binder-list index is reachable: filter (applied during pool
    # collection) selecting one index yields EN-down specs for it. The
    # single-binder policy emits None instead of index 0.
    marked_ei = next(ei for ei, atom in binder_map.items() if atom == contact_atom_idx)
    en_marked: list = []
    for ei in binder_map:
        target = ei if multi_binder else None
        filtered = enumerate_placement_specs(
            conformers,
            slab,
            config,
            smiles,
            n_desired=8,
            site_context=ctx,
            filter_spec=lambda s, t=target: (
                s.orientation_type != "EN-down" or s.en_atom_index == t
            ),
        )
        en_filtered = [s for s in filtered if s.orientation_type == "EN-down"]
        assert en_filtered
        assert all(s.en_atom_index == target for s in en_filtered)
        if ei == marked_ei:
            en_marked = en_filtered

    # Binder-list index semantics per *binder_map*: the resolved binder
    # points along −normal.
    exclusive = _exclusive_marked_binders(smiles)
    canonical = conformers[0].get_positions()
    canonical = canonical - canonical.mean(axis=0)
    normal = np.array([0.0, 0.0, 1.0])
    for ei, atom_idx in binder_map.items():
        pos, _R = _surface_aligned_rotation(
            canonical,
            normal,
            symbols,
            en_binder_index=ei,
            marked_indices=marked,
            exclusive_marked=exclusive,
        )
        binder_dir = pos[atom_idx] - pos.mean(axis=0)
        binder_dir /= np.linalg.norm(binder_dir)
        assert float(np.dot(binder_dir, normal)) == pytest.approx(-1.0, abs=1e-9)

    # A marked-binder pose materializes and its contact atom is the expected
    # heavy atom.
    en_spec = en_marked[0]
    conformer = conformers[en_spec.conformer_index].copy()
    pctx, fail = _pose_from_spec(
        conformer, en_spec, slab, config, smiles, site_context=ctx
    )
    assert fail is None and pctx is not None
    resolved = _contact_atom_index(
        pctx.rotated_pos,
        np.asarray(pctx.normal, dtype=float),
        symbols,
        orientation_type=en_spec.orientation_type,
        en_atom_index=en_spec.en_atom_index,
        marked_indices=marked,
        exclusive_marked=exclusive,
    )
    assert resolved == contact_atom_idx


def test_principal_axis_rotation_flat_hexagon_stays_near_flat():
    from metalsurfer.placement.geometry import _principal_axis_rotation

    hex_pos = np.array(
        [
            [1.4 * np.cos(i * np.pi / 3), 1.4 * np.sin(i * np.pi / 3), 0.0]
            for i in range(6)
        ],
        dtype=float,
    )
    hex_pos -= hex_pos.mean(axis=0)
    rotated, _score, _R = _principal_axis_rotation(hex_pos, np.array([0.0, 0.0, 1.0]))
    # Regular hexagon: in-plane inertia moments are exactly degenerate, so the
    # recovered normal is only approximate (~5° tilt, z-span ≈ 0.244 Å).
    # Deterministic eigensolver -> pin golden with small slack.
    assert float(np.ptp(rotated[:, 2])) == pytest.approx(0.24403607968514937, abs=0.01)


@pytest.mark.parametrize("tilt_deg", [0.0, 10.0, 15.0])
def test_surface_aligned_rotation_flips_binder_pointing_up(tilt_deg):
    """Binder near +normal must rotate to point toward the surface (−normal)."""
    from metalsurfer.placement.geometry import _surface_aligned_rotation

    normal = np.array([0.0, 0.0, 1.0])
    angle = np.deg2rad(tilt_deg)
    # C at origin, O tilted slightly from +z (away from surface).
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=float,
    )
    out, _R = _surface_aligned_rotation(pos, normal, symbols=["C", "O"])
    com = out.mean(axis=0)
    binder_dir = out[1] - com
    binder_dir /= np.linalg.norm(binder_dir)
    # The alignment rotation maps the binder axis onto -normal exactly,
    # regardless of the initial tilt.
    assert float(np.dot(binder_dir, normal)) == pytest.approx(-1.0, abs=1e-9)


def test_surface_aligned_rotation_noop_when_already_pointing_down():
    from metalsurfer.placement.geometry import _surface_aligned_rotation

    normal = np.array([0.0, 0.0, 1.0])
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=float)
    out, _R = _surface_aligned_rotation(pos, normal, symbols=["C", "O"])
    com = out.mean(axis=0)
    binder_dir = out[1] - com
    binder_dir /= np.linalg.norm(binder_dir)
    assert float(np.dot(binder_dir, normal)) == pytest.approx(-1.0, abs=1e-9)


def test_composed_rotation_reproduces_sequential():
    """R_base @ R_tilt @ canonical must equal the sequentially built rotated_pos."""
    from metalsurfer.placement.geometry import (
        _flat_orientation_from_principal_axis,
        _rotation_with_tilt,
    )

    normal = np.array([0.0, 0.0, 1.0])
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(7, 3))
    pos -= pos.mean(axis=0)
    base_pos, R_base = _flat_orientation_from_principal_axis(
        pos, normal, azimuth_in_plane_deg=37.0, face_flip=True
    )
    rotated, R_tilt = _rotation_with_tilt(
        base_pos, normal, tilt_deg=15.0, azimuth_deg=22.0
    )
    R_total = R_tilt @ R_base
    composed = (R_total @ np.asarray(pos, dtype=float).T).T
    assert np.allclose(composed, rotated, atol=1e-10)
