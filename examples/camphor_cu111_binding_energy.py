#!/usr/bin/env python3
"""Revisit (1S)-camphor on Cu(111) from Järvi et al. (Beilstein J. Nanotechnol. 2020).

Paper: https://doi.org/10.3762/bjnano.11.140
Zenodo BOSS dataset: https://doi.org/10.5281/zenodo.4680467

Uses metalsurfer BO-guided placement search to find local adsorption minima on the
paper's Cu(111) slab (192 Cu from NOMAD; bottom 2 layers frozen). MLIP energies are
compared qualitatively to the paper's eight DFT minima (not absolute eV).

Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"

Uses the paper's 192-atom Cu(111) slab from NOMAD and a 25-batch BO budget to search
for placements comparable to the eight published DFT minima.

If you hit CUDA OOM on a 15GB GPU, try:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python examples/camphor_cu111_binding_energy.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from metalsurfer import (
    AdsorptionConfig,
    BindingCampaignResult,
    configure_logging,
    resolved_bo_eval_budget,
    results_dir_for,
    run_adsorption_bo,
)
from metalsurfer.models import ScreeningResult
from metalsurfer.surface_prep import SlabContainer, prepare_substrate

# (1S)-(-)-camphor (PubChem InChIKey DSSYKIVIOFKYAU-OIBJUYFYSA-N).
# Do not use CC1(C)[C@@H]2CC[C@@]1(C)C(=O)C2 — that is the (1R)-(+)-enantiomer.
CAMPHOR_SMILES = "[H][C@]12CC[C@](C)(C(=O)C1)C2(C)C"
MOLECULE_NAME = "camphor"
SURFACE_TYPE = "camphor_cu111"
RESULTS_DIR = str(results_dir_for(SURFACE_TYPE))
# Acquisition batches after the initial random batch (~300+ MLIP evals on a 15GB GPU).
BO_TOTAL_BUDGET = 25

EXAMPLE_DIR = Path(__file__).resolve().parent
DATA_DIR = EXAMPLE_DIR / "camphor_cu111"
PAPER_SLAB_FILENAME = "dft_reference_slab.xyz"
# Järvi et al. 2020: 4-layer (6×4)√3 orthogonal Cu(111), bottom 2 layers frozen.
PAPER_SLAB_ATOMS = 192
PAPER_SLAB_CELL_ANG = (15.41, 17.79, 56.29)
PAPER_RELAXED_CU_LAYERS = 2
PAPER_TOP_LAYER_TOLERANCE = 2.1  # Å; spans 2 Cu layers (~2.08 Å interlayer spacing)
ZENODO_RECORD_URL = (
    "https://zenodo.org/records/4680467/files/Camphor_Cu111_BOSS_dataset.dat"
)
ZENODO_DAT_FILENAME = "Camphor_Cu111_BOSS_dataset.dat"

NOMAD_API = "https://nomad-lab.eu/prod/v1/api/v1"
NOMAD_DATASET_DOI = "10.17172/NOMAD/2021.04.12-1"
NOMAD_UPLOAD_ID = "gsssj6pDRa-uV26IfmUHkg"
NOMAD_INDEX_FILENAME = "nomad_index.json"
NOMAD_REF_DIRNAME = "nomad_references"

# BOSS surrogate energies (E_B column) from Table 1 — used to pick NOMAD calc IDs.
PAPER_BOSS_EB: dict[str, float] = {
    "Ox1": -0.961,
    "Ox2": -0.910,
    "Ox3": -0.889,
    "Ox4": -0.803,
    "Ox5": -0.704,
    "Hy2": -0.737,
    "Hy3": -0.658,
    "Hy1": -0.634,
}

# DFT adsorption energies (E_D column) from Table 1, Järvi et al. Beilstein J. Nanotechnol. 2020.
PAPER_DFT_MINIMA: tuple[tuple[str, str, float], ...] = (
    ("Ox1", "oxygen", -0.933),
    ("Ox2", "oxygen", -0.885),
    ("Ox3", "oxygen", -0.850),
    ("Ox4", "oxygen", -0.723),
    ("Ox5", "oxygen", -0.706),
    ("Hy2", "hydrogen", -0.719),
    ("Hy3", "hydrogen", -0.652),
    ("Hy1", "hydrogen", -0.631),
)


@dataclass(frozen=True)
class ResolvedBoBudget:
    bo_initial_random: int
    bo_batch_size: int
    bo_total_budget: int
    eval_budget: int


@dataclass(frozen=True)
class NomadCalculation:
    calc_id: str
    energy_eb: float
    mainfile: str
    entry_id: str


@dataclass(frozen=True)
class ReferenceMinimum:
    label: str
    binding_class: str
    boss_eb: float
    calculation: NomadCalculation
    geometry_path: Path


@dataclass(frozen=True)
class BindingGeometry:
    com_height: float
    o_distance: float
    h_distance: float
    mode: str


@dataclass(frozen=True)
class GeometryMatch:
    placement_id: int
    best_label: str
    adsorbate_rmsd: float
    com_height_delta: float
    mlip_mode: str
    dft_mode: str
    mode_match: bool


def _configure_logging(debug: bool = False) -> None:
    level_name = "DEBUG" if debug else "INFO"
    configure_logging(default_level=level_name)
    if debug:
        logging.getLogger("metalsurfer.filters").setLevel(logging.DEBUG)
        logging.getLogger("metalsurfer.workflow").setLevel(logging.DEBUG)


def ensure_zenodo_reference_data() -> Path:
    """Download the BOSS reference dataset if not already cached."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / ZENODO_DAT_FILENAME
    if dest.exists():
        return dest

    print(f"Downloading reference data from {ZENODO_RECORD_URL} ...")
    urllib.request.urlretrieve(ZENODO_RECORD_URL, dest)
    print(f"Saved to {dest}")
    return dest


def extract_clean_slab_from_nomad(atoms: Atoms) -> Atoms:
    """Extract the Cu slab from a NOMAD adslab, preserving atom order and cell."""
    symbols = atoms.get_chemical_symbols()
    cu_idx = [i for i, sym in enumerate(symbols) if sym == "Cu"]
    if not cu_idx:
        raise ValueError("No Cu atoms found in NOMAD structure")
    return Atoms(
        symbols=[symbols[i] for i in cu_idx],
        positions=atoms.get_positions()[cu_idx],
        cell=atoms.cell,
        pbc=atoms.pbc,
    )


def ensure_dft_reference_slab() -> Path:
    """Cache the paper's 192-atom Cu(111) slab extracted from NOMAD."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / PAPER_SLAB_FILENAME
    if dest.exists():
        return dest

    references = ensure_nomad_reference_structures()
    source = read(references[0].geometry_path, format="aims")
    slab = extract_clean_slab_from_nomad(source)
    if len(slab) != PAPER_SLAB_ATOMS:
        raise RuntimeError(
            f"Expected {PAPER_SLAB_ATOMS} Cu atoms in paper slab, got {len(slab)}"
        )
    lengths = slab.cell.lengths()
    for actual, target in zip(lengths, PAPER_SLAB_CELL_ANG, strict=True):
        if abs(actual - target) > 0.05:
            raise RuntimeError(
                f"Paper slab cell mismatch: got {lengths}, expected {PAPER_SLAB_CELL_ANG}"
            )
    slab.info["source"] = "NOMAD paper Cu(111) slab (Järvi et al. 2020)"
    slab.info["frozen_layers"] = PAPER_SLAB_ATOMS // 2
    slab.info["relaxed_layers"] = PAPER_RELAXED_CU_LAYERS
    write(dest, slab, format="extxyz")
    print(f"Cached paper DFT slab ({len(slab)} Cu) to {dest}")
    return dest


def prepare_campaign_slab(
    config: AdsorptionConfig,
    *,
    results_directory: str,
) -> SlabContainer:
    slab_path = ensure_dft_reference_slab()
    return prepare_substrate(
        slab_file=str(slab_path),
        config=config,
        results_dir=results_directory,
        align=False,
        relax_top_layer=True,
        top_layer_tolerance=PAPER_TOP_LAYER_TOLERANCE,
    )


def build_config(*, device: str) -> AdsorptionConfig:
    # GPU memory padding for ~15 GB cards; BO budget is acquisition batches.
    return AdsorptionConfig(
        material_type="slab",
        seed=42,
        num_conformers=1,
        autobatcher_max_memory_padding=0.8,
        autobatcher_max_memory_scaler=500,
        autobatcher_max_atoms_to_try=5000,
        device=device,
        stage2_steps=500,
        placement_z_range=(4.0, 7.0),
        placement_z_scale_by_covalent_radius=False,
        slab_relaxation_mode="none",
        top_layer_tolerance=PAPER_TOP_LAYER_TOLERANCE,
        bo_total_budget=BO_TOTAL_BUDGET,
        bo_acquisition="ei",
    )


def resolve_bo_budget(config: AdsorptionConfig) -> ResolvedBoBudget | None:
    if config.bo_initial_random is None or config.bo_batch_size is None:
        return None
    return ResolvedBoBudget(
        bo_initial_random=config.bo_initial_random,
        bo_batch_size=config.bo_batch_size,
        bo_total_budget=config.bo_total_budget,
        eval_budget=resolved_bo_eval_budget(config),
    )


def _nomad_post_json(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def fetch_nomad_index() -> list[NomadCalculation]:
    """List all 609 FHI-aims calculations in the NOMAD benchmark dataset."""
    index_path = DATA_DIR / NOMAD_INDEX_FILENAME
    if index_path.exists():
        raw = json.loads(index_path.read_text())
        return [NomadCalculation(**row) for row in raw]

    api = f"{NOMAD_API}/entries/archive/query"
    query = {
        "query": {"datasets.doi": NOMAD_DATASET_DOI},
        "pagination": {"page_size": 100},
        "required": {"metadata": {"mainfile": "*", "entry_id": "*"}},
    }
    calculations: list[NomadCalculation] = []
    page_after: str | None = None
    while True:
        body = dict(query)
        if page_after:
            body["pagination"] = {**query["pagination"], "page_after_value": page_after}
        payload = _nomad_post_json(api, body)
        for row in payload["data"]:
            mainfile = row["archive"]["metadata"]["mainfile"]
            match = re.search(r"/(\d{3})_(-?\d+\.\d+)/output$", mainfile)
            if not match:
                continue
            calculations.append(
                NomadCalculation(
                    calc_id=match.group(1),
                    energy_eb=float(match.group(2)),
                    mainfile=mainfile,
                    entry_id=row["entry_id"],
                )
            )
        page_after = payload["pagination"].get("next_page_after_value")
        if not page_after or len(calculations) >= payload["pagination"]["total"]:
            break

    calculations.sort(key=lambda row: row.calc_id)
    index_path.write_text(json.dumps([row.__dict__ for row in calculations], indent=2))
    return calculations


def assign_reference_calculations(
    calculations: list[NomadCalculation],
) -> dict[str, NomadCalculation]:
    """Map each paper minimum label to the closest unused NOMAD calculation."""
    assigned: dict[str, NomadCalculation] = {}
    used: set[str] = set()
    for label in sorted(PAPER_BOSS_EB, key=lambda key: PAPER_BOSS_EB[key]):
        target = PAPER_BOSS_EB[label]
        for calc in sorted(calculations, key=lambda row: abs(row.energy_eb - target)):
            if calc.calc_id in used:
                continue
            assigned[label] = calc
            used.add(calc.calc_id)
            break
    if len(assigned) != len(PAPER_BOSS_EB):
        raise RuntimeError(
            f"Could only assign {len(assigned)}/{len(PAPER_BOSS_EB)} NOMAD references"
        )
    return assigned


def _nomad_geometry_url(mainfile: str) -> str:
    folder = mainfile.removesuffix("/output")
    return f"{NOMAD_API}/uploads/{NOMAD_UPLOAD_ID}/raw/{folder}/geometry.in"


def ensure_nomad_reference_structures() -> list[ReferenceMinimum]:
    """Download eight NOMAD geometry.in files for the paper's BOSS minima."""
    calculations = fetch_nomad_index()
    assigned = assign_reference_calculations(calculations)
    ref_dir = DATA_DIR / NOMAD_REF_DIRNAME
    ref_dir.mkdir(parents=True, exist_ok=True)

    class_by_label = {label: cls for label, cls, _ in PAPER_DFT_MINIMA}
    references: list[ReferenceMinimum] = []
    for label, calc in assigned.items():
        dest = ref_dir / f"{label}.in"
        if not dest.exists():
            url = _nomad_geometry_url(calc.mainfile)
            print(f"Downloading NOMAD {label} (calc {calc.calc_id}) ...")
            _download_file(url, dest)
        references.append(
            ReferenceMinimum(
                label=label,
                binding_class=class_by_label[label],
                boss_eb=PAPER_BOSS_EB[label],
                calculation=calc,
                geometry_path=dest,
            )
        )
    return references


def split_slab_adsorbate(atoms: Atoms) -> tuple[Atoms, Atoms]:
    symbols = atoms.get_chemical_symbols()
    slab_idx = [i for i, sym in enumerate(symbols) if sym == "Cu"]
    ads_idx = [i for i, sym in enumerate(symbols) if sym != "Cu"]
    if not ads_idx:
        raise ValueError("No adsorbate atoms found")
    return atoms[slab_idx], atoms[ads_idx]


def _kabsch_align_moving_to_reference(
    reference: np.ndarray, moving: np.ndarray
) -> tuple[np.ndarray, float]:
    """Align moving onto reference (permutation-aware Kabsch). Returns positions and RMSD."""
    ref = reference - reference.mean(axis=0)
    mob = moving - moving.mean(axis=0)
    cost = np.linalg.norm(ref[:, None, :] - mob[None, :, :], axis=2)
    row_idx, col_idx = linear_sum_assignment(cost)
    ref_sel = ref[row_idx]
    mob_matched = mob[col_idx]

    covariance = ref_sel.T @ mob_matched
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    mob_aligned = mob_matched @ rotation.T
    ref_origin = reference.mean(axis=0)
    positions = mob_aligned + ref_origin
    rmsd = float(np.sqrt(np.mean(np.sum((ref_sel - mob_aligned) ** 2, axis=1))))
    return positions, rmsd


def adsorbate_rmsd_kabsch(a: Atoms, b: Atoms) -> float:
    """Permutation-aware Kabsch RMSD between two camphor conformers (Å)."""
    pa = a.get_positions()
    pb = b.get_positions()
    if len(pa) != len(pb):
        return float("inf")
    _, rmsd = _kabsch_align_moving_to_reference(pa, pb)
    return rmsd


def align_adsorbate_onto_reference(reference: Atoms, moving: Atoms) -> np.ndarray:
    """Return moving adsorbate Cartesian positions aligned onto reference (Å)."""
    ref = reference.get_positions()
    mob = moving.get_positions()
    if len(ref) != len(mob):
        raise ValueError(
            f"Adsorbate atom count mismatch: reference={len(ref)}, moving={len(mob)}"
        )
    aligned, _ = _kabsch_align_moving_to_reference(ref, mob)
    return aligned


def tag_full_adslab(atoms: Atoms, *, source: str) -> Atoms:
    """Tag a full adslab for VMD export (0=slab, 1=adsorbate)."""
    tagged = atoms.copy()
    symbols = tagged.get_chemical_symbols()
    tags = np.array([0 if sym == "Cu" else 1 for sym in symbols], dtype=int)
    tagged.set_array("tags", tags)
    tagged.info["source"] = source
    tagged.info["tag_legend"] = "0=slab, 1=adsorbate"
    return tagged


def uses_dft_reference_slab(results_directory: str | Path) -> bool:
    clean_slab = Path(results_directory) / "clean_slab.xyz"
    if not clean_slab.exists():
        return False
    return len(read(clean_slab)) == PAPER_SLAB_ATOMS


def build_slab_ads_structure(
    slab: Atoms,
    adsorbate: Atoms,
    *,
    source: str,
    cell: Atoms,
) -> Atoms:
    """Full slab + adsorbate with tags for VMD (0=slab, 1=adsorbate)."""
    symbols = slab.get_chemical_symbols() + adsorbate.get_chemical_symbols()
    positions = np.vstack((slab.get_positions(), adsorbate.get_positions()))
    tags = [TAG_SLAB] * len(slab) + [TAG_ADSORBATE] * len(adsorbate)
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell.cell, pbc=cell.pbc)
    atoms.set_array("tags", np.array(tags, dtype=int))
    atoms.info["source"] = source
    atoms.info["tag_legend"] = "0=slab, 1=adsorbate"
    return atoms


def binding_geometry(atoms: Atoms) -> BindingGeometry:
    slab, ads = split_slab_adsorbate(atoms)
    slab_top = float(np.max(slab.get_positions()[:, 2]))
    ads_pos = ads.get_positions()
    ads_syms = ads.get_chemical_symbols()
    com_height = float(np.mean(ads_pos[:, 2]) - slab_top)

    tree = cKDTree(slab.get_positions())
    o_idx = [i for i, sym in enumerate(ads_syms) if sym == "O"]
    h_idx = [i for i, sym in enumerate(ads_syms) if sym == "H"]
    o_dist = float(tree.query(ads_pos[o_idx[0]], k=1)[0]) if o_idx else float("inf")
    h_dist = (
        float(min(tree.query(ads_pos[i], k=1)[0] for i in h_idx))
        if h_idx
        else float("inf")
    )
    mode = "Ox" if o_dist <= h_dist else "Hy"
    return BindingGeometry(
        com_height=com_height,
        o_distance=o_dist,
        h_distance=h_dist,
        mode=mode,
    )


def load_results_from_disk(
    results_directory: str | None = None,
) -> list[ScreeningResult]:
    """Load relaxed placements from a previous results directory."""
    base = results_directory or RESULTS_DIR
    detailed_csv = Path(base) / "adsorption_energies_detailed.csv"
    if not detailed_csv.exists():
        return []

    results: list[ScreeningResult] = []
    with detailed_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            xyz_path = row.get("xyz_path")
            if not xyz_path:
                continue
            path = Path(xyz_path)
            if not path.is_absolute():
                path = Path.cwd() / path
            if not path.exists():
                continue
            results.append(
                ScreeningResult(
                    molecule=row["molecule"],
                    placement_id=int(row["placement_id"]),
                    energy_adslab=float(row["energy_adslab"]),
                    energy_slab=float(row["energy_slab"]),
                    energy_adsorbate=float(row["energy_adsorbate"]),
                    energy_adsorption=float(row["energy_adsorption"]),
                    atoms=read(path),
                    slab_size=0,
                    distance=float(row["distance"]),
                    placement_descriptor=None,  # type: ignore[arg-type]
                )
            )
    return results


def _paper_binding_mode(binding_class: str) -> str:
    return "Ox" if binding_class == "oxygen" else "Hy"


def compare_geometries_to_nomad(
    results: list[ScreeningResult],
    references: list[ReferenceMinimum],
) -> list[GeometryMatch]:
    ref_ads = {
        ref.label: split_slab_adsorbate(read(ref.geometry_path, format="aims"))[1]
        for ref in references
    }
    ref_geom = {
        ref.label: binding_geometry(read(ref.geometry_path, format="aims"))
        for ref in references
    }
    ref_mode = {ref.label: _paper_binding_mode(ref.binding_class) for ref in references}

    matches: list[GeometryMatch] = []
    for result in sorted(results, key=lambda row: row.energy_adsorption):
        _, mlip_ads = split_slab_adsorbate(result.atoms)
        mlip_geom = binding_geometry(result.atoms)

        best_label = ""
        best_rmsd = float("inf")
        for label, dft_ads in ref_ads.items():
            rmsd = adsorbate_rmsd_kabsch(mlip_ads, dft_ads)
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_label = label

        dft_geom = ref_geom[best_label]
        dft_mode = ref_mode[best_label]
        matches.append(
            GeometryMatch(
                placement_id=result.placement_id,
                best_label=best_label,
                adsorbate_rmsd=best_rmsd,
                com_height_delta=mlip_geom.com_height - dft_geom.com_height,
                mlip_mode=mlip_geom.mode,
                dft_mode=dft_mode,
                mode_match=mlip_geom.mode == dft_mode,
            )
        )
    return matches


TAG_SLAB = 0
TAG_ADSORBATE = 1
TAG_MLIP_ADS = 1
TAG_DFT_ADS = 2


@dataclass(frozen=True)
class OverlayExport:
    label: str
    placement_id: int
    adsorbate_rmsd: float
    mlip_energy_adsorption: float
    binding_class: str
    full_path: Path
    slab_patch_path: Path
    vmd_mlip_path: Path
    vmd_dft_path: Path


def _slab_patch_indices(slab: Atoms, adsorbate: Atoms, cutoff: float) -> np.ndarray:
    ads_pos = adsorbate.get_positions()
    slab_pos = slab.get_positions()
    tree = cKDTree(ads_pos)
    distances, _ = tree.query(slab_pos, k=1)
    return np.where(distances <= cutoff)[0]


def build_overlay_structure(
    slab: Atoms,
    mlip_ads: Atoms,
    dft_ads_aligned: np.ndarray,
    *,
    slab_indices: np.ndarray | None = None,
) -> Atoms:
    """Assemble slab + MLIP adsorbate + Kabsch-aligned DFT adsorbate for figure export."""
    slab_sel = slab if slab_indices is None else slab[slab_indices]
    n_slab = len(slab_sel)
    n_ads = len(mlip_ads)

    symbols = slab_sel.get_chemical_symbols() + mlip_ads.get_chemical_symbols()
    symbols += mlip_ads.get_chemical_symbols()
    positions = np.vstack(
        (
            slab_sel.get_positions(),
            mlip_ads.get_positions(),
            dft_ads_aligned,
        )
    )
    tags = [TAG_SLAB] * n_slab + [TAG_MLIP_ADS] * n_ads + [TAG_DFT_ADS] * n_ads

    overlay = Atoms(
        symbols=symbols,
        positions=positions,
        cell=slab.cell,
        pbc=slab.pbc,
    )
    overlay.set_array("tags", np.array(tags, dtype=int))
    overlay.info["tag_legend"] = (
        "0=slab, 1=metalsurfer adsorbate, 2=DFT adsorbate (Kabsch-aligned onto MLIP)"
    )
    return overlay


def export_figure_overlays(
    results: list[ScreeningResult],
    references: list[ReferenceMinimum],
    out_dir: str | Path,
    *,
    slab_patch_cutoff: float = 6.0,
    native_frame: bool = False,
) -> list[OverlayExport]:
    """Write overlay XYZ files pairing each DFT minimum with its best metalsurfer match."""
    base = Path(out_dir) / "figure_overlays"
    full_dir = base / "full"
    patch_dir = base / "slab_patch"
    vmd_dir = base / "vmd_pairs"
    for directory in (full_dir, patch_dir, vmd_dir):
        directory.mkdir(parents=True, exist_ok=True)

    results_by_id = {result.placement_id: result for result in results}
    ref_by_label = {ref.label: ref for ref in references}
    coverage = reference_coverage_matrix(results, references)
    class_by_label = {label: cls for label, cls, _ in PAPER_DFT_MINIMA}

    exports: list[OverlayExport] = []
    manifest_rows: list[dict[str, object]] = []

    for label in sorted(coverage, key=lambda key: coverage[key][0]):
        best_rmsd, placement_id = coverage[label]
        if placement_id < 0 or not np.isfinite(best_rmsd):
            continue

        result = results_by_id[placement_id]
        ref = ref_by_label[label]
        mlip_atoms = result.atoms
        mlip_slab, mlip_ads = split_slab_adsorbate(mlip_atoms)
        dft_atoms = read(ref.geometry_path, format="aims")
        _, dft_ads = split_slab_adsorbate(dft_atoms)
        if native_frame:
            dft_ads_positions = dft_ads.get_positions()
        else:
            dft_ads_positions = align_adsorbate_onto_reference(mlip_ads, dft_ads)
        dft_ads_on_mlip = Atoms(
            symbols=dft_ads.get_chemical_symbols(),
            positions=dft_ads_positions,
        )

        stem = f"{label}_placement{placement_id}_rmsd{best_rmsd:.3f}"
        full_path = full_dir / f"{stem}.xyz"
        patch_path = patch_dir / f"{stem}.xyz"
        vmd_mlip_path = vmd_dir / f"{stem}_mlip.xyz"
        vmd_dft_path = (
            vmd_dir / f"{stem}_dft.xyz"
            if native_frame
            else vmd_dir / f"{stem}_dft_ads_on_mlip_slab.xyz"
        )

        full_overlay = build_overlay_structure(mlip_slab, mlip_ads, dft_ads_positions)
        patch_idx = _slab_patch_indices(mlip_slab, mlip_ads, slab_patch_cutoff)
        patch_overlay = build_overlay_structure(
            mlip_slab, mlip_ads, dft_ads_positions, slab_indices=patch_idx
        )

        if native_frame:
            mlip_vmd = tag_full_adslab(mlip_atoms, source="metalsurfer")
            dft_vmd = tag_full_adslab(dft_atoms, source="dft")
        else:
            mlip_vmd = build_slab_ads_structure(
                mlip_slab, mlip_ads, source="metalsurfer", cell=mlip_atoms
            )
            dft_vmd = build_slab_ads_structure(
                mlip_slab,
                dft_ads_on_mlip,
                source="dft_adsorbate_kabsch",
                cell=mlip_atoms,
            )

        metadata = {
            "dft_label": label,
            "placement_id": placement_id,
            "adsorbate_rmsd_A": round(best_rmsd, 4),
            "rmsd_method": "adsorbate_only_kabsch",
            "native_frame": native_frame,
            "shared_slab": "metalsurfer"
            if not native_frame
            else "paper_cell_both_native",
            "mlip_E_ads_eV": round(result.energy_adsorption, 4),
            "dft_E_ads_eV": next(e for lbl, _, e in PAPER_DFT_MINIMA if lbl == label),
            "nomad_calc_id": ref.calculation.calc_id,
        }
        for atoms, path in (
            (full_overlay, full_path),
            (patch_overlay, patch_path),
            (mlip_vmd, vmd_mlip_path),
            (dft_vmd, vmd_dft_path),
        ):
            atoms.info.update(metadata)
            write(path, atoms, format="extxyz")

        export = OverlayExport(
            label=label,
            placement_id=placement_id,
            adsorbate_rmsd=best_rmsd,
            mlip_energy_adsorption=result.energy_adsorption,
            binding_class=class_by_label[label],
            full_path=full_path,
            slab_patch_path=patch_path,
            vmd_mlip_path=vmd_mlip_path,
            vmd_dft_path=vmd_dft_path,
        )
        exports.append(export)
        manifest_rows.append(
            {
                **metadata,
                "binding_class": export.binding_class,
                "full_overlay": str(full_path),
                "slab_patch_overlay": str(patch_path),
                "vmd_mlip": str(vmd_mlip_path),
                "vmd_dft": str(vmd_dft_path),
            }
        )

    readme = base / "README.txt"
    readme.write_text(
        "\n".join(
            (
                "Camphor/Cu(111) DFT vs metalsurfer overlay structures",
                "======================================================",
                "",
                "RMSD in manifest is adsorbate-only (permutation-aware Kabsch).",
                "",
                "Subdirectories:",
                "  vmd_pairs/       — two files per pose for VMD (load both together)",
                "     *_mlip.xyz + *_dft.xyz (full adslabs, paper cell)",
                "  full/            — single-file overlay (tags 0/1/2) for OVITO",
                "  slab_patch/      — local Cu patch + both adsorbates (tags 0/1/2)",
                "",
                "VMD: color by file/source. tag 0=slab Cu, tag 1=adsorbate.",
                "",
                "OVITO (full/ and slab_patch/): tag 1=MLIP ads, tag 2=DFT ads, tag 0=Cu.",
                "",
                "DFT references: NOMAD 10.17172/NOMAD/2021.04.12-1",
                "Paper: Järvi et al., Beilstein J. Nanotechnol. 2020, 11, 140",
                "",
                f"Exported {len(exports)} overlay(s). See manifest.json for placement IDs and RMSDs.",
            )
        )
        + "\n"
    )
    (base / "manifest.json").write_text(json.dumps(manifest_rows, indent=2) + "\n")
    return exports


def print_overlay_export_summary(
    exports: list[OverlayExport], out_dir: str | Path
) -> None:
    base = Path(out_dir) / "figure_overlays"
    print()
    print("=" * 72)
    print("Figure overlay XYZ export")
    print("=" * 72)
    print(f"Output directory: {base}")
    print("VMD pairs: vmd_pairs/ (native full adslabs, paper cell)")
    print("OVITO overlays: full/ and slab_patch/ (tags 0/1/2)")
    print("RMSD: adsorbate-only Kabsch (slabs not superposed)")
    print()
    print(f"{'Label':<6} {'Placement':>10} {'Ads RMSD (Å)':>14}")
    print("-" * 72)
    for export in sorted(exports, key=lambda row: row.adsorbate_rmsd):
        print(
            f"{export.label:<6} {export.placement_id:>10} "
            f"{export.adsorbate_rmsd:>14.3f}"
        )
    print()
    print(
        f"Wrote {len(exports)} poses: "
        f"{len(exports)} VMD pairs + {len(exports)} full + {len(exports)} slab_patch overlays."
    )
    print(f"Manifest: {base / 'manifest.json'}")


def reference_coverage_matrix(
    results: list[ScreeningResult],
    references: list[ReferenceMinimum],
) -> dict[str, tuple[float, int]]:
    """Best adsorbate RMSD and placement id for each DFT reference label."""
    ref_ads = {
        ref.label: split_slab_adsorbate(read(ref.geometry_path, format="aims"))[1]
        for ref in references
    }
    coverage: dict[str, tuple[float, int]] = {
        label: (float("inf"), -1) for label in ref_ads
    }
    for result in results:
        _, mlip_ads = split_slab_adsorbate(result.atoms)
        for label, dft_ads in ref_ads.items():
            rmsd = adsorbate_rmsd_kabsch(mlip_ads, dft_ads)
            best_rmsd, _ = coverage[label]
            if rmsd < best_rmsd:
                coverage[label] = (rmsd, result.placement_id)
    return coverage


def print_geometry_comparison(
    references: list[ReferenceMinimum],
    matches: list[GeometryMatch],
    coverage: dict[str, tuple[float, int]] | None = None,
) -> None:
    print()
    print("=" * 72)
    print("Geometry comparison vs NOMAD DFT references")
    print("=" * 72)
    print(
        "DFT structures: NOMAD dataset "
        f"{NOMAD_DATASET_DOI} (FHI-aims, 192-atom Cu(111) slab)."
    )
    print(
        "Adsorbate RMSD uses centered permutation-aware Kabsch alignment; "
        "slabs are not superposed (different cells)."
    )
    print()
    print(f"{'Label':<6} {'Calc':<6} {'Class':<8} {'E_B (eV)':>10}  NOMAD geometry.in")
    print("-" * 72)
    for ref in references:
        print(
            f"{ref.label:<6} {ref.calculation.calc_id:<6} {ref.binding_class:<8} "
            f"{ref.boss_eb:>10.4f}  {ref.geometry_path.name}"
        )

    if not matches:
        print("\nNo metalsurfer placements available for comparison.")
        return

    print()
    print(
        f"{'Placement':<12} {'Best DFT':<8} {'RMSD (Å)':>10} "
        f"{'Δz_COM':>10} {'MLIP':<6} {'Paper':<6} {'Mode':>6}"
    )
    print("-" * 72)
    for match in matches:
        mode_flag = "yes" if match.mode_match else "no"
        print(
            f"{match.placement_id:<12} {match.best_label:<8} "
            f"{match.adsorbate_rmsd:>10.3f} {match.com_height_delta:>+10.3f} "
            f"{match.mlip_mode:<6} {match.dft_mode:<6} {mode_flag:>6}"
        )

    rmsds = [match.adsorbate_rmsd for match in matches]
    mode_hits = sum(1 for match in matches if match.mode_match)
    print()
    print(
        f"Matched {len(matches)} placement(s); adsorbate RMSD "
        f"[{min(rmsds):.3f}, {max(rmsds):.3f}] Å; "
        f"binding-mode agreement {mode_hits}/{len(matches)}."
    )
    print(
        "Paper reports ~0.13 Å internal RMSD change upon full DFT relaxation of "
        "BOSS minima (SI Table SIII)."
    )

    if coverage:
        print()
        print("DFT reference coverage (best adsorbate RMSD per paper minimum):")
        print(f"{'Label':<6} {'Class':<8} {'Best RMSD (Å)':>14} {'Placement':>12}")
        print("-" * 44)
        class_by_label = {label: cls for label, cls, _ in PAPER_DFT_MINIMA}
        for label in sorted(coverage, key=lambda key: coverage[key][0]):
            best_rmsd, placement_id = coverage[label]
            placement = str(placement_id) if placement_id >= 0 else "—"
            rmsd_str = f"{best_rmsd:.3f}" if np.isfinite(best_rmsd) else "—"
            print(
                f"{label:<6} {class_by_label[label]:<8} {rmsd_str:>14} {placement:>12}"
            )
        finite = [rmsd for rmsd, _ in coverage.values() if np.isfinite(rmsd)]
        if finite:
            print(
                f"\nCoverage: {len(finite)}/{len(coverage)} references matched; "
                f"best RMSD range [{min(finite):.3f}, {max(finite):.3f}] Å."
            )


def print_paper_reference_table() -> None:
    print()
    print("=" * 60)
    print("Paper reference: 8 DFT minima (Beilstein J. Nanotechnol. 2020)")
    print("=" * 60)
    print(f"{'Label':<6} {'Class':<10} {'E_DFT (eV)':>12}")
    print("-" * 32)
    for label, cls, e_dft in PAPER_DFT_MINIMA:
        print(f"{label:<6} {cls:<10} {e_dft:>12.4f}")
    e_min = min(e for _, _, e in PAPER_DFT_MINIMA)
    e_max = max(e for _, _, e in PAPER_DFT_MINIMA)
    print(f"\nDFT range: [{e_min:.4f}, {e_max:.4f}] eV (PBE+vdWsurf)")


def print_found_minima(
    results: list[ScreeningResult],
    *,
    config: AdsorptionConfig,
    budget: ResolvedBoBudget | None,
) -> None:
    print()
    print("=" * 60)
    print("Metalsurfer unique minima (MLIP, after relax + dedup)")
    print("=" * 60)
    if budget is not None:
        print(
            f"BO budget: initial={budget.bo_initial_random}, "
            f"batch={budget.bo_batch_size}, "
            f"batches={budget.bo_total_budget} "
            f"(eval_budget={budget.eval_budget})"
        )
    else:
        print(
            f"BO acquisition batches: {config.bo_total_budget} "
            "(initial/batch sizes autotuned — see workflow log)"
        )
    print(
        "Note: MLIP absolute eV differ from paper DFT; compare count, spread, and ranking."
    )
    print()

    if not results:
        print("No valid minima found.")
        print(f"Paper reports {len(PAPER_DFT_MINIMA)} unique stable adsorbates.")
        return

    sorted_results = sorted(results, key=lambda r: r.energy_adsorption)
    print(f"{'Rank':<6} {'Placement':<12} {'E_ads (eV)':>12} {'Dist (Å)':>10}")
    print("-" * 44)
    for rank, result in enumerate(sorted_results, start=1):
        print(
            f"{rank:<6} {result.placement_id:<12} "
            f"{result.energy_adsorption:>12.4f} {result.distance:>10.3f}"
        )

    e_min = sorted_results[0].energy_adsorption
    e_max = sorted_results[-1].energy_adsorption
    print()
    print(
        f"Found {len(sorted_results)} unique minima; MLIP range [{e_min:.4f}, {e_max:.4f}] eV"
    )
    print(f"Paper reports {len(PAPER_DFT_MINIMA)} unique stable adsorbates.")


def run_geometry_comparison(
    results: list[ScreeningResult] | None = None,
    *,
    results_directory: str | None = None,
    export_overlays: bool = False,
) -> int:
    references = ensure_nomad_reference_structures()
    if results is None:
        results = load_results_from_disk(results_directory)
    matches = compare_geometries_to_nomad(results, references)
    coverage = reference_coverage_matrix(results, references)
    print_geometry_comparison(references, matches, coverage)
    if export_overlays:
        if not results_directory:
            raise ValueError("results_directory is required when export_overlays=True")
        native = uses_dft_reference_slab(results_directory)
        exports = export_figure_overlays(
            results,
            references,
            results_directory,
            native_frame=native,
        )
        print_overlay_export_summary(exports, results_directory)
    return 0


def _validate_campaign(campaign: BindingCampaignResult) -> None:
    if not campaign.molecule_summaries:
        print("No molecule summaries produced.", file=sys.stderr)
        raise SystemExit(1)

    summary = campaign.molecule_summaries[0]
    if summary.n_valid_placements < 5:
        print(
            f"Expected >= 5 valid BO minima, got {summary.n_valid_placements}.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    best = summary.best_adsorption_energy
    if best is None or best >= 0.0:
        print(
            f"Expected favorable camphor binding (best E_ads < 0 eV), got {best}.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def run_campaign(config: AdsorptionConfig) -> BindingCampaignResult:
    ensure_zenodo_reference_data()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    slab = prepare_campaign_slab(config, results_directory=RESULTS_DIR)
    write(f"{RESULTS_DIR}/clean_slab.xyz", slab.atoms, format="extxyz")

    return run_adsorption_bo(
        slab=slab,
        molecules=[(CAMPHOR_SMILES, MOLECULE_NAME)],
        config=config,
        surface_type=SURFACE_TYPE,
        system_name="Cu_111",
        skip_existing=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "(1S)-camphor on Cu(111): BO-guided local-minima search "
            "(Järvi et al. Beilstein J. Nanotechnol. 2020)"
        )
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("METALSURFER_DEVICE", "cuda"),
        help="Device: cuda or cpu",
    )
    parser.add_argument(
        "--compare-geometries",
        action="store_true",
        help="Compare existing results to NOMAD DFT geometries (skip MLIP run)",
    )
    parser.add_argument(
        "--export-overlays",
        action="store_true",
        help=(
            "Write figure overlay XYZ files (DFT adsorbate Kabsch-aligned onto each "
            "best-matching metalsurfer placement) under results/figure_overlays/"
        ),
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    debug = args.debug or (
        os.environ.get("METALSURFER_DEBUG", "").lower() in ("1", "true", "yes")
    )
    _configure_logging(debug=debug)
    device = args.device if args.device in ("cuda", "cpu") else "cuda"

    if args.compare_geometries or args.export_overlays:
        return run_geometry_comparison(
            results_directory=RESULTS_DIR,
            export_overlays=args.export_overlays,
        )

    config = build_config(device=device)
    campaign = run_campaign(config)
    _validate_campaign(campaign)

    print()
    print(
        campaign.format_summary(
            title="Binding energy summary (camphor / Cu(111))",
            results_dir=RESULTS_DIR,
        )
    )

    run_result = campaign.run_results[0] if campaign.run_results else None
    results = run_result.results if run_result is not None else []
    print_found_minima(results, config=config, budget=resolve_bo_budget(config))
    print_paper_reference_table()
    if results:
        run_geometry_comparison(
            results,
            results_directory=RESULTS_DIR,
            export_overlays=args.export_overlays,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
