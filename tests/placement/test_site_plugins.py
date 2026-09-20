"""Site generator plugin registry and resolution."""

from dataclasses import fields

import numpy as np
import pytest
from ase.build import molecule

from metalsurfer.config import SITE_GENERATOR_OPTIONS, AdsorptionConfig
from metalsurfer.placement.generators import (
    enumerate_placement_specs,
    generate_placements_from_specs,
)
from metalsurfer.placement.site_enumeration import get_unified_sites
from metalsurfer.placement.site_plugins import (
    SITE_GENERATORS,
    resolve_site_generator,
    resolved_site_generator_name,
)
from metalsurfer.placement.site_plugins.adaptive_grid import AdaptiveGridGenerator
from metalsurfer.placement.site_plugins.base import (
    SiteCandidateBatch,
    SiteGenerationContext,
)
from metalsurfer.placement.site_plugins.topology_np import TopologyNPGenerator
from metalsurfer.placement.site_plugins.topology_slab import TopologySlabGenerator
from metalsurfer.placement.site_plugins.voronoi import VoronoiGenerator
from metalsurfer.placement.site_types import Site

from ..conftest import make_nanoparticle, make_porous_framework, make_slab


def test_registry_matches_config_options():
    assert SITE_GENERATORS == ("topology", "voronoi", "adaptive_grid")
    assert SITE_GENERATOR_OPTIONS == ("auto",) + SITE_GENERATORS


def test_auto_defaults_by_material():
    assert resolved_site_generator_name("auto", "slab") == "topology"
    assert resolved_site_generator_name("auto", "nanoparticle") == "topology"
    assert resolved_site_generator_name("auto", "porous") == "voronoi"
    assert resolved_site_generator_name("adaptive_grid", "slab") == "adaptive_grid"


@pytest.mark.parametrize(
    ("name", "material_type", "cls"),
    [
        ("auto", "slab", TopologySlabGenerator),
        ("topology", "nanoparticle", TopologyNPGenerator),
        ("voronoi", "porous", VoronoiGenerator),
        ("voronoi", "slab", VoronoiGenerator),
        ("adaptive_grid", "slab", AdaptiveGridGenerator),
        ("adaptive_grid", "nanoparticle", AdaptiveGridGenerator),
        ("adaptive_grid", "porous", AdaptiveGridGenerator),
    ],
)
def test_resolve_plugin(name, material_type, cls):
    assert isinstance(resolve_site_generator(name, material_type), cls)


def test_unknown_and_incompatible_raise():
    with pytest.raises(ValueError, match="Unknown site_generator"):
        resolve_site_generator("rolling_probe", "slab")
    with pytest.raises(ValueError, match="incompatible"):
        resolve_site_generator("topology", "porous")
    with pytest.raises(ValueError, match="incompatible"):
        resolve_site_generator("voronoi", "nanoparticle")


@pytest.mark.parametrize(
    ("material_type", "explicit", "factory"),
    [
        ("slab", "topology", make_slab),
        ("nanoparticle", "topology", make_nanoparticle),
        ("porous", "voronoi", make_porous_framework),
    ],
)
def test_auto_matches_explicit_plugin(material_type, explicit, factory):
    atoms = factory()
    auto = get_unified_sites(atoms, material_type=material_type, site_generator="auto")
    named = get_unified_sites(
        atoms, material_type=material_type, site_generator=explicit
    )
    assert len(auto) == len(named) > 0
    assert [s.site_type for s in auto] == [s.site_type for s in named]


def test_all_plugins_share_batch_and_site_contract():
    """Every plugin emits SiteCandidateBatch; enumerator yields placement-ready Sites."""
    core = {"vertices", "nn_dists", "source_hints", "atom_indices"}
    enrich = {"normals", "clearances"}
    names = {f.name for f in fields(SiteCandidateBatch)}
    assert core <= names
    assert enrich <= names
    assert "env_fingerprints" not in names

    slab = make_slab(nx=2, ny=2, n_layers=2)
    pos = slab.get_positions()
    cell = np.asarray(slab.get_cell(), dtype=float)
    pbc = np.array([True, True, False])
    ctx = SiteGenerationContext(
        positions=pos,
        cell=cell,
        pbc=pbc,
        symbols=list(slab.get_chemical_symbols()),
        material_type="slab",
        probe_radius=1.2,
        max_site_distance=3.5,
        top_layer_tolerance=1.0,
        enrich=True,
        planar_z_variance_threshold=0.1,
        n_jobs=1,
    )
    for name in ("topology", "voronoi", "adaptive_grid"):
        batch = resolve_site_generator(name, "slab").generate(ctx)
        assert isinstance(batch, SiteCandidateBatch)
        n = len(batch.vertices)
        assert len(batch.nn_dists) == n
        assert len(batch.source_hints) == n
        assert len(batch.atom_indices) == n
        assert batch.normals is not None and len(batch.normals) == n
        assert batch.clearances is not None and len(batch.clearances) == n

        sites = get_unified_sites(
            slab,
            material_type="slab",
            site_generator=name,
            n_jobs=1,
            probe_radius=1.2,
            max_site_distance=3.5,
        )
        assert sites
        assert all(isinstance(s, Site) for s in sites)
        for s in sites:
            assert s.xyz.shape == (3,)
            assert s.normal.shape == (3,)
            assert len(s.env_fingerprint) == 3
            assert s.site_type
            assert s.tangent_basis is not None
            assert np.asarray(s.tangent_basis).shape == (2, 3)
            assert s.clearance is not None


def test_slab_topology_emits_nonempty_supports():
    slab = make_slab(nx=3, ny=3, n_layers=2)
    ctx = SiteGenerationContext(
        positions=slab.get_positions(),
        cell=np.asarray(slab.get_cell(), dtype=float),
        pbc=np.array([True, True, False]),
        symbols=list(slab.get_chemical_symbols()),
        material_type="slab",
        probe_radius=1.2,
        max_site_distance=3.5,
        top_layer_tolerance=1.0,
        enrich=False,
        planar_z_variance_threshold=0.1,
        n_jobs=1,
    )
    batch = resolve_site_generator("topology", "slab").generate(ctx)
    assert any(h.startswith("topology_") for h in batch.source_hints)
    for hint, atoms in zip(batch.source_hints, batch.atom_indices, strict=True):
        if hint.startswith("topology_"):
            assert atoms
    sites = get_unified_sites(
        slab, material_type="slab", site_generator="topology", n_jobs=1
    )
    assert all(s.slab_indices for s in sites)


def test_voronoi_porous_keeps_empty_supports_and_pores():
    atoms = make_porous_framework()
    ctx = SiteGenerationContext(
        positions=atoms.get_positions(),
        cell=np.asarray(atoms.get_cell(), dtype=float),
        pbc=np.array([True, True, True]),
        symbols=list(atoms.get_chemical_symbols()),
        material_type="porous",
        probe_radius=1.5,
        max_site_distance=4.0,
        top_layer_tolerance=1.0,
        enrich=True,
        planar_z_variance_threshold=0.1,
        n_jobs=1,
    )
    batch = resolve_site_generator("voronoi", "porous").generate(ctx)
    assert batch.atom_indices and all(len(a) == 0 for a in batch.atom_indices)
    sites = get_unified_sites(
        atoms, material_type="porous", site_generator="voronoi", n_jobs=1
    )
    assert any(s.site_type == "pore" for s in sites)


@pytest.mark.parametrize(
    ("material_type", "plugin", "factory"),
    [
        ("slab", "topology", make_slab),
        ("slab", "adaptive_grid", make_slab),
        ("nanoparticle", "topology", make_nanoparticle),
        ("nanoparticle", "adaptive_grid", make_nanoparticle),
        ("porous", "voronoi", make_porous_framework),
        ("porous", "adaptive_grid", make_porous_framework),
    ],
)
def test_plugins_materialize_clash_free(material_type, plugin, factory):
    atoms = factory()
    cfg = AdsorptionConfig(
        material_type=material_type,
        site_generator=plugin,
        seed=0,
        num_conformers=1,
        num_placements=8,
        n_jobs=1,
        slab_relaxation_mode="none",
    )
    ads = molecule("H2")
    specs = enumerate_placement_specs([ads], atoms, cfg, "H2", n_desired=8, seed=0)
    assert specs
    results = generate_placements_from_specs(specs, [ads], atoms, cfg, smiles="H2")
    assert any(pair is not None for pair, _reason in results)


def test_voronoi_ridge_enrich_n_jobs_deterministic():
    atoms = make_porous_framework()
    serial = get_unified_sites(
        atoms, material_type="porous", site_generator="voronoi", enrich=True, n_jobs=1
    )
    parallel = get_unified_sites(
        atoms, material_type="porous", site_generator="voronoi", enrich=True, n_jobs=2
    )
    assert len(serial) == len(parallel) > 0
    assert sorted(tuple(np.round(s.xyz, 6)) for s in serial) == sorted(
        tuple(np.round(s.xyz, 6)) for s in parallel
    )


@pytest.mark.parametrize(
    ("material_type", "plugin", "factory"),
    [
        ("slab", "topology", make_slab),
        ("slab", "adaptive_grid", make_slab),
        ("nanoparticle", "topology", make_nanoparticle),
        ("nanoparticle", "adaptive_grid", make_nanoparticle),
        ("porous", "voronoi", make_porous_framework),
        ("porous", "adaptive_grid", make_porous_framework),
    ],
)
def test_plugins_n_tuplet_expands_and_materializes(material_type, plugin, factory):
    """n-tuplet sampling expands to the clustered lattice and place ≥2 clash-free."""
    from metalsurfer.placement.site_context import (
        resolve_site_context_for_sampling,
        site_context_for_sampling,
        skip_symmetry_for_sampling,
    )

    atoms = factory()
    cfg = AdsorptionConfig(
        material_type=material_type,
        site_generator=plugin,
        seed=0,
        num_conformers=1,
        num_placements=8,
        n_jobs=1,
        slab_relaxation_mode="none",
        saturation_molecules_per_step=2,
    )
    assert (
        skip_symmetry_for_sampling(
            symmetry_broken=False,
            slab_for_sites=atoms,
            full_slab=atoms,
            config=cfg,
        )
        is True
    )
    reduced = resolve_site_context_for_sampling(atoms, cfg, symmetry_broken=False)
    sampling = site_context_for_sampling(atoms, cfg, reduced)
    assert sampling.clustered_sites is not None
    assert len(sampling.sites) == len(sampling.clustered_sites)
    assert len(sampling.sites) >= 2

    ads = molecule("H2")
    specs = enumerate_placement_specs(
        [ads], atoms, cfg, "H2", n_desired=8, seed=0, site_context=sampling
    )
    assert len(specs) >= 2
    site_indices = {s.site_index for s in specs if s.site_index is not None}
    assert len(site_indices) >= 2
    results = generate_placements_from_specs(
        specs, [ads], atoms, cfg, smiles="H2", site_context=sampling
    )
    ok = [pair for pair, _reason in results if pair is not None]
    assert len(ok) >= 2


@pytest.mark.parametrize(
    ("material_type", "plugin", "factory"),
    [
        ("slab", "topology", make_slab),
        ("slab", "adaptive_grid", make_slab),
        ("nanoparticle", "topology", make_nanoparticle),
        ("nanoparticle", "adaptive_grid", make_nanoparticle),
        ("porous", "voronoi", make_porous_framework),
        ("porous", "adaptive_grid", make_porous_framework),
    ],
)
def test_plugins_multimol_share_catalog_materialize(material_type, plugin, factory):
    """Two molecules share one SiteContext and both materialize clash-free."""
    from metalsurfer.placement.site_context import resolve_site_context_for_sampling

    atoms = factory()
    cfg = AdsorptionConfig(
        material_type=material_type,
        site_generator=plugin,
        seed=0,
        num_conformers=1,
        num_placements=6,
        n_jobs=1,
        slab_relaxation_mode="none",
        multi_molecule_saturation=True,
    )
    shared = resolve_site_context_for_sampling(atoms, cfg, symmetry_broken=False)
    assert shared.sites

    h2 = molecule("H2")
    co = molecule("CO")
    for ads, smiles in ((h2, "H2"), (co, "CO")):
        specs = enumerate_placement_specs(
            [ads], atoms, cfg, smiles, n_desired=6, seed=0, site_context=shared
        )
        assert specs
        results = generate_placements_from_specs(
            specs, [ads], atoms, cfg, smiles=smiles, site_context=shared
        )
        assert any(pair is not None for pair, _reason in results)
