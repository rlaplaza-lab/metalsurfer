"""Site generator plugin registry and resolution."""

import pytest

from metalsurfer.config import SITE_GENERATOR_OPTIONS
from metalsurfer.placement.site_enumeration import get_unified_sites
from metalsurfer.placement.site_plugins import (
    PUBLIC_SITE_GENERATORS,
    SITE_GENERATORS,
    resolve_site_generator,
    resolved_site_generator_name,
)
from metalsurfer.placement.site_plugins.adaptive_grid import AdaptiveGridGenerator
from metalsurfer.placement.site_plugins.topology_np import TopologyNPGenerator
from metalsurfer.placement.site_plugins.topology_slab import TopologySlabGenerator
from metalsurfer.placement.site_plugins.voronoi import VoronoiGenerator

from ..conftest import make_nanoparticle, make_porous_framework, make_slab


def test_registry_matches_config_options():
    assert SITE_GENERATORS == ("topology", "voronoi", "adaptive_grid")
    assert PUBLIC_SITE_GENERATORS == ("topology", "voronoi", "adaptive_grid")
    assert SITE_GENERATOR_OPTIONS == ("auto",) + PUBLIC_SITE_GENERATORS


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
    from dataclasses import fields

    import numpy as np

    from metalsurfer.placement.site_plugins.base import (
        SiteCandidateBatch,
        SiteGenerationContext,
    )
    from metalsurfer.placement.site_types import Site

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
        if batch.normals is not None:
            assert len(batch.normals) == n
        if batch.clearances is not None:
            assert len(batch.clearances) == n

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
