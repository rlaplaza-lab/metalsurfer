"""Site generator plugin registry and resolution."""

import pytest

from metalsurfer.config import SITE_GENERATOR_OPTIONS
from metalsurfer.placement.site_enumeration import get_unified_sites
from metalsurfer.placement.site_plugins import (
    SITE_GENERATORS,
    resolve_site_generator,
    resolved_site_generator_name,
)
from metalsurfer.placement.site_plugins.topology_np import TopologyNPGenerator
from metalsurfer.placement.site_plugins.topology_slab import TopologySlabGenerator
from metalsurfer.placement.site_plugins.voronoi import VoronoiGenerator

from ..conftest import make_nanoparticle, make_porous_framework, make_slab


def test_registry_matches_config_options():
    assert SITE_GENERATORS == ("topology", "voronoi")
    assert SITE_GENERATOR_OPTIONS == ("auto",) + SITE_GENERATORS


def test_auto_defaults_by_material():
    assert resolved_site_generator_name("auto", "slab") == "topology"
    assert resolved_site_generator_name("auto", "nanoparticle") == "topology"
    assert resolved_site_generator_name("auto", "porous") == "voronoi"


@pytest.mark.parametrize(
    ("name", "material_type", "cls"),
    [
        ("auto", "slab", TopologySlabGenerator),
        ("topology", "nanoparticle", TopologyNPGenerator),
        ("voronoi", "porous", VoronoiGenerator),
        ("voronoi", "slab", VoronoiGenerator),
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
