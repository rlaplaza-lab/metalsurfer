"""Site-generator plugin ids and material allowlists (leaf module).

Kept free of ``config`` / ``generators`` imports so ``AdsorptionConfig`` and
``site_plugins.base`` can share one allowlist without circular imports.
"""

from __future__ import annotations

# Registered plugins (also the AdsorptionConfig / YAML selectable set).
SITE_GENERATORS: tuple[str, ...] = (
    "topology",
    "voronoi",
    "adaptive_grid",
    "rolling_probe",
)

# Material allowlist per plugin id (single source of truth for config validation).
PLUGIN_ALLOWED_MATERIALS: dict[str, frozenset[str]] = {
    "topology": frozenset({"slab", "nanoparticle"}),
    "voronoi": frozenset({"slab", "porous"}),
    "adaptive_grid": frozenset({"slab", "nanoparticle", "porous"}),
    "rolling_probe": frozenset({"slab", "nanoparticle", "porous"}),
}

AUTO_SITE_GENERATOR_DEFAULTS: dict[str, str] = {
    "slab": "topology",
    "nanoparticle": "topology",
    "porous": "voronoi",
}
