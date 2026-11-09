"""Ligand atop/bridge sites under periodic boundaries."""

import numpy as np
from ase import Atoms

from metalsurfer.placement.site_ligands import ligand_local_sites


def test_ligand_bridge_uses_mic_neighbor_across_the_cell_face():
    """A ligand near +a bonds to the substrate atom just inside a=0."""
    cell = np.diag([8.0, 8.0, 16.0])
    substrate = Atoms(
        "Ru2",
        positions=[[0.15, 4.0, 5.0], [2.35, 4.0, 5.0]],
        cell=cell,
        pbc=[True, True, False],
    )
    ligand = Atoms(
        "C", positions=[[7.25, 4.0, 7.0]], cell=cell, pbc=substrate.get_pbc()
    )
    full = substrate + ligand
    sites = ligand_local_sites(substrate, full, material_type="slab")
    bridges = [s for s in sites if s.site_type == "bridge"]
    assert bridges, "MIC neighbour across +a must produce a ligand–metal bridge"
    assert all(0 in s.slab_indices for s in bridges)
    assert all(1 not in s.slab_indices for s in bridges)
    atop = [s for s in sites if s.site_type == "atop"]
    assert atop
    assert float(atop[0].normal[2]) > 0.5
