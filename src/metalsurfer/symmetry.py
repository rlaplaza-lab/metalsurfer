"""Equivalent adsorption sites via `spglib` (periodic cell or padded cluster-in-box).

Slab symmetry follows the 3D ASE supercell (not layer groups). `symmetry_tolerance`
is `symprec` for spglib and the Cartesian threshold for site matching.
"""


import hashlib
from typing import Any, Literal

import numpy as np
import spglib
import spglib.error as _spglib_error_module
from ase import Atoms

from ._numeric_defaults import DEFAULT_SYMMETRY_TOLERANCE
from .placement.site_types import Site, with_symmetry

# Opt into the new spglib error handling (raises SpglibError instead of
# returning None) and suppress the DeprecationWarning it would emit otherwise.
_spglib_error_module.OLD_ERROR_HANDLING = False

SymmetryMode = Literal["auto", "periodic", "cluster"]


class SymmetryAnalysisError(RuntimeError):
    """Raised when symmetry data is missing, invalid, or internal checks fail."""


class SymmetryAnalyzer:
    """Equivalent adsorption sites: periodic ASE cell, or cluster-in-box (see module doc)."""

    def __init__(
        self,
        atoms: Atoms,
        symmetry_tolerance: float = DEFAULT_SYMMETRY_TOLERANCE,
        mode: SymmetryMode = "auto",
        *,
        angle_tolerance: float | None = None,
    ):
        self.atoms = atoms
        self.symmetry_tolerance = float(symmetry_tolerance)
        self.symprec = max(self.symmetry_tolerance, 1e-5)
        self._angle_tolerance = angle_tolerance
        self.pbc = atoms.get_pbc()
        self.cell = np.asarray(atoms.get_cell(), dtype=float)
        self.positions = np.asarray(atoms.get_positions(), dtype=float)
        self.symbols = atoms.get_chemical_symbols()
        self.numbers = np.array(atoms.get_atomic_numbers(), dtype=int)

        if mode == "auto":
            pbc_arr = np.asarray(self.pbc, dtype=bool)
            self._mode: Literal["periodic", "cluster"] = (
                "cluster" if not np.any(pbc_arr) else "periodic"
            )
        elif mode == "cluster":
            self._mode = "cluster"
        else:
            self._mode = "periodic"

        self._lattice: np.ndarray
        self._fractional: np.ndarray
        self._cluster_com: np.ndarray | None = None
        self._cluster_half: np.ndarray | None = None

        self._prepare_lattice_and_fractional()

        self._symmetry_operations: list[np.ndarray] | None = None
        self._equivalent_atoms: list[list[int]] | None = None
        self._dataset: Any | None = None
        self._operations_frac: list[tuple[np.ndarray, np.ndarray]] | None = None

    def get_spglib_cell_tuple(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(lattice, fractional_positions, atomic_numbers)`` passed to spglib."""
        return (
            np.asarray(self._lattice, dtype=float).copy(),
            np.asarray(self._fractional, dtype=float).copy(),
            np.asarray(self.numbers, dtype=int).copy(),
        )

    def _prepare_lattice_and_fractional(self) -> None:
        if self._mode == "periodic":
            if self.cell.shape != (3, 3) or np.linalg.det(self.cell) <= 0:
                raise ValueError(
                    "Periodic symmetry requires a valid 3x3 cell with det > 0"
                )
            self._lattice = self.cell.copy()
            inv = np.linalg.inv(self._lattice)
            self._fractional = self.positions @ inv.T
            return

        # Cluster: orthorhombic box, atoms centered
        com = self.positions.mean(axis=0)
        rel = self.positions - com
        margin = max(8.0 * self.symprec, 5.0)
        half = np.max(np.abs(rel), axis=0) + margin
        half = np.maximum(half, margin)
        self._cluster_com = com
        self._cluster_half = half
        self._lattice = np.diag(2.0 * half)
        cart_in_box = rel + half
        self._fractional = cart_in_box / (2.0 * half)

    def _spglib_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {"symprec": self.symprec}
        if self._angle_tolerance is not None:
            kw["angle_tolerance"] = float(self._angle_tolerance)
        return kw

    def _ensure_dataset(self) -> Any:
        if self._dataset is not None:
            return self._dataset
        cell_tuple = (
            self._lattice,
            self._fractional,
            self.numbers,
        )
        try:
            self._dataset = spglib.get_symmetry_dataset(
                cell_tuple,  # type: ignore[arg-type]
                **self._spglib_kwargs(),
            )
        except spglib.SpglibError as exc:
            det = float(np.linalg.det(self._lattice))
            raise SymmetryAnalysisError(
                f"spglib.get_symmetry_dataset failed: {exc} "
                f"(mode={self._mode}, n_atoms={len(self.numbers)}, "
                f"pbc={np.asarray(self.pbc, dtype=bool).tolist()}, det={det:.6g}, "
                f"symprec={self.symprec}, angle_tolerance={self._angle_tolerance})"
            ) from exc
        if self._dataset is None:
            raise SymmetryAnalysisError("spglib.get_symmetry_dataset returned None")
        return self._dataset

    def _frac_ops_from_dataset(self) -> list[tuple[np.ndarray, np.ndarray]]:
        if self._operations_frac is not None:
            return self._operations_frac
        ds = self._ensure_dataset()
        rots = np.asarray(ds.rotations, dtype=float)
        trans = np.asarray(ds.translations, dtype=float)
        ops: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(len(rots)):
            ops.append((rots[i], trans[i]))
        self._operations_frac = ops
        return self._operations_frac

    def _symop_to_cartesian_4x4(
        self, R_frac: np.ndarray, t_frac: np.ndarray
    ) -> np.ndarray:
        """Column-vector convention: r_frac' = R @ r_frac + t; r_cart = L @ r_frac."""
        L = self._lattice
        Linv = np.linalg.inv(L)
        R_cart = L @ R_frac @ Linv
        t_cart = L @ t_frac.reshape(3)
        T = np.eye(4)
        T[:3, :3] = R_cart
        T[:3, 3] = t_cart
        return T

    def _dedupe_4x4(self, ops: list[np.ndarray], eps: float = 1e-5) -> list[np.ndarray]:
        unique: list[np.ndarray] = []
        for op in ops:
            flat = op.flatten()
            if not any(np.max(np.abs(flat - u.flatten())) < eps for u in unique):
                unique.append(op)
        return unique

    def detect_symmetry_operations(self) -> list[np.ndarray]:
        """Symmetry operations as 4x4 matrices [R|t; 0|1] in Cartesian coordinates."""
        if self._symmetry_operations is not None:
            return self._symmetry_operations

        frac_ops = self._frac_ops_from_dataset()
        cart_ops = [self._symop_to_cartesian_4x4(R, t) for R, t in frac_ops]
        self._symmetry_operations = self._dedupe_4x4(cart_ops)
        return self._symmetry_operations

    def _is_periodic_xy(self) -> bool:
        if self._mode != "periodic":
            return False
        pbc = np.asarray(self.pbc, dtype=bool)
        return bool(pbc[0] and pbc[1])

    def _cart_to_frac(self, cart: np.ndarray) -> np.ndarray:
        inv = np.linalg.inv(self._lattice)
        return cart @ inv.T

    def _wrap_frac(self, frac: np.ndarray) -> np.ndarray:
        if self._mode == "periodic":
            return frac % 1.0
        return frac

    def _apply_frac_symop(
        self, frac_row: np.ndarray, R: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        """Apply r' = r @ R.T + t for row vectors (matches spglib Python examples)."""
        return frac_row @ R.T + t

    def _mic_frac_delta(self, fa: np.ndarray, fb: np.ndarray) -> np.ndarray:
        """Shortest fractional difference (MIC in the spglib unit cell)."""
        d = np.asarray(fa, dtype=float) - np.asarray(fb, dtype=float)
        for k in range(3):
            d[k] -= np.round(d[k])
        return d

    def _cart_sep_from_frac_delta(self, d_frac: np.ndarray) -> np.ndarray:
        """Cartesian separation vector (row) from fractional MIC difference."""
        return d_frac @ self._lattice.T

    def _slab_normal(self) -> np.ndarray:
        """Unit normal from lattice a × b (slab plane)."""
        a = np.asarray(self._lattice[0], dtype=float)
        b = np.asarray(self._lattice[1], dtype=float)
        n = np.cross(a, b)
        norm_n = float(np.linalg.norm(n))
        if norm_n < 1e-12:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return n / norm_n

    def _separation_distance(self, sep: np.ndarray, planar: bool) -> float:
        """Cartesian MIC distance; when *planar*, drop the slab-normal component."""
        if not planar:
            return float(np.linalg.norm(sep))
        n = self._slab_normal()
        sep_plane = sep - float(np.dot(sep, n)) * n
        return float(np.linalg.norm(sep_plane))

    def _site_distance_under_symop(
        self,
        frac_i: np.ndarray,
        frac_j: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        planar: bool,
    ) -> float:
        frac_p = self._apply_frac_symop(frac_i, R, t)
        if self._mode == "periodic":
            frac_p = self._wrap_frac(frac_p)
        d_frac = self._mic_frac_delta(frac_p, frac_j)
        sep = self._cart_sep_from_frac_delta(d_frac)
        return self._separation_distance(sep, planar)

    def _site_pair_connected_by_ops(
        self,
        i: int,
        j: int,
        cart_pts: list[np.ndarray],
        frac_ops: list[tuple[np.ndarray, np.ndarray]],
        planar: bool,
        site_types: list[str] | None = None,
    ) -> bool:
        if i == j:
            return True
        if site_types is not None and site_types[i] != site_types[j]:
            return False
        frac_i = self._cart_to_frac(cart_pts[i])
        frac_j = self._cart_to_frac(cart_pts[j])
        tol = self.symmetry_tolerance
        for R, t in frac_ops:
            if self._site_distance_under_symop(frac_i, frac_j, R, t, planar) < tol:
                return True
        return False

    def _verify_site_orbits(
        self,
        cart_pts: list[np.ndarray],
        frac_ops: list[tuple[np.ndarray, np.ndarray]],
        planar: bool,
        orbits: list[list[int]],
        site_types: list[str] | None = None,
    ) -> None:
        """Every pair in an orbit must be related by at least one symmetry operation."""
        for idxs in orbits:
            for ii, i in enumerate(idxs):
                for j in idxs[ii + 1 :]:
                    if not self._site_pair_connected_by_ops(
                        i, j, cart_pts, frac_ops, planar, site_types=site_types
                    ):
                        raise SymmetryAnalysisError(
                            "site orbit failed verification: no symmetry operation "
                            f"maps site {i} to site {j} within tolerance"
                        )

    def find_equivalent_atoms(self) -> list[list[int]]:
        """Wyckoff-equivalent atom groups from spglib."""
        if self._equivalent_atoms is not None:
            return self._equivalent_atoms

        ds = self._ensure_dataset()
        eq = np.asarray(ds.equivalent_atoms, dtype=int)
        buckets: dict[int, list[int]] = {}
        for i, rep in enumerate(eq):
            buckets.setdefault(int(rep), []).append(i)
        self._equivalent_atoms = [sorted(v) for v in buckets.values()]
        self._equivalent_atoms.sort(key=lambda g: g[0])
        return self._equivalent_atoms

    def _site_3d_cart(self, site: Site) -> np.ndarray:
        return np.asarray(site.xyz, dtype=float).reshape(3).copy()

    def _site_sort_key(self, site: Site) -> tuple[float, float, float, str]:
        xy = site.xy
        return (float(xy[0]), float(xy[1]), float(site.z), str(site.site_type))

    def _build_orbit_output(
        self,
        sites: list[Site],
        orbits: list[list[int]],
    ) -> list[Site]:
        out: list[Site] = []
        for idxs in orbits:
            rep = min(idxs, key=lambda i: self._site_sort_key(sites[i]))
            equiv_xy = tuple(sites[k].xy.copy() for k in idxs)
            out.append(
                with_symmetry(
                    sites[rep],
                    symmetry_multiplicity=len(idxs),
                    symmetry_equivalent_sites=equiv_xy,
                )
            )
        return out

    def analyze_site_symmetry(
        self,
        sites: list[Site],
        planar: bool | None = None,
    ) -> list[Site]:
        """Group equivalent adsorption sites using spglib operations and union-find.

        Full symmetry operations are available from :meth:`detect_symmetry_operations`
        or :meth:`get_symmetry_info`; returned sites carry multiplicity and
        equivalent-site coordinates.
        """
        if not sites:
            return []

        if planar is None:
            zs = np.array([float(s.z) for s in sites], dtype=float)
            planar = bool(zs.size > 0 and float(np.ptp(zs)) < self.symmetry_tolerance)

        sorted_sites = sorted(sites, key=self._site_sort_key)
        frac_ops = self._frac_ops_from_dataset()
        n = len(sorted_sites)
        cart_pts = [self._site_3d_cart(s) for s in sorted_sites]
        site_types = [str(s.site_type) for s in sorted_sites]

        parent: list[int] = list(range(n))
        rank = [0] * n

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        frac_pts = [self._cart_to_frac(p) for p in cart_pts]
        for i in range(n):
            frac_i = frac_pts[i]
            for R, t in frac_ops:
                frac_p = self._apply_frac_symop(frac_i, R, t)
                if self._mode == "periodic":
                    frac_p = self._wrap_frac(frac_p)
                for j in range(n):
                    if i == j or site_types[i] != site_types[j]:
                        continue
                    d_frac = self._mic_frac_delta(frac_p, frac_pts[j])
                    sep = self._cart_sep_from_frac_delta(d_frac)
                    dist = self._separation_distance(sep, bool(planar))
                    if dist < self.symmetry_tolerance:
                        union(i, j)

        roots: dict[int, list[int]] = {}
        for i in range(n):
            r = find(i)
            roots.setdefault(r, []).append(i)

        orbits = [sorted(v) for _k, v in sorted(roots.items(), key=lambda x: x[0])]
        self._verify_site_orbits(
            cart_pts, frac_ops, planar, orbits, site_types=site_types
        )
        return self._build_orbit_output(sorted_sites, orbits)

    def detect_symmetry_breaking(self, reference_atoms: Atoms) -> bool:
        """True if space group or symmetry operation set differs from reference."""
        ref = SymmetryAnalyzer(
            reference_atoms,
            self.symmetry_tolerance,
            mode="auto",
            angle_tolerance=self._angle_tolerance,
        )
        ds_cur = self._ensure_dataset()
        ds_ref = ref._ensure_dataset()

        if int(ds_ref.number) != int(ds_cur.number):
            return True

        fp_ref = self._operations_fingerprint(ref)
        fp_cur = self._operations_fingerprint(self)
        return fp_ref != fp_cur

    def _operations_fingerprint(self, analyzer: "SymmetryAnalyzer") -> str:
        frac_ops = list(analyzer._frac_ops_from_dataset())

        def sort_key(
            item: tuple[np.ndarray, np.ndarray],
        ) -> tuple[bytes, bytes]:
            R, t = item
            return (
                np.round(R, decimals=5).tobytes(),
                np.round(t, decimals=8).tobytes(),
            )

        frac_ops.sort(key=sort_key)
        parts: list[bytes] = []
        for R, t in frac_ops:
            parts.append(np.round(R, decimals=5).tobytes())
            parts.append(np.round(t, decimals=8).tobytes())
        return hashlib.sha256(b"".join(parts)).hexdigest()

    def get_symmetry_info(self) -> dict[str, Any]:
        """Symmetry metadata including spglib space group."""
        operations = self.detect_symmetry_operations()
        equivalent_atoms = self.find_equivalent_atoms()
        ds = self._ensure_dataset()

        info: dict[str, Any] = {
            "n_symmetry_operations": len(operations),
            "symmetry_operations": [op.tolist() for op in operations],
            "n_equivalent_atom_groups": len(equivalent_atoms),
            "equivalent_atom_groups": [list(g) for g in equivalent_atoms],
            "is_periodic_xy": self._is_periodic_xy(),
            "symmetry_mode": self._mode,
            "has_rotational_symmetry": any(
                self._is_rotational_op(op) for op in operations
            ),
            "has_reflection_symmetry": any(
                self._is_reflection_op(op) for op in operations
            ),
            "has_translational_symmetry": any(
                self._is_translational_op(op) for op in operations
            ),
            "spacegroup_number": int(ds.number),
            "international_symbol": str(ds.international),
            "hall_symbol": str(ds.hall),
        }
        return info

    def _is_rotational_op(self, op: np.ndarray) -> bool:
        rotation_part = op[:3, :3]
        det = float(np.linalg.det(rotation_part))
        if abs(det - 1.0) > 1e-5:
            return False
        identity_rotation = np.eye(3)
        is_identity = np.allclose(rotation_part, identity_rotation) and np.allclose(
            op[:3, 3], 0
        )
        return not is_identity

    def _is_reflection_op(self, op: np.ndarray) -> bool:
        rotation_part = op[:3, :3]
        det = float(np.linalg.det(rotation_part))
        return abs(det + 1.0) < 1e-5

    def _is_translational_op(self, op: np.ndarray) -> bool:
        rotation_part = op[:3, :3]
        identity_rotation = np.eye(3)
        if not np.allclose(rotation_part, identity_rotation):
            return False
        return not np.allclose(op[:3, 3], 0)
