"""Equivalent adsorption sites via `spglib` (periodic cell or padded cluster-in-box).

Slab symmetry follows the 3D ASE supercell (not layer groups). `symmetry_tolerance`
is `symprec` for spglib and the Cartesian threshold for site matching.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import spglib
import spglib.error as _spglib_error_module
from ase import Atoms

from ._geom_pbc import (
    cart_to_frac,
    frac_to_cart,
    minimum_image_fractional_delta,
    slab_normal,
    wrap_fractional,
)
from ._numeric_defaults import DEFAULT_SYMMETRY_TOLERANCE
from ._utils import cell_has_volume

if TYPE_CHECKING:
    from .placement.site_types import Site

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
        self._slab_normal_cache: np.ndarray | None = None

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
            if not cell_has_volume(self.cell):
                raise ValueError(
                    "Periodic symmetry requires a valid 3x3 cell with non-zero volume"
                )
            self._lattice = self.cell.copy()
            inv = np.linalg.inv(self._lattice)
            self._fractional = self.positions @ inv
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
        """Cartesian 4x4 for the row-vector cell convention ``r_cart = r_frac @ L``.

        The fractional op acts on rows as ``r_frac' = r_frac @ R.T + t`` (see
        :meth:`_apply_frac_symop`), so in Cartesian space
        ``r_cart' = r_cart @ (L^-1 R.T L) + t @ L``. The returned matrix acts on
        *column* vectors, hence ``R_cart = L.T @ R @ inv(L.T)`` and
        ``t_cart = L.T @ t``. Using ``L`` instead of ``L.T`` would yield
        non-orthogonal "rotations" for non-orthogonal cells.
        """
        Lt = self._lattice.T
        R_cart = Lt @ R_frac @ np.linalg.inv(Lt)
        t_cart = Lt @ t_frac.reshape(3)
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

    def _symmetry_pbc(self) -> np.ndarray:
        """Per-axis periodicity used for wrapping and minimum-image folding.

        Periodic mode hands spglib a genuine 3D lattice, so all three axes fold.
        Cluster mode builds a *padded box* around a finite object: that box has
        no periodicity at all, and folding across it would merge antipodal sites
        of any cluster wider than roughly twice the padding margin.
        """
        if self._mode == "periodic":
            return np.array([True, True, True], dtype=bool)
        return np.array([False, False, False], dtype=bool)

    def _cart_to_frac(self, cart: np.ndarray) -> np.ndarray:
        """Cartesian → fractional in the *same* frame spglib was given.

        Cluster mode re-applies the centre-of-mass shift and half-box offset
        used by :meth:`_prepare_lattice_and_fractional`; without it, site
        fractional coordinates live in a different origin than the atomic ones
        and the orbit assignment depends on where the cluster happens to sit in
        absolute Cartesian space.
        """
        arr = np.asarray(cart, dtype=float)
        if (
            self._mode == "cluster"
            and self._cluster_com is not None
            and self._cluster_half is not None
        ):
            arr = arr - self._cluster_com + self._cluster_half
        return cart_to_frac(arr, self._lattice)

    def _wrap_frac(self, frac: np.ndarray) -> np.ndarray:
        pbc = self._symmetry_pbc()
        if not np.any(pbc):
            return frac
        return wrap_fractional(frac, pbc)

    def _apply_frac_symop(
        self, frac_row: np.ndarray, R: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        """Apply r' = r @ R.T + t for row vectors (matches spglib Python examples)."""
        return frac_row @ R.T + t

    def _mic_frac_delta(self, fa: np.ndarray, fb: np.ndarray) -> np.ndarray:
        """Shortest fractional difference, folded only on genuinely periodic axes.

        Shape-generic: the last axis must be the three fractional components.
        """
        d = np.asarray(fa, dtype=float) - np.asarray(fb, dtype=float)
        return minimum_image_fractional_delta(d, self._symmetry_pbc(), copy=False)

    def _cart_sep_from_frac_delta(self, d_frac: np.ndarray) -> np.ndarray:
        """Cartesian separation vector (row) from fractional MIC difference."""
        return frac_to_cart(d_frac, self._lattice)

    def _slab_normal(self) -> np.ndarray:
        """Unit normal from lattice a × b (slab plane)."""
        if self._slab_normal_cache is not None:
            return self._slab_normal_cache
        n_hat = slab_normal(self._lattice)
        self._slab_normal_cache = n_hat
        return n_hat

    def _separation_norms(self, sep: np.ndarray, planar: bool) -> np.ndarray:
        """Cartesian separation norms; when *planar*, drop the slab-normal component."""
        arr = np.asarray(sep, dtype=float)
        if planar:
            n = self._slab_normal()
            arr = arr - (arr @ n)[..., None] * n
        return np.linalg.norm(arr, axis=-1)

    def _separation_distance(self, sep: np.ndarray, planar: bool) -> float:
        """Cartesian MIC distance; when *planar*, drop the slab-normal component."""
        return float(
            self._separation_norms(np.asarray(sep, dtype=float).reshape(1, 3), planar)[
                0
            ]
        )

    def _orbit_connectivity(
        self,
        frac_pts: np.ndarray,
        frac_ops: list[tuple[np.ndarray, np.ndarray]],
        source: int,
        targets: list[int],
        planar: bool,
    ) -> np.ndarray:
        """Boolean per target: is ``targets[k]`` the image of *source* under some op?

        Batched over targets, looping over operations, with an early exit once
        every target is accounted for.
        """
        if not targets:
            return np.zeros(0, dtype=bool)
        frac_source = np.asarray(frac_pts[source], dtype=float)
        frac_targets = np.asarray(frac_pts, dtype=float)[np.asarray(targets, dtype=int)]
        connected = np.zeros(len(targets), dtype=bool)
        tol = self.symmetry_tolerance
        for R, t in frac_ops:
            moved = self._wrap_frac(self._apply_frac_symop(frac_source, R, t))
            d_frac = self._mic_frac_delta(moved, frac_targets)
            sep = self._cart_sep_from_frac_delta(d_frac)
            connected |= self._separation_norms(sep, planar) < tol
            if bool(connected.all()):
                break
        return connected

    def _site_pair_connected_by_ops(
        self,
        i: int,
        j: int,
        cart_pts: list[np.ndarray],
        frac_ops: list[tuple[np.ndarray, np.ndarray]],
        planar: bool,
        site_types: list[str] | None = None,
        frac_pts: np.ndarray | None = None,
    ) -> bool:
        if i == j:
            return True
        if site_types is not None and site_types[i] != site_types[j]:
            return False
        if frac_pts is None:
            frac_pts = self._cart_to_frac(np.asarray(cart_pts, dtype=float))
        return bool(self._orbit_connectivity(frac_pts, frac_ops, i, [j], planar)[0])

    def _verify_site_orbits(
        self,
        cart_pts: list[np.ndarray],
        frac_ops: list[tuple[np.ndarray, np.ndarray]],
        planar: bool,
        orbits: list[list[int]],
        site_types: list[str] | None = None,
        frac_pts: np.ndarray | None = None,
    ) -> None:
        """Every member of an orbit must be related to its representative by a symmetry operation.

        Verifying only representative→member (rather than every pair) catches the
        same failure mode at O(k) instead of O(k²). Union-find merges through
        transitive chains, so this is a genuine independent check.
        """
        if frac_pts is None:
            frac_pts = self._cart_to_frac(np.asarray(cart_pts, dtype=float))
        for idxs in orbits:
            if len(idxs) < 2:
                continue
            rep = min(idxs)
            members = [j for j in idxs if j != rep]
            connected = self._orbit_connectivity(
                frac_pts, frac_ops, rep, members, planar
            )
            if not bool(connected.all()):
                bad = members[int(np.argmin(connected))]
                raise SymmetryAnalysisError(
                    "site orbit failed verification: no symmetry operation "
                    f"maps site {rep} to site {bad} within tolerance"
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
        # Imported lazily: `placement` imports `symmetry` at module scope, so a
        # top-level import here would create a circular import.
        from .placement.site_types import with_symmetry

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

        frac_pts = self._cart_to_frac(np.asarray(cart_pts, dtype=float))
        type_index = {name: k for k, name in enumerate(dict.fromkeys(site_types))}
        type_codes = np.array([type_index[s] for s in site_types], dtype=int)
        same_type = type_codes[:, None] == type_codes[None, :]
        np.fill_diagonal(same_type, False)
        tol = self.symmetry_tolerance
        for R, t in frac_ops:
            dist = self._pairwise_symop_distances(frac_pts, R, t, bool(planar))
            for i, j in np.argwhere((dist < tol) & same_type):
                union(int(i), int(j))

        roots: dict[int, list[int]] = {}
        for i in range(n):
            r = find(i)
            roots.setdefault(r, []).append(i)

        orbits = [sorted(v) for _k, v in sorted(roots.items(), key=lambda x: x[0])]
        self._verify_site_orbits(
            cart_pts,
            frac_ops,
            planar,
            orbits,
            site_types=site_types,
            frac_pts=frac_pts,
        )
        return self._build_orbit_output(sorted_sites, orbits)

    def _pairwise_symop_distances(
        self,
        frac_pts: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        planar: bool,
    ) -> np.ndarray:
        """``dist[i, j]`` = distance from ``op(site_i)`` to ``site_j``.

        Vectorised equivalent of looping ``_site_distance_under_symop`` over all
        ``(i, j)``. Only one ``n×n×3`` buffer is materialised per operation; the
        full ``m×n×n×3`` stack is never built (~484 MB at a 6×6 slab).
        """
        transformed = self._wrap_frac(frac_pts @ R.T + t)
        delta = transformed[:, None, :] - frac_pts[None, :, :]
        # ``delta`` is a freshly allocated n×n×3 temporary, so fold it in place:
        # the copying form would transiently double this buffer (~40 MB at a 6×6 slab).
        delta = minimum_image_fractional_delta(delta, self._symmetry_pbc(), copy=False)
        sep = frac_to_cart(delta, self._lattice)
        if planar:
            n_hat = self._slab_normal()
            sep = sep - (sep @ n_hat)[..., None] * n_hat
        return np.linalg.norm(sep, axis=-1)

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

    def _operations_fingerprint(self, analyzer: SymmetryAnalyzer) -> str:
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
