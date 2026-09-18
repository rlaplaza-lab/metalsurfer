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
from scipy.spatial import KDTree
from threadpoolctl import threadpool_limits

from ._geom_pbc import (
    cart_to_frac,
    frac_to_cart,
    minimum_image_fractional_delta,
    slab_normal,
    wrap_fractional,
)
from ._numeric_defaults import DEFAULT_SYMMETRY_TOLERANCE
from ._utils import cell_has_volume, union_find_cluster

if TYPE_CHECKING:
    from .placement.site_types import Site

# Opt into the new spglib error handling (raises SpglibError instead of
# returning None) and suppress the DeprecationWarning it would emit otherwise.
_spglib_error_module.OLD_ERROR_HANDLING = False

# Cap peak RAM for the chunked matcher fallback: one float64 (chunk × n × 3) buffer.
_SYMMETRY_PAIR_CHUNK_BYTES = 16 * 1024 * 1024
# Prefer KDTree orbit matching once catalogs exceed this size.
_SYMMETRY_KDTREE_MIN_SITES = 32

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
        """Instantiate the symmetry analyzer.

        Parameters
        ----------
        atoms
            ASE Atoms object.
        symmetry_tolerance
            Tolerance for symmetry detection.
        mode
            Symmetry mode ("auto", "periodic", or "cluster").
        angle_tolerance
            Optional angle tolerance.
        """
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

        self._dataset: Any | None = None
        self._operations_frac: list[tuple[np.ndarray, np.ndarray]] | None = None

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

    def _symmetry_pbc(self) -> np.ndarray:
        """Per-axis periodicity used for **site–site** minimum-image folding.

        Periodic mode hands spglib a genuine 3D lattice, so all three axes fold.
        Cluster mode builds a *padded box* around a finite object: that box has
        no physical periodicity, and MIC-folding site–site deltas across it
        would merge antipodal sites of any cluster wider than roughly twice the
        padding margin. Transformed fractional coordinates after a symop are
        wrapped separately via :meth:`_wrap_frac` so origin-centred point-group
        ops still map sites onto each other.
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
        """Wrap fractional coordinates after a symmetry operation.

        Periodic mode wraps on the true lattice. Cluster mode wraps into the
        padded orthorhombic box so spglib rotations about the box origin land
        back on partner sites; site–site deltas still use
        :meth:`_symmetry_pbc` (no MIC) so unrelated antipodes are not merged.
        """
        if self._mode == "periodic":
            return wrap_fractional(frac, self._symmetry_pbc())
        return wrap_fractional(frac, np.array([True, True, True], dtype=bool))

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
        """Return unit normal from lattice a × b (slab plane)."""
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

    def _orbit_connectivity(
        self,
        frac_pts: np.ndarray,
        frac_ops: list[tuple[np.ndarray, np.ndarray]],
        source: int,
        targets: list[int],
        planar: bool,
    ) -> np.ndarray:
        """Return whether each target is an image of *source* under some symmetry op.

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

    def _verify_site_orbits(
        self,
        frac_pts: np.ndarray,
        frac_ops: list[tuple[np.ndarray, np.ndarray]],
        planar: bool,
        orbits: list[list[int]],
    ) -> None:
        """Every member of an orbit must be related to its representative by a symmetry operation.

        Verifying only representative→member (rather than every pair) catches the
        same failure mode at O(k) instead of O(k²). Union-find merges through
        transitive chains, so this is a genuine independent check.
        """
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

        Returned sites carry multiplicity and equivalent-site coordinates.
        Grouping is blocked by classified ``site_type`` only; ``site_source``
        (topology / Voronoi / atop injection) does not affect orbits.

        Parameters
        ----------
        sites
            List of adsorption sites to analyze (typically the clustered catalog).
        planar
            When true, ignore differences along the slab normal.
        """
        if not sites:
            return []

        if planar is None:
            zs = np.array([float(s.z) for s in sites], dtype=float)
            planar = bool(zs.size > 0 and float(np.ptp(zs)) < self.symmetry_tolerance)

        # Cap BLAS/OpenMP workers: after PyTorch/CUDA init, threaded norms on
        # orbit distance buffers can deadlock or thrash.
        with threadpool_limits(limits=1):
            sorted_sites = sorted(sites, key=self._site_sort_key)
            frac_ops = self._frac_ops_from_dataset()
            n = len(sorted_sites)
            cart_pts = [self._site_3d_cart(s) for s in sorted_sites]
            site_types = [str(s.site_type) for s in sorted_sites]

            frac_pts = self._cart_to_frac(np.asarray(cart_pts, dtype=float))
            type_index = {name: k for k, name in enumerate(dict.fromkeys(site_types))}
            type_codes = np.array([type_index[s] for s in site_types], dtype=int)
            merge_pairs: list[tuple[int, int]] = []
            # Block-diagonal by site_type; chunked MIC pairs avoid a full n×n×3.
            for type_code in range(len(type_index)):
                idx = np.nonzero(type_codes == type_code)[0]
                if len(idx) < 2:
                    continue
                sub_frac = frac_pts[idx]
                for R, t in frac_ops:
                    for li, lj in self._symop_match_pairs(sub_frac, R, t, bool(planar)):
                        merge_pairs.append((int(idx[int(li)]), int(idx[int(lj)])))

            components = union_find_cluster(n, merge_pairs)
            orbits = [sorted(comp) for comp in sorted(components, key=min)]
            self._verify_site_orbits(frac_pts, frac_ops, planar, orbits)
            return self._build_orbit_output(sorted_sites, orbits)

    def _symop_match_pairs(
        self,
        frac_pts: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        planar: bool,
    ) -> list[tuple[int, int]]:
        """Local ``(i, j)`` pairs where ``op(site_i)`` lands on ``site_j`` within tol.

        Dense catalogs use KDTree nearest-neighbour queries (O(n log n) per
        symop). Small catalogs keep a row-chunked dense matcher so peak RAM
        stays near :data:`_SYMMETRY_PAIR_CHUNK_BYTES`.
        """
        n = int(len(frac_pts))
        if n < 2:
            return []
        if n >= _SYMMETRY_KDTREE_MIN_SITES:
            return self._symop_match_pairs_kdtree(frac_pts, R, t, planar)
        return self._symop_match_pairs_dense(frac_pts, R, t, planar)

    def _symop_match_pairs_kdtree(
        self,
        frac_pts: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        planar: bool,
    ) -> list[tuple[int, int]]:
        """KDTree matching of transformed sites onto the catalog."""
        n = int(len(frac_pts))
        tol = float(self.symmetry_tolerance)
        transformed = self._wrap_frac(frac_pts @ R.T + t)
        pbc = self._symmetry_pbc()
        n_hat = self._slab_normal() if planar else None

        cart_ref = frac_to_cart(frac_pts, self._lattice)
        cart_query = frac_to_cart(transformed, self._lattice)
        if n_hat is not None:
            cart_ref = cart_ref - (cart_ref @ n_hat)[..., None] * n_hat
            cart_query = cart_query - (cart_query @ n_hat)[..., None] * n_hat

        if np.any(pbc) and cell_has_volume(self._lattice):
            # Expand reference by ±1 lattice images on periodic axes so
            # query_ball covers MIC neighbours within *tol* without an n×n matrix.
            cell = np.asarray(self._lattice, dtype=float)
            offsets = [np.zeros(3, dtype=float)]
            for dim in range(3):
                if bool(pbc[dim]):
                    offsets.extend([cell[dim], -cell[dim]])
            # Two-axis combinations (corners) when two+ axes are periodic.
            axes = [d for d in range(3) if bool(pbc[d])]
            if len(axes) >= 2:
                for i, a in enumerate(axes):
                    for b in axes[i + 1 :]:
                        for sa in (-1, 1):
                            for sb in (-1, 1):
                                offsets.append(sa * cell[a] + sb * cell[b])
            expanded = np.vstack([cart_ref + off for off in offsets])
            tree = KDTree(expanded)
            hits = tree.query_ball_point(cart_query, r=tol)
            pairs: list[tuple[int, int]] = []
            for i, js in enumerate(hits):
                seen: set[int] = set()
                for h in js:
                    j = int(h) % n
                    if j == i or j in seen:
                        continue
                    seen.add(j)
                    d_frac = minimum_image_fractional_delta(
                        transformed[i] - frac_pts[j], pbc, copy=True
                    )
                    sep = frac_to_cart(d_frac, self._lattice)
                    if n_hat is not None:
                        sep = sep - float(np.dot(sep, n_hat)) * n_hat
                    if float(np.linalg.norm(sep)) < tol:
                        pairs.append((i, j))
            return pairs

        tree = KDTree(cart_ref)
        hits = tree.query_ball_point(cart_query, r=tol)
        pairs = []
        for i, js in enumerate(hits):
            for j in js:
                if int(j) != i:
                    pairs.append((i, int(j)))
        return pairs

    def _symop_match_pairs_dense(
        self,
        frac_pts: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        planar: bool,
    ) -> list[tuple[int, int]]:
        """Row-chunked dense MIC matcher for small catalogs."""
        n = int(len(frac_pts))
        if n < 2:
            return []
        tol = self.symmetry_tolerance
        per_col = n * 3 * 8
        chunk = max(1, min(n, _SYMMETRY_PAIR_CHUNK_BYTES // max(per_col, 1)))
        transformed = self._wrap_frac(frac_pts @ R.T + t)
        pairs: list[tuple[int, int]] = []
        pbc = self._symmetry_pbc()
        n_hat = self._slab_normal() if planar else None
        for i0 in range(0, n, chunk):
            i1 = min(n, i0 + chunk)
            delta = transformed[i0:i1, None, :] - frac_pts[None, :, :]
            delta = minimum_image_fractional_delta(delta, pbc, copy=False)
            sep = frac_to_cart(delta, self._lattice)
            if n_hat is not None:
                sep = sep - (sep @ n_hat)[..., None] * n_hat
            dist = np.sqrt(np.sum(sep * sep, axis=-1))
            for local_row, i in enumerate(range(i0, i1)):
                dist[local_row, i] = np.inf
            li, lj = np.nonzero(dist < tol)
            for a, b in zip(li.tolist(), lj.tolist(), strict=True):
                pairs.append((i0 + int(a), int(b)))
        return pairs

    def detect_symmetry_breaking(
        self,
        reference_atoms: Atoms,
        *,
        reference_analyzer: SymmetryAnalyzer | None = None,
    ) -> bool:
        """Check whether space group or symmetry operation set differs from reference.

        Parameters
        ----------
        reference_atoms
            ASE Atoms to compare against.
        reference_analyzer
            Optional pre-built analyzer for *reference_atoms* (avoids re-running
            spglib on an unchanging clean reference).
        """
        ref = reference_analyzer or SymmetryAnalyzer(
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
            """Sort key for fractional symmetry operations.

            Parameters
            ----------
            item
                Tuple of (rotation, translation) arrays.
            """
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
