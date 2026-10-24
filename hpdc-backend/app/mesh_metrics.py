"""Pure-numpy geometry metrics on a triangle mesh (units: mm).

Kept independent from the CAD kernel so it can be unit-tested without cadquery.
"""
from __future__ import annotations
import numpy as np


def orient_outward(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Flip winding if signed volume is negative so normals point outward."""
    t = v[f]
    signed_vol = np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0
    return f[:, ::-1] if signed_vol < 0 else f


def tri_normals_areas(v: np.ndarray, f: np.ndarray):
    t = v[f]
    c = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    a = np.linalg.norm(c, axis=1)
    n = c / np.maximum(a[:, None], 1e-12)
    return n, a / 2.0


def mesh_volume_mm3(v, f) -> float:
    t = v[f]
    return abs(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)


def projected_area_mm2(v, f) -> tuple[float, int]:
    """Largest silhouette area over X/Y/Z. Returns (area, axis_index).

    For a closed mesh, projected area along d = 0.5 * sum(A_i * |n_i . d|).
    The largest value is the conservative figure used for clamping-force sizing.
    """
    n, a = tri_normals_areas(v, f)
    best, axis = 0.0, 0
    for i in range(3):
        p = 0.5 * float((a * np.abs(n[:, i])).sum())
        if p > best:
            best, axis = p, i
    return best, axis


def local_thickness(v, f, n_samples=1500, max_t=60.0, seed=0) -> np.ndarray:
    """Ray-cast inward from sampled surface points; hit distance = local wall thickness."""
    rng = np.random.default_rng(seed)
    n, a = tri_normals_areas(v, f)
    idx = rng.choice(len(f), size=n_samples, p=a / a.sum())
    r1, r2 = rng.random(n_samples), rng.random(n_samples)
    s = np.sqrt(r1)
    t = v[f[idx]]
    pts = (1 - s)[:, None] * t[:, 0] + (s * (1 - r2))[:, None] * t[:, 1] + (s * r2)[:, None] * t[:, 2]
    dirs = -n[idx]
    origins = pts + dirs * 1e-3

    tri = v[f]
    v0, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    out = np.full(n_samples, np.inf)
    chunk = 200
    for s0 in range(0, n_samples, chunk):
        o = origins[s0:s0 + chunk, None, :]
        d = dirs[s0:s0 + chunk, None, :]
        h = np.cross(d, e2[None])
        det = np.einsum("ijk,ijk->ij", e1[None], h)
        ok = np.abs(det) > 1e-10
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        sv = o - v0[None]
        u = np.einsum("ijk,ijk->ij", sv, h) * inv
        q = np.cross(sv, e1[None])
        w = np.einsum("ijk,ijk->ij", d, q) * inv
        dist = np.einsum("ijk,ijk->ij", e2[None], q) * inv
        hit = ok & (u >= 0) & (w >= 0) & (u + w <= 1) & (dist > 1e-4)
        out[s0:s0 + chunk] = np.where(hit, dist, np.inf).min(axis=1)
    out = out[np.isfinite(out) & (out <= max_t)]
    return out
