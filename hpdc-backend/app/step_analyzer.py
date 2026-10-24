"""STEP file -> geometric features used by the HPDC process model."""
from __future__ import annotations
import math, os, re, tempfile
from dataclasses import dataclass, asdict
import numpy as np
from .mesh_metrics import (orient_outward, tri_normals_areas, mesh_volume_mm3,
                           projected_area_mm2, local_thickness)

THIN_WALL_MM = 2.5   # walls below this are "thin" for HPDC aluminium


@dataclass
class GeometryFeatures:
    component_name: str
    component_type: str
    estimated_wall_thickness_mm: float
    component_volume_cm3: float
    projected_area_cm2: float
    geometry_complexity: str
    thin_wall_fraction: float
    # extras (useful for the ML model / UI, not required by the spec)
    complexity_score: float = 0.0
    surface_area_cm2: float = 0.0
    bbox_mm: tuple = (0.0, 0.0, 0.0)
    face_count: int = 0

    def to_dict(self):
        return asdict(self)


def _step_product_name(path: str, fallback: str) -> str:
    try:
        with open(path, "r", errors="ignore") as fh:
            head = fh.read(400_000)
        m = re.search(r"PRODUCT\s*\(\s*'([^']*)'\s*,\s*'([^']*)'", head, re.I)
        if m:
            name = (m.group(2) or m.group(1)).strip()
            if name and name.lower() not in {"none", "unknown", "part", "open cascade step translator"}:
                return name
    except OSError:
        pass
    return fallback


def _load_mesh(path: str):
    try:
        import cadquery as cq
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("cadquery is required to read STEP files (conda install -c conda-forge cadquery)") from e
    shape = cq.importers.importStep(path).val()
    bb = shape.BoundingBox()
    diag = math.sqrt(bb.xlen**2 + bb.ylen**2 + bb.zlen**2)
    verts, tris = shape.tessellate(max(diag * 5e-4, 0.01), 0.2)
    v = np.array([[p.x, p.y, p.z] for p in verts], dtype=float)
    f = np.array(tris, dtype=int)
    return v, f, len(shape.Faces()), (bb.xlen, bb.ylen, bb.zlen)


def classify_component(name: str, bbox, thickness: float, thin_frac: float, area_cm2: float, vol_cm3: float) -> str:
    n = name.lower()
    keywords = [("housing", "Housing"), ("case", "Housing"), ("gearbox", "Gearbox Housing"),
                ("cover", "Cover"), ("lid", "Cover"), ("cap", "Cover"), ("bracket", "Bracket"),
                ("mount", "Bracket"), ("flange", "Flange"), ("pump", "Pump Body"),
                ("valve", "Valve Body"), ("block", "Engine Block"), ("heat", "Heat Sink"),
                ("sink", "Heat Sink"), ("frame", "Frame")]
    for k, label in keywords:
        if k in n:
            return label
    d = sorted(bbox)
    flat = d[0] / max(d[2], 1e-6)
    hollow_ratio = (area_cm2 * 0.1 * max(thickness, 0.1)) / max(vol_cm3, 1e-6)  # ~2 for thin shell
    if flat < 0.15 and thin_frac > 0.3:
        return "Cover / Plate"
    if hollow_ratio > 1.4 and thin_frac > 0.25:
        return "Housing"
    if d[0] / max(d[1], 1e-6) < 0.3 and flat < 0.35:
        return "Bracket"
    return "General Structural Casting"


def complexity(face_count, vol_cm3, surf_cm2, thick, thin_frac):
    # sphericity: 1 for sphere, lower = more convoluted
    sph = (math.pi ** (1 / 3)) * (6 * vol_cm3) ** (2 / 3) / max(surf_cm2, 1e-6)
    face_term = min(math.log10(max(face_count, 1)) / 3.0, 1.0)
    shape_term = 1.0 - min(max(sph, 0.0), 1.0)
    thick_term = float(np.clip(thick.std() / max(np.median(thick), 1e-6) / 1.5, 0, 1)) if len(thick) > 5 else 0.5
    score = 0.35 * face_term + 0.30 * shape_term + 0.20 * thick_term + 0.15 * thin_frac
    label = "Low" if score < 0.35 else "Medium" if score < 0.55 else "High"
    return label, round(float(score), 3)


def analyze_mesh(v, f, face_count: int, bbox, name: str) -> GeometryFeatures:
    f = orient_outward(v, f)
    vol_mm3 = mesh_volume_mm3(v, f)
    _, a = tri_normals_areas(v, f)
    surf_mm2 = float(a.sum())
    proj_mm2, _ = projected_area_mm2(v, f)
    thick = local_thickness(v, f)
    if len(thick) < 20:                       # ray-cast failed -> shell approximation 2V/A
        est = 2 * vol_mm3 / max(surf_mm2, 1e-6)
        thick = np.array([est])
        thin_frac = 1.0 if est < THIN_WALL_MM else 0.0
    else:
        est = float(np.median(thick))
        thin_frac = float((thick < THIN_WALL_MM).mean())
    vol_cm3, surf_cm2 = vol_mm3 / 1e3, surf_mm2 / 1e2
    label, score = complexity(face_count, vol_cm3, surf_cm2, thick, thin_frac)
    ctype = classify_component(name, bbox, est, thin_frac, surf_cm2, vol_cm3)
    return GeometryFeatures(name, ctype, round(est, 2), round(vol_cm3, 2), round(proj_mm2 / 100, 2),
                            label, round(thin_frac, 3), score, round(surf_cm2, 2),
                            tuple(round(x, 1) for x in bbox), face_count)


def analyze_step_bytes(data: bytes, filename: str, return_mesh: bool = False):
    stem = os.path.splitext(os.path.basename(filename))[0]
    with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        name = _step_product_name(path, stem)
        v, f, nfaces, bbox = _load_mesh(path)
        feats = analyze_mesh(v, f, nfaces, bbox, name)
        return (feats, v, f) if return_mesh else feats
    finally:
        os.unlink(path)
