"""Geometry features -> HPDC process parameters.

Two layers:
  1. A physics/rule-based baseline (works with zero training data).
  2. Optional trained ML models (models/hpdc_models.joblib, see train_model.py).
     If present they override the baseline per-target.
"""
from __future__ import annotations
import os
from functools import lru_cache
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "hpdc_models.joblib")
FEATURES = ["estimated_wall_thickness_mm", "component_volume_cm3", "projected_area_cm2",
            "complexity_score", "thin_wall_fraction"]
TARGETS = ["melt_temp_C", "die_temp_C", "injection_speed_mps", "injection_pressure_bar",
           "intensification_pressure_bar", "filling_time_s", "cooling_time_s", "vacuum_pressure_bar"]
STRUCTURAL = ("Housing", "Gearbox Housing", "Frame", "Engine Block", "Bracket")


def pick_alloy(g: dict) -> str:
    t, thin, cx = g["estimated_wall_thickness_mm"], g["thin_wall_fraction"], g["geometry_complexity"]
    if g["component_type"] in STRUCTURAL and t <= 3.5:
        return "AlSi10MnMg (Silafont-36)"          # structural, crash/ductile, thin walls
    if t < 2.2 or thin > 0.45 or cx == "High":
        return "AlSi10Mg (A360)"                    # better fluidity / pressure tightness
    return "AlSi9Cu3 (ADC12 / A383)"                # general purpose


def baseline(g: dict) -> dict:
    t = max(g["estimated_wall_thickness_mm"], 0.8)
    thin, sc = g["thin_wall_fraction"], g["complexity_score"]
    area, vol = g["projected_area_cm2"], g["component_volume_cm3"]
    thinness = float(np.clip((3.5 - t) / 2.5, 0, 1))      # 0 thick .. 1 very thin

    melt = 665 + 25 * thinness + 12 * sc + 8 * thin
    die = 190 + 35 * thinness + 15 * sc - 10 * float(np.clip((t - 4) / 4, 0, 1))
    fast = 2.0 + 2.5 * thinness + 1.0 * sc + 0.5 * thin           # plunger fast-shot, m/s
    gate_v = 30 + 30 * thinness + 8 * sc                          # info only (shot_profile)

    inj_p = 450 + 220 * thinness + 120 * sc + 60 * thin           # metal pressure, bar
    intens = inj_p * (1.35 + 0.25 * thinness) + 80 * sc
    fill = float(np.clip(0.018 * t * (1 + 0.45 * sc) * (1 + 0.1 * (melt - 665) / 20)
                         * (1 + 0.002 * (200 - die)), 0.025, 0.20))
    cool = float(np.clip(2.5 + 1.15 * t * t + 0.04 * np.sqrt(max(vol, 1)) * t, 3, 60)
                 * (1 + 0.003 * (die - 200)))
    vac = 0.05 if (thin > 0.35 or sc > 0.55) else 0.08 if sc > 0.35 else 0.12

    return {
        "melt_temp_C": round(melt, 0), "die_temp_C": round(die, 0),
        "injection_speed_mps": round(fast, 2), "injection_pressure_bar": round(inj_p, 0),
        "intensification_pressure_bar": round(intens, 0), "filling_time_s": round(fill, 3),
        "cooling_time_s": round(cool, 1), "vacuum_pressure_bar": round(vac, 3),
        "_extra": {
            "clamping_force_tonnes_est": round(area * intens / 100 * 1.2 / 9.81, 0),
            "gate_velocity_mps_est": round(gate_v, 0),
        },
    }


@lru_cache(maxsize=1)
def _load_models():
    if not os.path.exists(MODEL_PATH):
        return None
    import joblib
    return joblib.load(MODEL_PATH)


def predict(g: dict) -> dict:
    out = baseline(g)
    extra = out.pop("_extra")
    out["alloy_type"] = pick_alloy(g)
    source = "rule_based_baseline"
    bundle = _load_models()
    if bundle:
        x = np.array([[g[k] for k in FEATURES]])
        for tgt, mdl in bundle["regressors"].items():
            out[tgt] = float(round(mdl.predict(x)[0], 3))
        if "alloy_clf" in bundle:
            out["alloy_type"] = str(bundle["alloy_clf"].predict(x)[0])
        source = "trained_ml_model"
    shot = {
        "slow_shot_speed_mps": round(0.25 + 0.05 * g["complexity_score"] * 4, 2),
        "fast_shot_speed_mps": out["injection_speed_mps"],
        "changeover": "Switch to fast shot after runner/biscuit is filled (~35-50% stroke); tune on first shots",
        **extra,
    }
    return {"parameters": out, "shot_profile": shot, "model_source": source}
