
"""
Smart Die-Casting Control (HPDC)
Run: streamlit run streamlit_app.py
"""

import hashlib

import numpy as np
import pandas as pd
import streamlit as st

from app.step_analyzer import analyze_step_bytes
from app.process_model import predict


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Smart Die-Casting Control",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 Smart Die-Casting Control")
st.caption(
    "CAD Geometry Analysis | HPDC Parameter Prediction | Engineering Report"
)

UNITS = {
    "melt_temp_C": "°C",
    "die_temp_C": "°C",
    "injection_speed_mps": "m/s",
    "injection_pressure_bar": "bar",
    "intensification_pressure_bar": "bar",
    "filling_time_s": "s",
    "cooling_time_s": "s",
    "vacuum_pressure_bar": "bar",
}

GEOMETRY_FIELDS = [
    "component_name",
    "component_type",
    "estimated_wall_thickness_mm",
    "component_volume_cm3",
    "projected_area_cm2",
    "geometry_complexity",
    "thin_wall_fraction",
]


# ==================================================
# 2. HELPER FUNCTIONS
# ==================================================
@st.cache_data(show_spinner="Analyzing CAD geometry...")
def run_analysis(data: bytes, filename: str):
    features, vertices, faces = analyze_step_bytes(
        data,
        filename,
        return_mesh=True,
    )
    return features.to_dict(), vertices, faces


def mesh_figure(vertices, faces, max_tris=40000):
    import plotly.graph_objects as go

    if len(faces) > max_tris:
        indices = np.random.default_rng(0).choice(
            len(faces), max_tris, replace=False
        )
        faces = faces[indices]

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="#8fa3b8",
                flatshading=True,
                opacity=1.0,
            )
        ]
    )

    fig.update_layout(
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=10, b=0),
        height=450,
    )
    return fig


def get_dimensions(bbox):
    """Read X, Y and Z dimensions from the analyzer's bounding box."""

    if isinstance(bbox, dict):
        aliases = {
            "X": ["x", "X", "length", "Length", "width_x"],
            "Y": ["y", "Y", "width", "Width", "width_y"],
            "Z": ["z", "Z", "height", "Height", "width_z"],
        }

        dimensions = {}

        for axis, keys in aliases.items():
            for key in keys:
                if key in bbox:
                    dimensions[axis] = bbox[key]
                    break

        return dimensions if len(dimensions) == 3 else None

    try:
        values = np.asarray(bbox, dtype=float).flatten()

        if len(values) >= 3:
            return {
                "X": values[0],
                "Y": values[1],
                "Z": values[2],
            }
    except (TypeError, ValueError):
        pass

    return None


def format_value(value, decimals=3):
    if value is None:
        return "Not available"

    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(value):
            return "Not available"
        return f"{value:.{decimals}f}"

    return str(value)


def geometry_table(geo):
    bbox_dimensions = get_dimensions(geo.get("bbox_mm"))

    dimension_values = bbox_dimensions or {}

    rows = [
        ("X Dimension", dimension_values.get("X"), "mm"),
        ("Y Dimension", dimension_values.get("Y"), "mm"),
        ("Z Dimension", dimension_values.get("Z"), "mm"),
        (
            "Estimated Wall Thickness",
            geo.get("estimated_wall_thickness_mm"),
            "mm",
        ),
        ("Component Volume", geo.get("component_volume_cm3"), "cm³"),
        ("Projected Area", geo.get("projected_area_cm2"), "cm²"),
        ("Surface Area", geo.get("surface_area_cm2"), "cm²"),
        ("Thin-Wall Fraction", geo.get("thin_wall_fraction"), ""),
        ("Geometry Complexity", geo.get("geometry_complexity"), ""),
        ("Complexity Score", geo.get("complexity_score"), ""),
        ("Face Count", geo.get("face_count"), ""),
    ]

    return pd.DataFrame(
        [
            {
                "Measurement": label,
                "Value": format_value(value),
                "Unit": unit,
            }
            for label, value, unit in rows
        ]
    )


# ==================================================
# 3. SIDEBAR
# ==================================================
with st.sidebar:
    st.header("📂 CAD Input")

    uploaded_file = st.file_uploader(
        "Upload STEP / STP file",
        type=["step", "stp"],
    )

    st.divider()

    st.caption(
        "Predictions are engineering estimates from your configured "
        "model or rule-based baseline. Validate settings before "
        "production use."
    )


if uploaded_file is None:
    st.info("Upload a STEP/STP CAD file from the sidebar to begin.")
    st.stop()


# ==================================================
# 4. ANALYZE CAD FILE
# ==================================================
file_bytes = uploaded_file.getvalue()
file_key = hashlib.sha256(file_bytes).hexdigest()

if st.session_state.get("file_key") != file_key:
    st.session_state["file_key"] = file_key
    st.session_state.pop("generated_result", None)
    st.session_state.pop("generated_geo", None)

try:
    geo, vertices, faces = run_analysis(file_bytes, uploaded_file.name)

except Exception as error:
    st.error(f"Could not process the STEP file: {error}")
    st.stop()

st.success(f"CAD file loaded successfully: {uploaded_file.name}")


# ==================================================
# 5. COMPONENT GEOMETRY
# ==================================================
st.header("1. Component Geometry")

st.write(
    f"**Component:** {geo.get('component_name', 'Unknown')}  "
    f"| **Type:** {geo.get('component_type', 'Unknown')}"
)

geometry_col, preview_col = st.columns([1, 1.15], gap="large")

with geometry_col:
    st.subheader("Extracted Dimensions")

    st.dataframe(
        geometry_table(geo),
        hide_index=True,
        use_container_width=True,
        height=420,
        column_config={
            "Measurement": st.column_config.TextColumn(
                "Measurement", width="large"
            ),
            "Value": st.column_config.TextColumn(
                "Value", width="medium"
            ),
            "Unit": st.column_config.TextColumn(
                "Unit", width="small"
            ),
        },
    )

with preview_col:
    st.subheader("3D CAD Preview")

    try:
        st.plotly_chart(
            mesh_figure(vertices, faces),
            use_container_width=True,
        )
    except Exception as error:
        st.warning(f"3D preview unavailable: {error}")

with st.expander("View complete extracted geometry data"):
    st.json(geo)


# ==================================================
# 6. REVIEW AND ADJUST GEOMETRY
# ==================================================
st.divider()
st.header("2. Review and Adjust Geometry")

st.caption(
    "Adjust the extracted values if required. "
    "These values will be used when generating the prediction."
)

complexity_options = ["Low", "Medium", "High"]

current_complexity = geo.get("geometry_complexity", "Medium")

if current_complexity not in complexity_options:
    current_complexity = "Medium"

try:
    default_wall = float(
        geo.get("estimated_wall_thickness_mm") or 2.0
    )
except (TypeError, ValueError):
    default_wall = 2.0

default_wall = min(30.0, max(0.5, default_wall))

try:
    default_thin_fraction = float(
        geo.get("thin_wall_fraction")
        if geo.get("thin_wall_fraction") is not None
        else 0.0
    )
except (TypeError, ValueError):
    default_thin_fraction = 0.0

default_thin_fraction = min(
    1.0, max(0.0, default_thin_fraction)
)

with st.form("hpdc_generation_form"):
    edit1, edit2, edit3 = st.columns(3)

    with edit1:
        wall_thickness = st.number_input(
            "Wall Thickness (mm)",
            min_value=0.5,
            max_value=30.0,
            value=default_wall,
            step=0.1,
        )

    with edit2:
        thin_wall_fraction = st.slider(
            "Thin-Wall Fraction",
            min_value=0.0,
            max_value=1.0,
            value=default_thin_fraction,
            step=0.01,
        )

    with edit3:
        complexity = st.selectbox(
            "Geometry Complexity",
            options=complexity_options,
            index=complexity_options.index(current_complexity),
        )

    use_complexity_score = st.checkbox(
        "Update complexity score from selected label",
        value=False,
    )

    st.divider()

    generate = st.form_submit_button(
        "⚙️ Generate Die-Casting Parameters",
        type="primary",
        use_container_width=True,
    )


# ==================================================
# 7. GENERATE PARAMETERS ON BUTTON CLICK
# ==================================================
if generate:
    prediction_geo = geo.copy()

    prediction_geo["estimated_wall_thickness_mm"] = wall_thickness
    prediction_geo["thin_wall_fraction"] = thin_wall_fraction
    prediction_geo["geometry_complexity"] = complexity

    if use_complexity_score:
        prediction_geo["complexity_score"] = {
            "Low": 0.2,
            "Medium": 0.45,
            "High": 0.7,
        }[complexity]

    try:
        with st.spinner("Generating HPDC parameters..."):
            prediction_result = predict(prediction_geo)

        st.session_state["generated_result"] = prediction_result
        st.session_state["generated_geo"] = prediction_geo

        st.success("HPDC parameters generated successfully!")

    except Exception as error:
        st.session_state.pop("generated_result", None)
        st.session_state.pop("generated_geo", None)
        st.error(f"Prediction failed: {error}")


# ==================================================
# 8. DISPLAY RESULTS
# ==================================================
result = st.session_state.get("generated_result")
result_geo = st.session_state.get("generated_geo")

if result is None:
    st.info(
        "Review the geometry and click **Generate Die-Casting "
        "Parameters** to view the predicted results."
    )
    st.stop()

params = result["parameters"]
shot_profile = result.get("shot_profile", {})

st.divider()
st.header("3. Predicted HPDC Process Parameters")

st.caption(
    f"Prediction source: {result.get('model_source', 'Unknown')}"
)

parameter_rows = []

for key, value in params.items():
    parameter_rows.append(
        {
            "Process Parameter": key.replace("_", " ").title(),
            "Predicted Value": format_value(value),
            "Unit": UNITS.get(key, "—"),
        }
    )

st.dataframe(
    pd.DataFrame(parameter_rows),
    hide_index=True,
    use_container_width=True,
    height=min(420, 40 + 38 * len(parameter_rows)),
    column_config={
        "Process Parameter": st.column_config.TextColumn(
            "Process Parameter", width="large"
        ),
        "Predicted Value": st.column_config.TextColumn(
            "Predicted Value", width="medium"
        ),
        "Unit": st.column_config.TextColumn(
            "Unit", width="small"
        ),
    },
)


# ==================================================
# 9. SHOT PROFILE
# ==================================================
st.header("4. Shot Profile and Machine Estimates")

shot_definitions = [
    ("Slow Shot Speed", "slow_shot_speed_mps", "m/s"),
    ("Fast Shot Speed", "fast_shot_speed_mps", "m/s"),
    ("Estimated Gate Velocity", "gate_velocity_mps_est", "m/s"),
    ("Estimated Clamping Force", "clamping_force_tonnes_est", "tonnes"),
]

shot_rows = []

for label, key, unit in shot_definitions:
    shot_rows.append(
        {
            "Parameter": label,
            "Estimated Value": format_value(shot_profile.get(key)),
            "Unit": unit,
        }
    )

st.dataframe(
    pd.DataFrame(shot_rows),
    hide_index=True,
    use_container_width=True,
)

if shot_profile.get("changeover"):
    st.info(f"Shot changeover: {shot_profile['changeover']}")


# ==================================================
# 10. CSV EXPORT
# ==================================================
st.divider()
st.header("5. Export Engineering Report")

export_data = {
    key: result_geo.get(key)
    for key in GEOMETRY_FIELDS
}

export_data.update(
    {
        "bbox_mm": str(result_geo.get("bbox_mm")),
        "surface_area_cm2": result_geo.get("surface_area_cm2"),
        "complexity_score": result_geo.get("complexity_score"),
    }
)

export_data.update(params)

export_data.update(
    {
        f"shot_{key}": value
        for key, value in shot_profile.items()
    }
)

export_data["model_source"] = result.get("model_source", "Unknown")

output_df = pd.DataFrame([export_data])

st.download_button(
    label="⬇️ Download Complete HPDC Report (CSV)",
    data=output_df.to_csv(index=False).encode("utf-8"),
    file_name="hpdc_prediction.csv",
    mime="text/csv",
    use_container_width=True,
)

st.caption(
    "Engineering estimates must be validated against the component "
    "CAD, alloy specifications, die design and machine limits "
    "before production use."
)
