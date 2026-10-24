# HPDC Smart Die-Casting - Streamlit app

STEP file -> geometry features -> predicted HPDC set-up parameters.

## Run
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt     # cadquery is easiest via: conda install -c conda-forge cadquery
    streamlit run streamlit_app.py

## Layout
- streamlit_app.py      UI (upload, extracted data, 3D preview, predictions, CSV download)
- app/step_analyzer.py  STEP -> volume, projected area, wall thickness, thin-wall fraction, complexity
- app/mesh_metrics.py   numpy geometry maths
- app/process_model.py  rule-based baseline + optional trained ML models
- train_model.py        train from your plant CSV -> models/hpdc_models.joblib

## Deploy
Streamlit Community Cloud works; put `cadquery` in requirements.txt (or use a conda environment.yml).
