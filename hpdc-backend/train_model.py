"""Train ML models from real plant data.

CSV columns required: the FEATURES and TARGETS in app/process_model.py, plus 'alloy_type'.
One row = one good-quality production setup (die/component + the parameters that worked).

    python train_model.py data/setups.csv
"""
import sys, joblib, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from app.process_model import FEATURES, TARGETS, MODEL_PATH

df = pd.read_csv(sys.argv[1]).dropna(subset=FEATURES + TARGETS)
X = df[FEATURES].values
bundle = {"regressors": {}}
for t in TARGETS:
    m = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8)
    r2 = cross_val_score(m, X, df[t].values, cv=min(5, len(df)), scoring="r2").mean()
    print(f"{t:32s} CV R2 = {r2:.3f}")
    bundle["regressors"][t] = m.fit(X, df[t].values)
if "alloy_type" in df:
    bundle["alloy_clf"] = RandomForestClassifier(n_estimators=200).fit(X, df["alloy_type"])
joblib.dump(bundle, MODEL_PATH)
print("saved", MODEL_PATH)
