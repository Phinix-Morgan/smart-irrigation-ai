"""
ml/model.py
─────────────────────────────────────────────────────────────────────────────
Production model loader.
Loads pre-trained artefacts from ml/saved_model/ at startup.
No training happens here — training is done via train_and_save.py locally.
"""

import warnings

warnings.filterwarnings("ignore")

import os

import joblib
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
_SAVE_DIR = os.path.join(_BASE, "saved_model")

# ── Global state ───────────────────────────────────────────────────────────
_state: dict = {}


def _load_artefacts() -> bool:
    """
    Load all saved model artefacts from disk into _state.
    Returns True on success, False if files are missing.
    """
    global _state

    required = [
        "classifier.pkl",
        "regressor.pkl",
        "kmeans.pkl",
        "scaler.pkl",
        "encoders.pkl",
        "state_meta.pkl",
    ]

    missing = [f for f in required if not os.path.exists(os.path.join(_SAVE_DIR, f))]

    if missing:
        print(f"⚠️  Missing model files: {missing}")
        print(f"   Run train_and_save.py locally first.")
        return False

    try:
        print("🔄 Loading pre-trained model artefacts…")

        meta = joblib.load(os.path.join(_SAVE_DIR, "state_meta.pkl"))
        clf = joblib.load(os.path.join(_SAVE_DIR, "classifier.pkl"))
        reg = joblib.load(os.path.join(_SAVE_DIR, "regressor.pkl"))
        kmeans = joblib.load(os.path.join(_SAVE_DIR, "kmeans.pkl"))
        scaler = joblib.load(os.path.join(_SAVE_DIR, "scaler.pkl"))
        encoders = joblib.load(os.path.join(_SAVE_DIR, "encoders.pkl"))

        # Merge model objects into the meta state
        _state = {
            **meta,
            "clf": clf,
            "reg": reg,
            "kmeans_model": kmeans,
            "scaler": scaler,
            "encoders": encoders,
            "trained": True,
        }

        print(f"✅ Models loaded successfully.")
        print(f"   Dataset : {_state['n_rows']:,} rows × {_state['n_cols']} cols")
        print(
            f"   Accuracy: {_state['acc_tuned']:.4f} (tuned)  |  R²: {_state['r2']:.4f}"
        )
        return True

    except Exception as e:
        print(f"❌ Failed to load model artefacts: {e}")
        _state = {}
        return False


def get_state() -> dict:
    """Return the current global state (loaded models + metrics)."""
    return _state


# ── Load on import ─────────────────────────────────────────────────────────
# This runs once when Flask imports the module — zero cost on every request.
_load_artefacts()


# ── Prediction helpers ─────────────────────────────────────────────────────


def _build_feature_row(form_data: dict) -> pd.DataFrame:
    """
    Convert raw form data dict into a single-row DataFrame
    with all features in the correct order.
    """
    ALL_FEATURES = _state["ALL_FEATURES"]
    NUMERIC_FEATS = _state["NUMERIC_FEATS"]
    cat_cols = _state["cat_cols"]
    encoders = _state["encoders"]

    row = {}

    for feat in NUMERIC_FEATS:
        try:
            row[feat] = float(form_data.get(feat, 0))
        except (ValueError, TypeError):
            row[feat] = 0.0

    for col in cat_cols:
        enc_col = col + "_enc"
        if enc_col in ALL_FEATURES:
            enc = encoders.get(col)
            val = form_data.get(col, "")
            try:
                row[enc_col] = int(enc.transform([val])[0])
            except Exception:
                row[enc_col] = 0

    return pd.DataFrame([row])[ALL_FEATURES]


def predict_single(form_data: dict) -> dict:
    """
    Quick single-row prediction (used by /predict route).
    Returns irrigation_need, probabilities, and water_need_mm.
    """
    if not _state.get("trained"):
        return {"error": "Model not loaded. Check server logs."}

    clf = _state["clf"]
    reg = _state["reg"]
    target_map_inv = _state["target_map_inv"]

    X = _build_feature_row(form_data)
    clf_pred = int(clf.predict(X)[0])
    clf_proba = clf.predict_proba(X)[0].tolist()
    reg_pred = float(reg.predict(X)[0])

    return {
        "irrigation_need": target_map_inv[clf_pred],
        "proba_low": round(clf_proba[0] * 100, 1),
        "proba_medium": round(clf_proba[1] * 100, 1),
        "proba_high": round(clf_proba[2] * 100, 1),
        "water_need_mm": round(reg_pred, 2),
    }
