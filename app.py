"""
app.py
─────────────────────────────────────────────────────────────────────────────
IrriSmart AI — Flask application.
"""

import os
import numpy as np
import pandas as pd
from flask import (
    Flask, render_template, request,
    redirect, url_for, flash, jsonify,
)
from ml.model import get_state, predict_single

app = Flask(__name__)
app.secret_key = "smart_irrigation_2024_secret"


@app.template_filter("sum_values")
def sum_values_filter(d):
    try:
        return sum(d.values())
    except Exception:
        return 0


@app.context_processor
def inject_globals():
    return dict(get_state=get_state)


# ══════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    state   = get_state()
    trained = state.get("trained", False)

    if trained:
        tc               = state.get("target_counts", {})
        dominant_need    = max(tc, key=tc.get) if tc else "—"
        cd               = state.get("cluster_dist", {})
        dominant_cluster = max(cd, key=cd.get) if cd else "—"
        total            = sum(tc.values()) or 1
        need_fill        = round(tc.get(dominant_need, 0) / total * 100)
        avg_water        = round(state.get("rmse", 0) + state.get("mae", 0), 1)
        water_fill       = min(round((avg_water / 60) * 100), 100)
        clust_fill       = round(
            max(cd.values()) / sum(cd.values()) * 100
        ) if cd else 45
    else:
        dominant_need    = "—"
        avg_water        = "—"
        dominant_cluster = "—"
        need_fill        = 0
        water_fill       = 0
        clust_fill       = 0

    hero_stats = {
        "total_records" : f"{state['n_rows']:,}"      if trained else "10,000+",
        "total_models"  : "3",
        "total_features": str(state["n_features"])    if trained else "15",
        "acc_lgb"       : f"{state['acc_lgb']:.2%}"  if trained else "—",
        "r2"            : f"{state['r2']:.4f}"        if trained else "—",
    }

    hero_visuals = {
        "irrigation_need": dominant_need,
        "need_fill"      : need_fill,
        "water_needed"   : f"{avg_water} mm" if trained else "—",
        "water_fill"     : water_fill,
        "field_cluster"  : dominant_cluster,
        "clust_fill"     : clust_fill,
    }

    summary_items = []
    if trained:
        summary_items = [
            ("Dataset Rows",    f"{state['n_rows']:,}",      "#3B82F6", "database"),
            ("Dataset Columns", str(state["n_cols"]),         "#8B5CF6", "table"),
            ("Features Used",   str(state["n_features"]),     "#10B981", "list"),
            ("LightGBM Acc",    f"{state['acc_lgb']:.4f}",   "#3B82F6", "zap"),
            ("CV Mean Acc",     f"{state['cv_mean']:.4f}",   "#8B5CF6", "activity"),
            ("CV Std Dev",      f"{state['cv_std']:.4f}",    "#6366F1", "minus"),
            ("RMSE",            f"{state['rmse']:.4f}",       "#EF4444", "bar-chart"),
            ("MAE",             f"{state['mae']:.4f}",        "#F97316", "minus-circle"),
            ("R² Score",        f"{state['r2']:.4f}",         "#10B981", "trending-up"),
        ]

    return render_template(
        "index.html",
        trained       = trained,
        hero_stats    = hero_stats,
        hero_visuals  = hero_visuals,
        summary_items = summary_items,
    )


@app.route("/dashboard")
def dashboard():
    state = get_state()
    if not state.get("trained"):
        flash("Model is not loaded.", "warning")
        return redirect(url_for("index"))

    dash = {
        "acc_lgb"      : f"{state['acc_lgb']:.4f}",
        "cv_mean"      : f"{state['cv_mean']:.4f}",
        "cv_std"       : f"{state['cv_std']:.4f}",
        "r2"           : f"{state['r2']:.4f}",
        "rmse"         : f"{state['rmse']:.4f}",
        "mae"          : f"{state['mae']:.4f}",
        "n_rows"       : f"{state['n_rows']:,}",
        "n_cols"       : str(state["n_cols"]),
        "n_features"   : str(state["n_features"]),
        "report"       : state.get("report_lgb", {}),
        "cv_scores"    : state.get("cv_scores", []),
        "target_counts": state.get("target_counts", {}),
        "cluster_dist" : state.get("cluster_dist", {}),
        "plots"        : state.get("plots", {}),
    }

    return render_template("dashboard.html", state=state, dash=dash)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    state = get_state()
    if not state.get("trained"):
        flash("Model is not loaded.", "warning")
        return redirect(url_for("index"))

    result = None
    if request.method == "POST":
        result = predict_single(request.form.to_dict())

    cat_options = {
        col: sorted(list(enc.classes_))
        for col, enc in state.get("encoders", {}).items()
    }

    return render_template(
        "predict.html",
        state       = state,
        result      = result,
        cat_options = cat_options,
        acc_lgb     = f"{state['acc_lgb']:.4f}",
        r2          = f"{state['r2']:.4f}",
    )


@app.route("/field-predict", methods=["GET", "POST"])
def field_predict():
    state = get_state()
    if not state.get("trained"):
        flash("Model is not loaded.", "warning")
        return redirect(url_for("index"))

    result = None
    if request.method == "POST":
        result = _full_field_prediction(request.form.to_dict())

    cat_options = {
        col: sorted(list(enc.classes_))
        for col, enc in state.get("encoders", {}).items()
    }

    feature_ranges = {
        "Soil_pH"                : {"min": 4.0,  "max": 9.0,   "step": 0.01, "default": 6.5,  "unit": "pH"},
        "Soil_Moisture"          : {"min": 0.0,  "max": 100.0, "step": 0.1,  "default": 35.0, "unit": "%"},
        "Organic_Carbon"         : {"min": 0.0,  "max": 10.0,  "step": 0.01, "default": 2.5,  "unit": "%"},
        "Electrical_Conductivity": {"min": 0.0,  "max": 5.0,   "step": 0.01, "default": 0.8,  "unit": "dS/m"},
        "Temperature_C"          : {"min": 0.0,  "max": 55.0,  "step": 0.1,  "default": 28.0, "unit": "°C"},
        "Humidity"               : {"min": 0.0,  "max": 100.0, "step": 0.1,  "default": 60.0, "unit": "%"},
        "Rainfall_mm"            : {"min": 0.0,  "max": 300.0, "step": 0.1,  "default": 10.0, "unit": "mm"},
        "Sunlight_Hours"         : {"min": 0.0,  "max": 16.0,  "step": 0.1,  "default": 8.0,  "unit": "hrs"},
        "Wind_Speed_kmh"         : {"min": 0.0,  "max": 120.0, "step": 0.1,  "default": 15.0, "unit": "km/h"},
        "Field_Area_hectare"     : {"min": 0.1,  "max": 500.0, "step": 0.1,  "default": 5.0,  "unit": "ha"},
        "Previous_Irrigation_mm" : {"min": 0.0,  "max": 200.0, "step": 0.1,  "default": 20.0, "unit": "mm"},
    }

    return render_template(
        "field_predict.html",
        state          = state,
        result         = result,
        cat_options    = cat_options,
        feature_ranges = feature_ranges,
        acc_lgb        = f"{state['acc_lgb']:.4f}",
        r2             = f"{state['r2']:.4f}",
        form_data      = request.form.to_dict() if request.method == "POST" else {},
    )


@app.route("/analysis")
def analysis():
    state = get_state()
    if not state.get("trained"):
        flash("Model is not loaded.", "warning")
        return redirect(url_for("index"))

    analysis_metrics = {
        "acc_lgb" : f"{state['acc_lgb']:.4f}",
        "cv_mean" : f"{state['cv_mean']:.4f}",
        "cv_std"  : f"{state['cv_std']:.4f}",
        "rmse"    : f"{state['rmse']:.4f}",
        "mae"     : f"{state['mae']:.4f}",
        "r2"      : f"{state['r2']:.4f}",
        "report"  : state.get("report_lgb", {}),
    }

    return render_template(
        "analysis.html",
        state            = state,
        plots            = state.get("plots", {}),
        analysis_metrics = analysis_metrics,
        cluster_dist     = state.get("cluster_dist", {}),
    )


@app.route("/about")
def about():
    state   = get_state()
    trained = state.get("trained", False)

    about_stats = {
        "total_records" : f"{state['n_rows']:,}"     if trained else "10,000+",
        "total_cols"    : str(state["n_cols"])        if trained else "30",
        "total_features": str(state["n_features"])    if trained else "15",
        "acc_lgb"       : f"{state['acc_lgb']:.4f}"  if trained else "—",
        "r2"            : f"{state['r2']:.4f}"        if trained else "—",
        "rmse"          : f"{state['rmse']:.4f}"      if trained else "—",
        "cv_mean"       : f"{state['cv_mean']:.4f}"   if trained else "—",
        "cv_std"        : f"{state['cv_std']:.4f}"    if trained else "—",
    }

    return render_template(
        "about.html",
        state       = state,
        trained     = trained,
        about_stats = about_stats,
    )


@app.route("/api/metrics")
def api_metrics():
    state = get_state()
    if not state.get("trained"):
        return jsonify({"error": "Model not loaded"}), 503

    return jsonify({
        "acc_lgb"      : state["acc_lgb"],
        "cv_mean"      : state["cv_mean"],
        "cv_std"       : state["cv_std"],
        "cv_scores"    : state["cv_scores"],
        "rmse"         : state["rmse"],
        "mae"          : state["mae"],
        "r2"           : state["r2"],
        "target_counts": state["target_counts"],
        "cluster_dist" : state["cluster_dist"],
        "n_rows"       : state["n_rows"],
        "n_cols"       : state["n_cols"],
        "n_features"   : state["n_features"],
    })


# ══════════════════════════════════════════════════════════════════════════
#  FULL FIELD PREDICTION
# ══════════════════════════════════════════════════════════════════════════

def _full_field_prediction(form_data: dict) -> dict:
    state = get_state()
    if not state.get("trained"):
        return {"error": "Model not loaded."}

    clf            = state["clf"]
    reg            = state["reg"]
    encoders       = state["encoders"]
    ALL_FEATURES   = state["ALL_FEATURES"]
    NUMERIC_FEATS  = state["NUMERIC_FEATS"]
    cat_cols       = state["cat_cols"]
    target_map_inv = state["target_map_inv"]
    scaler         = state["scaler"]

    # ── Build feature row ─────────────────────────────────────────────────
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

    X = pd.DataFrame([row])[ALL_FEATURES]

    # ── Classification ────────────────────────────────────────────────────
    clf_pred    = int(clf.predict(X)[0])
    clf_proba   = clf.predict_proba(X)[0].tolist()
    irr_need    = target_map_inv[clf_pred]
    needs_water = irr_need in ["Medium", "High"]

    # Confidence = probability of the predicted class (0-100%)
    # This is NOT the same as model accuracy.
    # Accuracy = how often the model is correct across the whole test set.
    # Confidence = how certain the model is about THIS specific prediction.
    confidence = round(max(clf_proba) * 100, 1)

    # ── Regression ───────────────────────────────────────────────────────
    water_mm = float(reg.predict(X)[0])

    # ── Clustering ───────────────────────────────────────────────────────
    CLUSTER_FEATS = [
        "Soil_Moisture", "Temperature_C", "Humidity",
        "Rainfall_mm", "Soil_pH", "Previous_Irrigation_mm",
        "Field_Area_hectare", "Wind_Speed_kmh",
    ]
    cluster_row    = np.array([[row.get(f, 0.0) for f in CLUSTER_FEATS]])
    cluster_scaled = scaler.transform(cluster_row)
    kmeans         = state.get("kmeans_model")
    cluster_id     = int(kmeans.predict(cluster_scaled)[0]) if kmeans else 0

    cluster_meta = {
        0: {
            "label"      : "💧 High Water Demand",
            "description": (
                "Fields in this cluster have high soil temperature, "
                "low moisture, and minimal rainfall. "
                "Frequent and substantial irrigation is essential."
            ),
            "color": "#EF4444",
            "icon" : "droplets",
        },
        1: {
            "label"      : "🌧 Rain-Fed Zone",
            "description": (
                "Fields in this cluster receive adequate natural rainfall. "
                "Supplemental irrigation may only be needed during dry spells."
            ),
            "color": "#3B82F6",
            "icon" : "cloud-rain",
        },
        2: {
            "label"      : "☀️ Dry Zone",
            "description": (
                "Fields in this cluster are in arid conditions with "
                "very low humidity and rainfall. "
                "Careful irrigation scheduling is strongly recommended."
            ),
            "color": "#F59E0B",
            "icon" : "sun",
        },
    }
    cluster_info = cluster_meta.get(cluster_id, cluster_meta[0])

    # ── Risk Level ───────────────────────────────────────────────────────
    high_pct = clf_proba[2] * 100
    if high_pct >= 70:
        risk_level, risk_color, risk_icon = "Critical", "#EF4444", "alert-triangle"
    elif high_pct >= 40 or clf_proba[1] * 100 >= 60:
        risk_level, risk_color, risk_icon = "Moderate", "#F59E0B", "alert-circle"
    else:
        risk_level, risk_color, risk_icon = "Low", "#10B981", "check-circle"

    # ── Feature Contributions ────────────────────────────────────────────
    # FIX: LightGBM feature_importances_ are raw split counts (e.g. 368900),
    # NOT percentages. We must normalize to 0-100% relative to total.
    raw_importances = clf.feature_importances_.astype(float)
    total_imp       = raw_importances.sum()

    feat_contribs = []
    for feat, raw_imp in zip(ALL_FEATURES, raw_importances):
        # Normalize: each feature's share of total importance as a percentage
        pct = round((raw_imp / total_imp) * 100, 2) if total_imp > 0 else 0.0
        feat_contribs.append({
            "feature"   : feat.replace("_enc", "").replace("_", " "),
            "importance": pct,                          # true 0-100 %
            "value"     : round(float(X[feat].iloc[0]), 3),
        })
    feat_contribs.sort(key=lambda x: x["importance"], reverse=True)
    top_features = feat_contribs[:8]

    # ── Weekly Schedule ──────────────────────────────────────────────────
    daily_need = water_mm / 7 if needs_water else 0
    schedule   = []
    days       = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    factors    = [1.10, 0.90, 1.05, 1.00, 0.95, 1.15, 0.85]
    for day, factor in zip(days, factors):
        amt = round(daily_need * factor, 2) if needs_water else 0
        schedule.append({
            "day"        : day,
            "amount_mm"  : amt,
            "recommended": needs_water and amt > 0,
        })

    # ── AI Recommendation ────────────────────────────────────────────────
    # Rule-based recommendation generated from the prediction results.
    # Uses: predicted class, water_mm, soil_moisture, temperature, rainfall.
    soil_m   = row.get("Soil_Moisture", 50)
    temp     = row.get("Temperature_C", 25)
    rainfall = row.get("Rainfall_mm", 0)
    prev_irr = row.get("Previous_Irrigation_mm", 0)

    if irr_need == "High":
        rec = (
            f"⚠️ Immediate irrigation required. Apply {water_mm:.1f} mm "
            f"as soon as possible. Soil moisture ({soil_m:.1f}%) is critically "
            f"low and temperature ({temp:.1f}°C) is elevated."
        )
    elif irr_need == "Medium":
        rec = (
            f"📋 Schedule irrigation within 24–48 hours. Apply approximately "
            f"{water_mm:.1f} mm. Current rainfall ({rainfall:.1f} mm) provides "
            f"partial coverage."
        )
    else:
        rec = (
            f"✅ No immediate irrigation needed. Soil moisture ({soil_m:.1f}%) "
            f"is adequate. Previous irrigation ({prev_irr:.1f} mm) is still "
            f"effective. Next check recommended in 3–5 days."
        )

    # ── Water Volume & Cost ───────────────────────────────────────────────
    field_area  = row.get("Field_Area_hectare", 1.0)
    total_water = water_mm * field_area * 10
    cost_est    = round(total_water * 0.002, 2)

    return {
        "irrigation_need"    : irr_need,
        "needs_water"        : needs_water,
        "proba_low"          : round(clf_proba[0] * 100, 1),
        "proba_medium"       : round(clf_proba[1] * 100, 1),
        "proba_high"         : round(clf_proba[2] * 100, 1),
        "water_need_mm"      : round(water_mm, 2),
        "cluster_id"         : cluster_id,
        "cluster_label"      : cluster_info["label"],
        "cluster_description": cluster_info["description"],
        "cluster_color"      : cluster_info["color"],
        "cluster_icon"       : cluster_info["icon"],
        "risk_level"         : risk_level,
        "risk_color"         : risk_color,
        "risk_icon"          : risk_icon,
        "recommendation"     : rec,
        "top_features"       : top_features,
        "schedule"           : schedule,
        "total_water_litres" : round(total_water, 1),
        "cost_estimate"      : cost_est,
        "field_area"         : field_area,
        "confidence"         : confidence,
    }


if __name__ == "__main__":
    app.run(debug=True, port=5000)