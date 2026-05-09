import warnings

warnings.filterwarnings("ignore")

import os
import sys

import matplotlib

matplotlib.use("Agg")  # Headless — no display required

import joblib
import lightgbm as lgb
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR   = os.path.join(BASE_DIR, "ml", "saved_model")
STATIC_IMG = os.path.join(BASE_DIR, "static", "images")
CSV_PATH   = os.path.join(BASE_DIR, "uploads", "irrigation_prediction.csv")

os.makedirs(SAVE_DIR,   exist_ok=True)
os.makedirs(STATIC_IMG, exist_ok=True)

# ── Plot style ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update(
    {
        "figure.dpi"      : 130,
        "axes.titlesize"  : 13,
        "axes.labelsize"  : 11,
        "figure.facecolor": "white",
    }
)


# ── Helper ─────────────────────────────────────────────────────────────────
def _save(name: str, fig) -> str:
    """Save figure to static/images and return the relative web path."""
    path = os.path.join(STATIC_IMG, name)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return f"images/{name}"


# ── Main training routine ──────────────────────────────────────────────────
def run(csv_path: str):
    print(f"\n{'=' * 60}")
    print("  IrriSmart AI — Training & Artefact Export")
    print(f"{'=' * 60}\n")

    # ── 1. Load ──────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    print(f"  Dataset loaded — Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")

    # ── 2. Clean ─────────────────────────────────────────────────────────
    total_missing = df.isnull().sum().sum()
    if total_missing == 0:
        print("  No missing values found.")
    else:
        df.fillna(df.median(numeric_only=True), inplace=True)
        print(f"  {total_missing} missing values filled with column medians.")

    # ── 3. Encode categoricals ───────────────────────────────────────────
    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  ROOT-CAUSE FIX:                                                ║
    # ║  Old code reused ONE LabelEncoder instance for all columns,     ║
    # ║  so encoders[col] pointed to the SAME object (last col fitted). ║
    # ║  Fix: create a NEW LabelEncoder for every categorical column.   ║
    # ╚══════════════════════════════════════════════════════════════════╝
    encoders = {}
    cat_cols  = df.select_dtypes(include="object").columns.tolist()

    print(f"\n  Categorical columns found: {cat_cols}")

    for col in cat_cols:
        if col == "Irrigation_Need":
            continue                          # handled separately below
        enc              = LabelEncoder()     # ← NEW instance per column
        df[col + "_enc"] = enc.fit_transform(df[col])
        encoders[col]    = enc                # ← unique encoder saved
        print(
            f"  Encoded: {col:30s} "
            f"→ {len(enc.classes_)} classes: {list(enc.classes_)}"
        )

    # Target encoding
    target_map     = {"Low": 0, "Medium": 1, "High": 2}
    target_map_inv = {0: "Low", 1: "Medium", 2: "High"}
    df["Irrigation_Need_enc"] = df["Irrigation_Need"].map(target_map)
    print(f"\n  Target map: {target_map}")

    # Verify each encoder is independent
    print("\n  Encoder verification (each col → own classes):")
    for col, enc in encoders.items():
        print(f"    {col:30s} → {list(enc.classes_)}")

    # ── 4. Regression proxy target ───────────────────────────────────────
    water_map = {"Low": 1, "Medium": 2, "High": 3}
    np.random.seed(42)
    df["Water_Need_mm"] = (
        df["Irrigation_Need"].map(water_map) * 15
        + df["Temperature_C"] * 0.5
        - df["Rainfall_mm"]   * 0.01
        - df["Soil_Moisture"] * 0.2
        + np.random.normal(0, 3, len(df))
    ).clip(lower=5)
    print(
        f"\n  Water_Need_mm stats:\n"
        f"{df['Water_Need_mm'].describe().round(2)}"
    )

    # ── 5. Feature sets ──────────────────────────────────────────────────
    NUMERIC_FEATS = [
        "Soil_pH",
        "Soil_Moisture",
        "Organic_Carbon",
        "Electrical_Conductivity",
        "Temperature_C",
        "Humidity",
        "Rainfall_mm",
        "Sunlight_Hours",
        "Wind_Speed_kmh",
        "Field_Area_hectare",
        "Previous_Irrigation_mm",
    ]

    # Only include _enc columns that actually exist (from cat_cols above)
    ENCODED_FEATS = [
        c for c in df.columns
        if c.endswith("_enc") and c != "Irrigation_Need_enc"
    ]
    ALL_FEATURES = NUMERIC_FEATS + ENCODED_FEATS
    print(f"\n  Features ({len(ALL_FEATURES)}):")
    for f in ALL_FEATURES:
        print(f"    - {f}")

    # ── 6. Classification — train/test split ─────────────────────────────
    X_clf = df[ALL_FEATURES]
    y_clf = df["Irrigation_Need_enc"]

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    print(
        f"\n  Classification — "
        f"train: {X_train_c.shape[0]}  test: {X_test_c.shape[0]}"
    )

    # ── 7. LightGBM Classifier ───────────────────────────────────────────
    print("\n  Training LightGBM Classifier…")
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbose=-1,
    )
    lgb_clf.fit(X_train_c, y_train_c)
    y_pred_lgb = lgb_clf.predict(X_test_c)
    acc_lgb    = float(accuracy_score(y_test_c, y_pred_lgb))

    print("  Running 5-fold cross-validation…")
    cv_scores_lgb = cross_val_score(
        lgb_clf, X_clf, y_clf, cv=5, scoring="accuracy", n_jobs=-1
    )
    report_lgb = classification_report(
        y_test_c,
        y_pred_lgb,
        target_names=["Low", "Medium", "High"],
        output_dict=True,
    )

    print(f"\n  LightGBM accuracy : {acc_lgb:.4f}")
    print(
        f"  CV scores         : {np.round(cv_scores_lgb, 4)}"
    )
    print(
        f"  CV mean ± std     : {cv_scores_lgb.mean():.4f}"
        f" ± {cv_scores_lgb.std():.4f}"
    )
    print(f"\n  LightGBM Classification Report:")
    print(
        classification_report(
            y_test_c, y_pred_lgb,
            target_names=["Low", "Medium", "High"]
        )
    )

    # ── 8. Random Forest Regressor ───────────────────────────────────────
    print("  Training Random Forest Regressor…")
    X_reg = df[ALL_FEATURES]
    y_reg = df["Water_Need_mm"]

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    reg      = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    reg.fit(X_train_r, y_train_r)
    y_pred_r = reg.predict(X_test_r)

    rmse = float(np.sqrt(mean_squared_error(y_test_r, y_pred_r)))
    mae  = float(mean_absolute_error(y_test_r, y_pred_r))
    r2   = float(r2_score(y_test_r, y_pred_r))

    print(
        f"  RF Regressor — RMSE: {rmse:.4f}  "
        f"MAE: {mae:.4f}  R²: {r2:.4f}"
    )

    # ── 9. K-Means Clustering ─────────────────────────────────────────────
    print("\n  Running K-Means clustering…")
    CLUSTER_FEATS = [
        "Soil_Moisture",
        "Temperature_C",
        "Humidity",
        "Rainfall_mm",
        "Soil_pH",
        "Previous_Irrigation_mm",
        "Field_Area_hectare",
        "Wind_Speed_kmh",
    ]

    scaler  = StandardScaler()
    X_clust = scaler.fit_transform(df[CLUSTER_FEATS])

    np.random.seed(42)
    idx  = np.random.choice(len(X_clust), size=3000, replace=False)
    X_s  = X_clust[idx]
    df_s = df.iloc[idx].copy()

    # Elbow method
    inertias = []
    k_range  = range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_s)
        inertias.append(km.inertia_)

    # Final K-Means k=3
    kmeans            = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_s["KMeans_Cluster"] = kmeans.fit_predict(X_s)

    cluster_labels = {
        0: "Cluster A: High Water Demand",
        1: "Cluster B: Rain-Fed Zone",
        2: "Cluster C: Dry Zone",
    }
    df_s["KMeans_Label"] = df_s["KMeans_Cluster"].map(cluster_labels)

    print("  KMeans cluster distribution:")
    print(df_s["KMeans_Label"].value_counts().to_string())

    # PCA 2-D
    pca   = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_s)
    pc1   = round(pca.explained_variance_ratio_[0] * 100, 1)
    pc2   = round(pca.explained_variance_ratio_[1] * 100, 1)

    # ── 10. Generate & save all plots ────────────────────────────────────
    print("\n  Generating plots…")
    plots = {}

    # Shared variables used by multiple plots
    order  = ["Low", "Medium", "High"]
    counts = [int(df["Irrigation_Need"].value_counts().get(k, 0)) for k in order]
    cm_lgb = confusion_matrix(y_test_c, y_pred_lgb)
    colors_plot = ["royalblue", "seagreen", "darkorange"]
    labels_plot = list(cluster_labels.values())
    lim_min     = float(min(y_test_r.min(), y_pred_r.min()))
    lim_max     = float(max(y_test_r.max(), y_pred_r.max()))

    # ── (a) EDA Feature Distributions ────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "EDA: Feature Distributions", fontsize=15, fontweight="bold"
    )
    num_cols_eda = [
        "Soil_Moisture", "Temperature_C", "Humidity",
        "Rainfall_mm",   "Soil_pH",       "Previous_Irrigation_mm",
    ]
    for ax, col in zip(axes.flat, num_cols_eda):
        sns.histplot(df[col], kde=True, ax=ax, color="steelblue", bins=35)
        ax.set_title(col.replace("_", " "))
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plots["eda_distributions"] = _save("eda_distributions.png", fig)
    print("    ✓ eda_distributions.png")

    # ── (b) Target Class Distribution ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        order, counts,
        color=["#90CAF9", "#42A5F5", "#1565C0"],
        edgecolor="white", linewidth=1.5, width=0.55,
    )
    for b, c in zip(bars, counts):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 50,
            f"{c:,}", ha="center", fontsize=10, fontweight="bold",
        )
    ax.set_title(
        "Target Class Distribution – Irrigation Need",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylabel("Count")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plots["target_distribution"] = _save("target_distribution.png", fig)
    print("    ✓ target_distribution.png")

    # ── (c) Correlation Heatmap ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 9))
    corr = df.select_dtypes(include=np.number).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", linewidths=0.5, ax=ax,
    )
    ax.set_title(
        "Correlation Matrix – Numeric Features",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plots["correlation_heatmap"] = _save("correlation_heatmap.png", fig)
    print("    ✓ correlation_heatmap.png")

    # ── (d) LightGBM Confusion Matrix ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm_lgb, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Low", "Medium", "High"],
        yticklabels=["Low", "Medium", "High"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(
        f"Confusion Matrix – LightGBM  (Acc = {acc_lgb:.4f})",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plots["confusion_matrix_lightgbm"] = _save(
        "confusion_matrix_lightgbm.png", fig
    )
    print("    ✓ confusion_matrix_lightgbm.png")

    # ── (e) Feature Importance — LightGBM ────────────────────────────────
    imp_lgb = (
        pd.Series(lgb_clf.feature_importances_, index=ALL_FEATURES)
        .sort_values(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(9, 8))
    imp_lgb.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(
        "Feature Importance – LightGBM Classification",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Importance Score")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plots["feature_importance_lightgbm"] = _save(
        "feature_importance_lightgbm.png", fig
    )
    print("    ✓ feature_importance_lightgbm.png")

    # ── (f) Regression — Actual vs Predicted + Residuals ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Water Usage Regression Results", fontsize=13, fontweight="bold"
    )
    axes[0].scatter(
        y_test_r.values[:500], y_pred_r[:500],
        alpha=0.3, color="teal", s=15,
    )
    axes[0].plot(
        [lim_min, lim_max], [lim_min, lim_max],
        "r--", linewidth=1.5, label="Perfect fit",
    )
    axes[0].set_title(
        f"Actual vs Predicted Water Need  (R² = {r2:.3f})",
        fontsize=13, fontweight="bold",
    )
    axes[0].set_xlabel("Actual Water Need (mm)")
    axes[0].set_ylabel("Predicted Water Need (mm)")
    axes[0].legend()

    residuals = y_test_r.values - y_pred_r
    axes[1].hist(residuals, bins=50, color="coral", edgecolor="white")
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1.5)
    axes[1].set_title(
        "Residual Distribution", fontsize=13, fontweight="bold"
    )
    axes[1].set_xlabel("Residual Error (Actual – Predicted)")
    axes[1].set_ylabel("Frequency")
    plt.tight_layout()
    plots["regression_results"] = _save("regression_results.png", fig)
    print("    ✓ regression_results.png")

    # ── (g) Feature Importance — Regression ──────────────────────────────
    imp_r = (
        pd.Series(reg.feature_importances_, index=ALL_FEATURES)
        .sort_values(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(9, 8))
    imp_r.plot(kind="barh", ax=ax, color="darkorange")
    ax.set_title(
        "Feature Importance – Random Forest Regression",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Importance Score")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plots["feature_importance_regression"] = _save(
        "feature_importance_regression.png", fig
    )
    print("    ✓ feature_importance_regression.png")

    # ── (h) K-Means Elbow ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(k_range), inertias, "bo-", markersize=8)
    ax.axvline(3, color="red", linestyle="--", alpha=0.7, label="Chosen k=3")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia (Within-cluster SSE)")
    ax.set_title(
        "K-Means Elbow Method – Optimal k Selection",
        fontsize=12, fontweight="bold",
    )
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plots["kmeans_elbow"] = _save("kmeans_elbow.png", fig)
    print("    ✓ kmeans_elbow.png")

    # ── (i) PCA Cluster Scatter ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for cl, color, label in zip([0, 1, 2], colors_plot, labels_plot):
        mask = df_s["KMeans_Cluster"] == cl
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            alpha=0.6, s=15, color=color, label=label, linewidths=0,
        )
    ax.set_title(
        "K-Means Clusters (PCA 2-D Projection)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel(f"PC1 ({pc1}%)")
    ax.set_ylabel(f"PC2 ({pc2}%)")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plots["clustering_pca_visualisation"] = _save(
        "clustering_pca_visualisation.png", fig
    )
    print("    ✓ clustering_pca_visualisation.png")

    # ── (j) Radar Chart — Cluster Profiles ───────────────────────────────
    cluster_means      = df_s.groupby("KMeans_Cluster")[CLUSTER_FEATS[:6]].mean()
    mn_r_v, mx_r_v     = cluster_means.min(), cluster_means.max()
    cluster_means_norm = (cluster_means - mn_r_v) / (mx_r_v - mn_r_v + 1e-9)
    labels_r_v         = [c.replace("_", " ") for c in CLUSTER_FEATS[:6]]
    angles             = np.linspace(
        0, 2 * np.pi, len(labels_r_v), endpoint=False
    ).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    colors_r = ["royalblue", "seagreen", "darkorange"]
    cnames_r = list(cluster_labels.values())

    for row_v, color, name in zip(
        cluster_means_norm.values, colors_r, cnames_r
    ):
        vals = row_v.tolist() + row_v[:1].tolist()
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=name)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels_r_v)
    ax.set_title(
        "K-Means Cluster Profiles – Radar Chart",
        fontsize=13, fontweight="bold", pad=20,
    )
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, -0.05))
    plt.tight_layout()
    plots["cluster_radar_chart"] = _save("cluster_radar_chart.png", fig)
    print("    ✓ cluster_radar_chart.png")

    # ── (k) Final Summary Dashboard ───────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Smart Irrigation Prediction – Summary Dashboard",
        fontsize=18, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.4)

    # k1 — Target counts bar
    ax0   = fig.add_subplot(gs[0, 0])
    bars0 = ax0.bar(order, counts, color=["#90CAF9", "#42A5F5", "#1565C0"])
    for b, c in zip(bars0, counts):
        ax0.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 50,
            str(c), ha="center", fontsize=9,
        )
    ax0.set_title("Target Class Counts")
    ax0.set_ylabel("Count")

    # k2 — Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 1])
    sns.heatmap(
        cm_lgb, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Low", "Med", "High"],
        yticklabels=["Low", "Med", "High"],
        ax=ax1, cbar=False,
    )
    ax1.set_title(f"LGBM Confusion Matrix\n(Acc = {acc_lgb:.4f})")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")

    # k3 — Regression scatter
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(
        y_test_r.values[:500], y_pred_r[:500],
        alpha=0.3, color="teal", s=10,
    )
    ax2.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1)
    ax2.set_title(
        f"Regression: Actual vs Pred\n"
        f"(R² = {r2:.3f}  RMSE = {rmse:.2f})"
    )
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")

    # k4 — Top-10 Feature Importance
    ax3   = fig.add_subplot(gs[1, :2])
    top10 = (
        pd.Series(lgb_clf.feature_importances_, index=ALL_FEATURES)
        .nlargest(10)
        .sort_values()
    )
    top10.plot(kind="barh", ax=ax3, color="steelblue")
    ax3.set_title("Top 10 Features – LightGBM")
    ax3.set_xlabel("Importance Score")

    # k5 — Elbow mini
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(list(k_range), inertias, "bo-", markersize=6)
    ax4.axvline(3, color="red", linestyle="--", alpha=0.7)
    ax4.set_title("K-Means Elbow")
    ax4.set_xlabel("k")
    ax4.set_ylabel("Inertia")

    # k6 — PCA Cluster Scatter mini
    ax5 = fig.add_subplot(gs[2, :2])
    for cl, color, label in zip([0, 1, 2], colors_plot, labels_plot):
        mask = df_s["KMeans_Cluster"] == cl
        ax5.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            alpha=0.5, s=12, color=color, label=label,
        )
    ax5.set_title("K-Means Clusters (PCA Projection)")
    ax5.set_xlabel("PC1")
    ax5.set_ylabel("PC2")
    ax5.legend(fontsize=8, loc="upper right")

    # k7 — Metrics text box
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis("off")
    summary_text = (
        "MODEL METRICS SUMMARY\n"
        "─────────────────────────\n\n"
        "Classification (LightGBM)\n"
        f"  Accuracy  : {acc_lgb:.4f}\n"
        f"  CV Mean   : {cv_scores_lgb.mean():.4f}\n"
        f"  CV Std    : {cv_scores_lgb.std():.4f}\n\n"
        "Regression (Random Forest)\n"
        f"  RMSE      : {rmse:.3f}\n"
        f"  MAE       : {mae:.3f}\n"
        f"  R²        : {r2:.4f}\n\n"
        "Clustering (K-Means, k=3)\n"
        f"  C0 (High) : {(df_s['KMeans_Cluster'] == 0).sum()}\n"
        f"  C1 (Rain) : {(df_s['KMeans_Cluster'] == 1).sum()}\n"
        f"  C2 (Dry)  : {(df_s['KMeans_Cluster'] == 2).sum()}\n"
    )
    ax6.text(
        0.05, 0.97, summary_text,
        transform=ax6.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#EEF7FF",
            edgecolor="#90CAF9",
        ),
    )

    plt.savefig(
        os.path.join(STATIC_IMG, "final_summary_dashboard.png"),
        bbox_inches="tight", dpi=140,
    )
    plt.close(fig)
    plots["final_summary_dashboard"] = "images/final_summary_dashboard.png"
    print("    ✓ final_summary_dashboard.png")
    print(f"\n  All {len(plots)} plots saved to: {STATIC_IMG}/")

    # ── 11. Build state dict ──────────────────────────────────────────────
    cluster_dist = df_s["KMeans_Label"].value_counts().to_dict()

    state = {
        # ── plots
        "plots"         : plots,
        # ── dataset info
        "n_rows"        : int(df.shape[0]),
        "n_cols"        : int(df.shape[1]),
        "n_features"    : len(ALL_FEATURES),
        # ── classification metrics
        "acc_lgb"       : round(acc_lgb, 4),
        "cv_mean"       : round(float(cv_scores_lgb.mean()), 4),
        "cv_std"        : round(float(cv_scores_lgb.std()),  4),
        "cv_scores"     : [round(float(s), 4) for s in cv_scores_lgb],
        "report_lgb"    : report_lgb,
        # ── regression metrics
        "rmse"          : round(rmse, 4),
        "mae"           : round(mae,  4),
        "r2"            : round(r2,   4),
        # ── clustering
        "cluster_dist"  : cluster_dist,
        # ── target distribution
        "target_counts" : {
            k: int(df["Irrigation_Need"].value_counts().get(k, 0))
            for k in ["Low", "Medium", "High"]
        },
        # ── feature / encoding metadata
        "ALL_FEATURES"  : ALL_FEATURES,
        "NUMERIC_FEATS" : NUMERIC_FEATS,
        "cat_cols"      : [c for c in cat_cols if c != "Irrigation_Need"],
        "target_map_inv": target_map_inv,
        "trained"       : True,
    }

    # ── 12. Save all artefacts ────────────────────────────────────────────
    print("\n  Saving model artefacts…")

    joblib.dump(lgb_clf,   os.path.join(SAVE_DIR, "classifier.pkl"),  compress=3)
    joblib.dump(reg,       os.path.join(SAVE_DIR, "regressor.pkl"),   compress=3)
    joblib.dump(kmeans,    os.path.join(SAVE_DIR, "kmeans.pkl"),      compress=3)
    joblib.dump(scaler,    os.path.join(SAVE_DIR, "scaler.pkl"),      compress=3)
    joblib.dump(encoders,  os.path.join(SAVE_DIR, "encoders.pkl"),    compress=3)
    joblib.dump(state,     os.path.join(SAVE_DIR, "state_meta.pkl"),  compress=3)

    print(f"\n  Artefacts saved to: {SAVE_DIR}/")
    print("\n  Files written:")
    for f in sorted(os.listdir(SAVE_DIR)):
        size = os.path.getsize(os.path.join(SAVE_DIR, f)) / (1024 * 1024)
        print(f"    {f:<30s}  {size:.2f} MB")

    # ── 13. Final encoder summary ─────────────────────────────────────────
    print("\n  ── Encoder Summary (saved to encoders.pkl) ──")
    for col, enc in encoders.items():
        print(f"    {col:30s} → {list(enc.classes_)}")

    print(f"\n{'=' * 60}")
    print("  Training complete!")
    print("  Next step: commit ml/saved_model/ to your repository.")
    print(f"{'=' * 60}\n")


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH
    if not os.path.exists(csv):
        print(f"\n  ERROR: CSV not found: {csv}")
        print(f"  Usage: python train_and_save.py [path/to/csv]")
        sys.exit(1)
    run(csv)