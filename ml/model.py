import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.ensemble        import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics         import (accuracy_score, classification_report,
                                     confusion_matrix, mean_squared_error,
                                     mean_absolute_error, r2_score)
from sklearn.cluster         import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition   import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi"       : 120,
    "axes.titlesize"   : 12,
    "axes.labelsize"   : 10,
    "figure.facecolor" : "white",
})

STATIC_IMG = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "static", "images"
)
os.makedirs(STATIC_IMG, exist_ok=True)

# ── Global state ──────────────────────────────────────────
_state: dict = {}


def _save(name: str, fig) -> str:
    path = os.path.join(STATIC_IMG, name)
    fig.savefig(path, bbox_inches="tight", dpi=110)
    plt.close(fig)
    return f"images/{name}"


def get_state() -> dict:
    return _state


# ─────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────
def train_pipeline(csv_path: str) -> dict:
    global _state

    # ── 1. Load ───────────────────────────────────────────
    df = pd.read_csv(csv_path)
    print(f"\n✅ Dataset loaded — Shape: {df.shape}")

    # ── 2. Clean ──────────────────────────────────────────
    if df.isnull().sum().sum() > 0:
        df.fillna(df.median(numeric_only=True), inplace=True)
        print("⚠️  Missing values imputed with medians.")
    else:
        print("✅ No missing values found.")

    # ── 3. Encode categoricals ────────────────────────────
    # CRITICAL FIX: create a FRESH LabelEncoder for EACH column
    encoders = {}
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    for col in cat_cols:
        if col != "Irrigation_Need":
            col_enc = LabelEncoder()                      # new instance per column
            df[col + "_enc"] = col_enc.fit_transform(df[col])
            encoders[col] = col_enc
            print(f"  Encoded: {col:30s} → classes: {list(col_enc.classes_)}")

    target_map     = {"Low": 0, "Medium": 1, "High": 2}
    target_map_inv = {0: "Low", 1: "Medium", 2: "High"}
    df["Irrigation_Need_enc"] = df["Irrigation_Need"].map(target_map)
    print(f"\n  Target map: {target_map}")

    # ── 4. Regression proxy target ─────────────────────────
    water_map = {"Low": 1, "Medium": 2, "High": 3}
    df["Water_Need_mm"] = (
        df["Irrigation_Need"].map(water_map) * 15
        + df["Temperature_C"] * 0.5
        - df["Rainfall_mm"]   * 0.3
        - df["Soil_Moisture"]  * 0.2
        + np.random.normal(0, 3, len(df))
    ).clip(lower=5)

    # ── 5. Feature sets ───────────────────────────────────
    NUMERIC_FEATS = [
        "Soil_pH", "Soil_Moisture", "Organic_Carbon",
        "Electrical_Conductivity", "Temperature_C", "Humidity",
        "Rainfall_mm", "Sunlight_Hours", "Wind_Speed_kmh",
        "Field_Area_hectare", "Previous_Irrigation_mm",
    ]
    ENCODED_FEATS = [
        c for c in df.columns
        if c.endswith("_enc") and c != "Irrigation_Need_enc"
    ]
    ALL_FEATURES = NUMERIC_FEATS + ENCODED_FEATS
    print(f"\n  Features ({len(ALL_FEATURES)}): {ALL_FEATURES}")

    # ── 6. Classification ─────────────────────────────────
    X_clf = df[ALL_FEATURES]
    y_clf = df["Irrigation_Need_enc"]

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train_c, y_train_c)
    y_pred_c = clf.predict(X_test_c)
    acc_base  = float(accuracy_score(y_test_c, y_pred_c))

    cv_scores = cross_val_score(clf, X_clf, y_clf, cv=5,
                                scoring="accuracy", n_jobs=-1)

    report = classification_report(
        y_test_c, y_pred_c,
        target_names=["Low", "Medium", "High"],
        output_dict=True,
    )
    print(f"\n  Baseline accuracy: {acc_base:.4f}")
    print(f"  CV mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── 7. Regression ─────────────────────────────────────
    X_reg = df[ALL_FEATURES]
    y_reg = df["Water_Need_mm"]

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    reg.fit(X_train_r, y_train_r)
    y_pred_r = reg.predict(X_test_r)

    rmse = float(np.sqrt(mean_squared_error(y_test_r, y_pred_r)))
    mae  = float(mean_absolute_error(y_test_r, y_pred_r))
    r2   = float(r2_score(y_test_r, y_pred_r))
    print(f"\n  RMSE: {rmse:.4f}  MAE: {mae:.4f}  R²: {r2:.4f}")

    # ── 8. Hyperparameter Tuning ──────────────────────────
    param_grid = {
        "n_estimators"     : [50, 100, 200],
        "max_depth"        : [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "max_features"     : ["sqrt", "log2"],
    }
    X_samp, _, y_samp, _ = train_test_split(
        X_clf, y_clf, train_size=0.20, random_state=42, stratify=y_clf
    )
    grid_search = GridSearchCV(
        estimator  = RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid = param_grid,
        cv=5, scoring="accuracy", n_jobs=-1, verbose=0,
    )
    grid_search.fit(X_samp, y_samp)

    best_clf = RandomForestClassifier(
        **grid_search.best_params_, random_state=42, n_jobs=-1
    )
    best_clf.fit(X_train_c, y_train_c)
    y_pred_best = best_clf.predict(X_test_c)
    acc_tuned   = float(accuracy_score(y_test_c, y_pred_best))
    print(f"\n  Best params: {grid_search.best_params_}")
    print(f"  Tuned accuracy: {acc_tuned:.4f}")

    # ── 9. Clustering ─────────────────────────────────────
    CLUSTER_FEATS = [
        "Soil_Moisture", "Temperature_C", "Humidity",
        "Rainfall_mm", "Soil_pH", "Previous_Irrigation_mm",
        "Field_Area_hectare", "Wind_Speed_kmh",
    ]
    scaler  = StandardScaler()
    X_clust = scaler.fit_transform(df[CLUSTER_FEATS])

    np.random.seed(42)
    idx  = np.random.choice(len(X_clust), size=3000, replace=False)
    X_s  = X_clust[idx]
    df_s = df.iloc[idx].copy()

    # Elbow
    inertias = []
    k_range  = range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_s)
        inertias.append(km.inertia_)

    # K-Means k=3
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_s["KMeans_Cluster"] = kmeans.fit_predict(X_s)

    cluster_labels = {
        0: "High Water Demand",
        1: "Rain-Fed Zone",
        2: "Dry Zone",
    }
    df_s["KMeans_Label"] = df_s["KMeans_Cluster"].map(cluster_labels)

    # DBSCAN
    dbscan = DBSCAN(eps=1.5, min_samples=15, n_jobs=-1)
    df_s["DBSCAN_Cluster"] = dbscan.fit_predict(X_s)
    n_clusters_db = len(set(df_s["DBSCAN_Cluster"])) - (
        1 if -1 in df_s["DBSCAN_Cluster"].values else 0
    )
    n_noise = int((df_s["DBSCAN_Cluster"] == -1).sum())

    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
    df_s["Agglomerative_Cluster"] = agg.fit_predict(X_s)

    # PCA 2D
    pca   = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_s)

    # ── 10. Save all plots ────────────────────────────────
    plots = {}

    # EDA distributions
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold")
    num_cols_eda = [
        "Soil_Moisture", "Temperature_C", "Humidity",
        "Rainfall_mm", "Soil_pH", "Previous_Irrigation_mm",
    ]
    for ax, col in zip(axes.flat, num_cols_eda):
        sns.histplot(df[col], kde=True, ax=ax, color="#3B82F6", bins=35)
        ax.set_title(col.replace("_", " "))
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plots["eda_distributions"] = _save("eda_distributions.png", fig)

    # Target distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    order  = ["Low", "Medium", "High"]
    colors = ["#93C5FD", "#3B82F6", "#1D4ED8"]
    cnts   = [df["Irrigation_Need"].value_counts().get(o, 0) for o in order]
    bars   = ax.bar(order, cnts, color=colors, edgecolor="white", linewidth=1.5,
                    width=0.55)
    for b, c in zip(bars, cnts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 50,
                f"{c:,}", ha="center", fontsize=10, fontweight="bold")
    ax.set_title("Target Class Distribution – Irrigation Need",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plots["target_distribution"] = _save("target_distribution.png", fig)

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 9))
    corr = df.select_dtypes(include=np.number).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", linewidths=0.4, ax=ax)
    ax.set_title("Correlation Matrix – Numeric Features",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plots["correlation_heatmap"] = _save("correlation_heatmap.png", fig)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(7, 5))
    cm_arr = confusion_matrix(y_test_c, y_pred_best)
    sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low", "Medium", "High"],
                yticklabels=["Low", "Medium", "High"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (Tuned Acc = {acc_tuned:.3f})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plots["confusion_matrix"] = _save("confusion_matrix.png", fig)

    # Feature importance — classification
    imp_c = pd.Series(
        best_clf.feature_importances_, index=ALL_FEATURES
    ).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    imp_c.plot(kind="barh", ax=ax, color="#3B82F6")
    ax.set_title("Feature Importance – Classification",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plots["feat_imp_clf"] = _save("feat_imp_clf.png", fig)

    # Regression results
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Water Usage Regression Results",
                 fontsize=13, fontweight="bold")
    axes[0].scatter(y_test_r.values[:800], y_pred_r[:800],
                    alpha=0.3, color="teal", s=10)
    lims = [y_test_r.min(), y_test_r.max()]
    axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    axes[0].set_xlabel("Actual Water_Need_mm")
    axes[0].set_ylabel("Predicted Water_Need_mm")
    axes[0].set_title(f"Actual vs Predicted (R²={r2:.3f})")
    axes[0].legend()
    residuals = y_test_r.values - y_pred_r
    axes[1].hist(residuals, bins=50, color="#F87171", edgecolor="white")
    axes[1].axvline(0, color="black", linestyle="--")
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    plt.tight_layout()
    plots["regression_results"] = _save("regression_results.png", fig)

    # Feature importance — regression
    imp_r = pd.Series(
        reg.feature_importances_, index=ALL_FEATURES
    ).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    imp_r.plot(kind="barh", ax=ax, color="#F97316")
    ax.set_title("Feature Importance – Regression",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plots["feat_imp_reg"] = _save("feat_imp_reg.png", fig)

    # GridSearch heatmap
    results_df = pd.DataFrame(grid_search.cv_results_)
    pivot = results_df.pivot_table(
        values="mean_test_score",
        index="param_max_depth",
        columns="param_n_estimators",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
    ax.set_title("GridSearchCV – Mean Accuracy: n_estimators vs max_depth",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plots["gridsearch_heatmap"] = _save("gridsearch_heatmap.png", fig)

    # Overfitting analysis
    n_est_range = [10, 20, 50, 100, 150, 200, 300]
    tr_sc, te_sc = [], []
    for n in n_est_range:
        tmp = RandomForestClassifier(
            n_estimators=n,
            **{k: v for k, v in grid_search.best_params_.items()
               if k != "n_estimators"},
            random_state=42, n_jobs=-1,
        )
        tmp.fit(X_train_c, y_train_c)
        tr_sc.append(accuracy_score(y_train_c, tmp.predict(X_train_c)))
        te_sc.append(accuracy_score(y_test_c,  tmp.predict(X_test_c)))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(n_est_range, tr_sc, "o-", label="Train", color="#3B82F6")
    ax.plot(n_est_range, te_sc, "s-", label="Test",  color="#F97316")
    ax.set_xlabel("Number of Estimators")
    ax.set_ylabel("Accuracy")
    ax.set_title("Overfitting Analysis – n_estimators vs Accuracy",
                 fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plots["overfitting_analysis"] = _save("overfitting_analysis.png", fig)

    # Elbow
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(k_range), inertias, "o-", color="#3B82F6", markersize=8)
    ax.axvline(3, color="red", linestyle="--", alpha=0.7, label="Chosen k=3")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("K-Means Elbow Method", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plots["kmeans_elbow"] = _save("kmeans_elbow.png", fig)

    # Dendrogram
    Z = linkage(X_s[:200], method="ward")
    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(Z, ax=ax, truncate_mode="level", p=5,
               leaf_font_size=8,
               color_threshold=0.7 * max(Z[:, 2]))
    ax.set_title("Hierarchical Clustering – Dendrogram (Ward)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    plots["dendrogram"] = _save("dendrogram.png", fig)

    # Cluster PCA
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Cluster Visualisation (PCA 2-D Projection)",
                 fontsize=13, fontweight="bold")
    for ax, (col, title) in zip(axes, [
        ("KMeans_Cluster",        "K-Means (k=3)"),
        ("DBSCAN_Cluster",        "DBSCAN (eps=1.5)"),
        ("Agglomerative_Cluster", "Agglomerative (Ward, k=3)"),
    ]):
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                        c=df_s[col], cmap="tab10",
                        alpha=0.5, s=8, linewidths=0)
        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        plt.colorbar(sc, ax=ax, label="Cluster")
    plt.tight_layout()
    plots["clustering_pca"] = _save("clustering_pca.png", fig)

    # Radar chart
    cluster_means      = df_s.groupby("KMeans_Cluster")[CLUSTER_FEATS[:6]].mean()
    mn_r, mx_r         = cluster_means.min(), cluster_means.max()
    cluster_means_norm  = (cluster_means - mn_r) / (mx_r - mn_r + 1e-9)
    labels_r            = [c.replace("_", " ") for c in CLUSTER_FEATS[:6]]
    angles              = np.linspace(0, 2 * np.pi, len(labels_r),
                                      endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    colors_r = ["#3B82F6", "#10B981", "#F97316"]
    cnames_r = list(cluster_labels.values())
    for row_v, color, name in zip(cluster_means_norm.values, colors_r, cnames_r):
        vals = row_v.tolist() + row_v[:1].tolist()
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=name)
        ax.fill(angles, vals, alpha=0.15, color=color)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels_r)
    ax.set_title("K-Means Cluster Profiles – Radar Chart",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.35, -0.05), fontsize=9)
    plt.tight_layout()
    plots["cluster_radar"] = _save("cluster_radar.png", fig)

    # ── 11. Collect state ─────────────────────────────────
    cluster_dist = df_s["KMeans_Label"].value_counts().to_dict()

    _state = {
        # plots
        "plots"         : plots,
        # dataset
        "n_rows"        : int(df.shape[0]),
        "n_cols"        : int(df.shape[1]),
        "n_features"    : len(ALL_FEATURES),
        # classification
        "acc_base"      : round(acc_base, 4),
        "acc_tuned"     : round(acc_tuned, 4),
        "cv_mean"       : round(float(cv_scores.mean()), 4),
        "cv_std"        : round(float(cv_scores.std()), 4),
        "cv_scores"     : [round(float(s), 4) for s in cv_scores],
        "report"        : report,
        # regression
        "rmse"          : round(rmse, 4),
        "mae"           : round(mae, 4),
        "r2"            : round(r2, 4),
        # tuning
        "best_params"   : grid_search.best_params_,
        "best_cv_acc"   : round(float(grid_search.best_score_), 4),
        # clustering
        "n_clusters_db" : n_clusters_db,
        "n_noise"       : n_noise,
        "cluster_dist"  : cluster_dist,
        # target
        "target_counts" : {
            k: int(df["Irrigation_Need"].value_counts().get(k, 0))
            for k in ["Low", "Medium", "High"]
        },
        # models & encoders
        "clf"           : best_clf,
        "reg"           : reg,
        "kmeans_model"  : kmeans,
        "scaler"        : scaler,
        "encoders"      : encoders,
        "cat_cols"      : [c for c in cat_cols if c != "Irrigation_Need"],
        "ALL_FEATURES"  : ALL_FEATURES,
        "NUMERIC_FEATS" : NUMERIC_FEATS,
        "target_map_inv": target_map_inv,
        "trained"       : True,
    }
    print("\n✅ Pipeline complete.")
    return _state


# ─────────────────────────────────────────────────────────
# QUICK PREDICT (single row)
# ─────────────────────────────────────────────────────────
def predict_single(form_data: dict) -> dict:
    state = get_state()
    if not state.get("trained"):
        return {"error": "Model not trained yet."}

    clf            = state["clf"]
    reg            = state["reg"]
    encoders       = state["encoders"]
    ALL_FEATURES   = state["ALL_FEATURES"]
    NUMERIC_FEATS  = state["NUMERIC_FEATS"]
    cat_cols       = state["cat_cols"]
    target_map_inv = state["target_map_inv"]

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

    clf_pred  = int(clf.predict(X)[0])
    clf_proba = clf.predict_proba(X)[0].tolist()
    reg_pred  = float(reg.predict(X)[0])

    return {
        "irrigation_need": target_map_inv[clf_pred],
        "proba_low"      : round(clf_proba[0] * 100, 1),
        "proba_medium"   : round(clf_proba[1] * 100, 1),
        "proba_high"     : round(clf_proba[2] * 100, 1),
        "water_need_mm"  : round(reg_pred, 2),
    }