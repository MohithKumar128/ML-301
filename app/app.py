"""ML-301 Machine Learning Capstone — Interactive Model Workbench & Comparison Hub.

Run locally:
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans
from sklearn.metrics import ConfusionMatrixDisplay

# Ensure local imports work whether launched from repo root or app/
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from model_hub import (  # noqa: E402
    CLASSIFICATION_MODELS,
    CLUSTERING_MODELS,
    MODELS_DIR,
    REGRESSION_MODELS,
    build_and_save_model_bundles,
    load_saved_comparisons,
    prepare_classification_data,
    prepare_clustering_data,
    prepare_regression_data,
    run_classification_model,
    run_clustering_model,
    run_regression_model,
)

st.set_page_config(
    page_title="ML-301 Capstone Workbench",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set consistent Matplotlib / Seaborn scientific theme
sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams.update({
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#CBD5E1",
    "axes.labelcolor": "#0F172A",
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "text.color": "#0F172A",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
})

# Custom CSS for clean scientific workbench typography and tabular numerals
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #0F172A;
        }
        ::selection {
            background: #DBEAFE;
            color: #0F172A;
        }
        h1, h2, h3, h4 {
            font-family: 'IBM Plex Sans', sans-serif;
            letter-spacing: -0.02em;
            color: #0F172A;
        }
        [data-testid="stMetricValue"] {
            font-family: 'IBM Plex Mono', monospace;
            font-variant-numeric: tabular-nums;
            font-size: 1.55rem !important;
            font-weight: 600 !important;
            color: #0F172A !important;
        }
        [data-testid="stMetricDelta"] {
            font-family: 'IBM Plex Mono', monospace;
            font-variant-numeric: tabular-nums;
        }
        [data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        }
        .workbench-banner {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 18px 22px;
            margin-bottom: 18px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }
        .workbench-banner h1 {
            font-size: 1.65rem;
            margin: 0 0 4px 0;
        }
        .workbench-banner p {
            margin: 0;
            color: #334155;
            font-size: 0.95rem;
            max-width: 72ch;
        }
        .spec-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #F1F5F9;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.8rem;
            color: #475569;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached Data & Bundle Loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading preprocessed Regression dataset...")
def get_regression_data():
    return prepare_regression_data()


@st.cache_resource(show_spinner="Loading preprocessed Classification dataset...")
def get_classification_data():
    return prepare_classification_data()


@st.cache_resource(show_spinner="Loading preprocessed Clustering dataset...")
def get_clustering_data():
    return prepare_clustering_data()


@st.cache_data(show_spinner="Loading saved evaluation leaderboards...")
def get_comparisons():
    return load_saved_comparisons()


@st.cache_resource(show_spinner="Loading serialized model bundles from models/...")
def get_serialized_bundles():
    reg_p = MODELS_DIR / "regression_bundle.pkl"
    clf_p = MODELS_DIR / "classification_bundle.pkl"
    clust_p = MODELS_DIR / "clustering_bundle.pkl"
    if not (reg_p.exists() and clf_p.exists() and clust_p.exists()):
        build_and_save_model_bundles()
    return {
        "regression": joblib.load(reg_p),
        "classification": joblib.load(clf_p),
        "clustering": joblib.load(clust_p),
    }


@st.cache_data(show_spinner="Computing K-Means Elbow inertia curve (K = 2..10)...")
def compute_elbow_curve(_X_sample: np.ndarray) -> tuple[list[int], list[float]]:
    ks = list(range(2, 11))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(_X_sample)
        inertias.append(float(km.inertia_))
    return ks, inertias


# ---------------------------------------------------------------------------
# Sidebar Navigation & Track Selection
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 23CSE301 ML Workbench")
    st.caption("End-to-End Model Runner, Comparison Leaderboards & Hosted Inference")

    workspace_mode = st.radio(
        "Workspace View",
        options=[
            "Interactive Model Runner",
            "Consolidated Comparisons",
            "Live Predictor & Hosted Models",
        ],
        index=0,
    )

    st.divider()

    active_track = st.selectbox(
        "Problem Track",
        options=[
            "Classification — Diabetes Risk (10 Models)",
            "Clustering — Bank Customer Segmentation (2 Models)",
            "Regression — Health Insurance Claims (10 Models)",
        ],
        index=0,
    )

    st.divider()
    st.markdown("**Dataset Summary**")
    if active_track.startswith("Classification"):
        st.caption(
            "Dataset: `diabetes_risk_prediction_dataset.csv`\n\n"
            "Records: `50,000` (`40,000` train / `10,000` test)\n\n"
            "Target: `Diabetes_Risk` (`Low`, `Moderate`, `High`)"
        )
    elif active_track.startswith("Clustering"):
        st.caption(
            "Dataset: `bank-additional-full.csv`\n\n"
            "Records: `41,176` deduplicated (`5,000` sample for hierarchy)\n\n"
            "Features: `63` encoded & standardized attributes"
        )
    else:
        st.caption(
            "Dataset: `health_insurance.csv`\n\n"
            "Records: `13,904` deduplicated (`11,123` train / `2,781` test)\n\n"
            "Target: `claim` (Continuous USD)"
        )


# Top Header Banner
st.markdown(
    """
    <div class="workbench-banner">
        <h1>23CSE301 Machine Learning Capstone Workbench</h1>
        <p>Select and execute any of the 22 algorithms across Classification, Clustering, and Regression with custom hyperparameters, inspect diagnostic plots, compare consolidated metrics, and run live predictions against serialized model bundles.</p>
        <div class="spec-strip">
            <span>Classification: 10 Algorithms + Tuned AdaBoost (F1 = 0.9571)</span>
            <span>Clustering: K-Means &amp; Agglomerative Ward (K = 4)</span>
            <span>Regression: 10 Algorithms (Best R² = 0.9636)</span>
            <span>Seed: random_state = 42</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# MODE 1: INTERACTIVE MODEL RUNNER
# ===========================================================================

if workspace_mode == "Interactive Model Runner":
    comparisons = get_comparisons()

    # -----------------------------------------------------------------------
    # 1A. Classification Model Runner
    # -----------------------------------------------------------------------
    if active_track.startswith("Classification"):
        st.subheader("Classification Track — Interactive Model Runner")
        st.write(
            "Choose any of the 10 classification algorithms, configure its hyperparameters (or select the Review 2 Tuned AdaBoost preset), and run it on the held-out test split (`10,000` patients)."
        )

        ctrl_col, result_col = st.columns([1.05, 2.45], gap="large")

        with ctrl_col:
            st.markdown("#### 1. Select Classifier & Parameters")
            selected_clf = st.selectbox(
                "Classification Algorithm",
                options=CLASSIFICATION_MODELS,
                index=CLASSIFICATION_MODELS.index("AdaBoost Classifier"),
            )

            params: dict = {}
            if selected_clf == "AdaBoost Classifier":
                preset = st.radio(
                    "Configuration Preset",
                    options=[
                        "Review 2 Tuned Best (n_estimators=200, lr=1.0)",
                        "Notebook Baseline (n_estimators=100, lr=1.0)",
                        "Custom Hyperparameters",
                    ],
                    index=0,
                )
                if preset.startswith("Review 2 Tuned"):
                    params = {"n_estimators": 200, "learning_rate": 1.0, "base_max_depth": 1}
                    st.info("Preset: `n_estimators = 200`, `learning_rate = 1.0`, `max_depth = 1`")
                elif preset.startswith("Notebook Baseline"):
                    params = {"n_estimators": 100, "learning_rate": 1.0, "base_max_depth": 1}
                    st.info("Preset: `n_estimators = 100`, `learning_rate = 1.0`, `max_depth = 1`")
                else:
                    params["n_estimators"] = st.slider("n_estimators", 50, 250, 150, step=25)
                    params["learning_rate"] = st.select_slider(
                        "learning_rate", options=[0.1, 0.25, 0.5, 0.8, 1.0, 1.2], value=1.0
                    )
                    params["base_max_depth"] = st.slider("Base Tree max_depth", 1, 3, 1)

            elif selected_clf == "Logistic Regression":
                params["C"] = st.select_slider("Regularization C", options=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0], value=1.0)
                params["max_iter"] = st.slider("max_iter", 200, 2000, 1000, step=200)

            elif selected_clf == "K-Nearest Neighbors (KNN)":
                params["n_neighbors"] = st.slider("n_neighbors (k)", 3, 25, 7, step=2)
                params["weights"] = st.selectbox("weights", ["uniform", "distance"], index=0)
                params["metric"] = st.selectbox("metric", ["minkowski", "euclidean", "manhattan"], index=0)

            elif selected_clf == "Gaussian Naive Bayes":
                params["var_smoothing"] = st.select_slider(
                    "var_smoothing", options=[1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6], value=1e-9
                )

            elif selected_clf == "Decision Tree Classifier":
                params["max_depth"] = st.slider("max_depth", 2, 25, 8)
                params["criterion"] = st.selectbox("criterion", ["gini", "entropy"], index=0)

            elif selected_clf == "Support Vector Machine (SVC)":
                params["C"] = st.select_slider("Regularization C", options=[0.1, 0.5, 1.0, 5.0, 10.0], value=1.0)
                params["kernel"] = st.selectbox("kernel", ["rbf", "linear", "poly"], index=0)
                params["train_limit"] = st.slider(
                    "Training sample size (for interactive speed)", 5000, 40000, 12000, step=5000
                )

            elif selected_clf == "Random Forest Classifier":
                params["n_estimators"] = st.slider("n_estimators", 50, 300, 200, step=50)
                max_d = st.slider("max_depth (0 = None)", 0, 30, 0, step=2)
                params["max_depth"] = None if max_d == 0 else max_d

            elif selected_clf == "Gradient Boosting Classifier":
                params["n_estimators"] = st.slider("n_estimators", 50, 200, 100, step=25)
                params["learning_rate"] = st.select_slider("learning_rate", options=[0.05, 0.1, 0.2], value=0.1)
                params["max_depth"] = st.slider("max_depth", 2, 6, 3)

            elif selected_clf == "Bagging Classifier (Decision Tree)":
                params["n_estimators"] = st.slider("n_estimators", 10, 100, 50, step=10)
                params["max_samples"] = st.slider("max_samples", 0.4, 1.0, 0.8, step=0.1)

            elif selected_clf == "MLP Classifier (Neural Network)":
                params["hidden_units"] = st.slider("Hidden Layer Units", 32, 256, 100, step=16)
                params["activation"] = st.selectbox("activation", ["relu", "tanh"], index=0)
                params["max_iter"] = st.slider("max_iter", 50, 200, 100, step=25)

            run_btn = st.button("Run Classifier on Test Split", type="primary", width="stretch")

        with result_col:
            clf_data = get_classification_data()
            clf_bundles = get_serialized_bundles()["classification"]["models"]
            state_key = f"clf_run_{selected_clf}"
            if run_btn or state_key not in st.session_state:
                pretrained = None
                if not run_btn:
                    if selected_clf == "AdaBoost Classifier":
                        pretrained = clf_bundles.get("AdaBoost (Tuned — Best Model)")
                    elif selected_clf == "Gradient Boosting Classifier":
                        pretrained = clf_bundles.get("Gradient Boosting Classifier")
                    elif selected_clf == "Logistic Regression":
                        pretrained = clf_bundles.get("Logistic Regression (Baseline)")
                with st.spinner(f"Evaluating {selected_clf}..."):
                    st.session_state[state_key] = run_classification_model(
                        selected_clf, params, clf_data, pretrained_model=pretrained
                    )

            res = st.session_state[state_key]
            m = res["metrics"]

            # Reference baseline from saved CSV if available
            base_df = comparisons["classification_raw"]
            name_map = {
                "K-Nearest Neighbors (KNN)": "KNN",
                "Decision Tree Classifier": "Decision Tree",
                "Support Vector Machine (SVC)": "SVC",
                "Random Forest Classifier": "Random Forest",
                "AdaBoost Classifier": "AdaBoost",
                "Gradient Boosting Classifier": "Gradient Boosting",
                "Bagging Classifier (Decision Tree)": "Bagging (Decision Tree)",
                "MLP Classifier (Neural Network)": "MLP Classifier",
            }
            csv_name = name_map.get(selected_clf, selected_clf)
            base_row = base_df[base_df["Model"] == csv_name]

            st.markdown(f"#### 2. Evaluation Results — `{selected_clf}`")
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)

            def _delta(metric_col: str, val: float) -> str | None:
                if base_row.empty:
                    return None
                diff = val - float(base_row.iloc[0][metric_col])
                if abs(diff) < 1e-5:
                    return "0.0000 (Baseline)"
                return f"{diff:+.4f} vs baseline"

            mc1.metric("Accuracy", f"{m['Accuracy']:.4f}", _delta("Accuracy", m["Accuracy"]))
            mc2.metric("Weighted Precision", f"{m['Weighted Precision']:.4f}", _delta("Weighted Precision", m["Weighted Precision"]))
            mc3.metric("Weighted Recall", f"{m['Weighted Recall']:.4f}", _delta("Weighted Recall", m["Weighted Recall"]))
            mc4.metric("Weighted F1-score", f"{m['Weighted F1-score']:.4f}", _delta("Weighted F1-score", m["Weighted F1-score"]))
            mc5.metric("OvR ROC-AUC", f"{m['ROC-AUC']:.4f}", _delta("ROC-AUC", m["ROC-AUC"]))

            plot_c1, plot_c2 = st.columns(2)

            with plot_c1:
                fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.5))
                ConfusionMatrixDisplay(
                    confusion_matrix=res["confusion_matrix"],
                    display_labels=res["classes"],
                ).plot(ax=ax_cm, cmap="Blues", values_format="d", colorbar=False)
                ax_cm.set_title(f"{selected_clf} — Confusion Matrix")
                ax_cm.grid(False)
                plt.tight_layout()
                st.pyplot(fig_cm)
                plt.close(fig_cm)

            with plot_c2:
                fig_roc, ax_roc = plt.subplots(figsize=(5.5, 4.5))
                for cls_label, (fpr, tpr) in res["roc_curves"].items():
                    ax_roc.plot(fpr, tpr, linewidth=2, label=f"Class: {cls_label}")
                ax_roc.plot([0, 1], [0, 1], linestyle="--", color="#64748B", linewidth=1, label="Random Chance")
                ax_roc.set_title(f"{selected_clf} — Multi-Class OvR ROC Curves")
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.legend(loc="lower right")
                plt.tight_layout()
                st.pyplot(fig_roc)
                plt.close(fig_roc)

            if res["feature_importance"] is not None:
                st.markdown("#### 3. Top Feature Importances / Coefficients")
                fi_df = res["feature_importance"]
                fig_fi, ax_fi = plt.subplots(figsize=(9, 3.8))
                sns.barplot(
                    data=fi_df,
                    x="Importance",
                    y="Feature",
                    hue="Feature",
                    legend=False,
                    palette="Blues_r",
                    ax=ax_fi,
                )
                ax_fi.set_title(f"Top 12 Influential Features — {selected_clf}")
                plt.tight_layout()
                st.pyplot(fig_fi)
                plt.close(fig_fi)

    # -----------------------------------------------------------------------
    # 1B. Clustering Model Runner
    # -----------------------------------------------------------------------
    elif active_track.startswith("Clustering"):
        st.subheader("Clustering Track — Interactive Model Runner")
        st.write(
            "Run **K-Means Clustering** or **Agglomerative Hierarchical Clustering** on the shared preprocessed Bank Marketing feature matrix (`X_processed`), inspect internal validation indices, and explore PCA and hierarchical visualizations."
        )

        ctrl_col, result_col = st.columns([1.05, 2.45], gap="large")

        with ctrl_col:
            st.markdown("#### 1. Select Clustering Algorithm")
            selected_clust = st.selectbox(
                "Clustering Algorithm",
                options=CLUSTERING_MODELS,
                index=1,
            )

            c_params: dict = {}
            c_params["n_clusters"] = st.slider("Number of Clusters (K)", 2, 8, 4)

            if selected_clust == "K-Means Clustering":
                c_params["n_init"] = st.slider("n_init (Centroid Initializations)", 5, 25, 10, step=5)
            else:
                c_params["linkage"] = st.selectbox(
                    "Linkage Strategy",
                    options=["ward", "complete", "average", "single"],
                    index=0,
                )
                c_params["p_leaves"] = st.slider("Dendrogram Truncated Leaves (p)", 15, 50, 30, step=5)

            run_clust_btn = st.button("Run Clustering Model", type="primary", width="stretch")

        with result_col:
            clust_data = get_clustering_data()
            c_state_key = f"clust_run_{selected_clust}_{c_params.get('n_clusters')}_{c_params.get('linkage', 'km')}"
            if run_clust_btn or c_state_key not in st.session_state:
                with st.spinner(f"Fitting {selected_clust}..."):
                    st.session_state[c_state_key] = run_clustering_model(selected_clust, c_params, clust_data)

            c_res = st.session_state[c_state_key]
            cm_metrics = c_res["metrics"]

            st.markdown(f"#### 2. Internal Evaluation Metrics — `{selected_clust} (K = {c_res['n_clusters']})`")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Silhouette Score", f"{cm_metrics['Silhouette Score']:.4f}", "Higher is better")
            k2.metric("Davies–Bouldin Index", f"{cm_metrics['Davies-Bouldin Index']:.4f}", "Lower is better")
            k3.metric("Calinski–Harabasz Index", f"{cm_metrics['Calinski-Harabasz Index']:.2f}", "Higher is better")
            k4.metric("Evaluated Customers", f"{len(c_res['labels']):,}", f"K = {c_res['n_clusters']} clusters")

            p_col1, p_col2 = st.columns([1.35, 1.0])

            with p_col1:
                fig_pca, ax_pca = plt.subplots(figsize=(6.5, 4.6))
                coords = c_res["pca_coords"]
                labels = c_res["labels"]
                palette = sns.color_palette("colorblind", n_colors=c_res["n_clusters"])
                for cid in range(c_res["n_clusters"]):
                    mask = labels == cid
                    ax_pca.scatter(
                        coords[mask, 0],
                        coords[mask, 1],
                        s=14,
                        alpha=0.65,
                        color=palette[cid],
                        label=f"Cluster {cid} (n={int(mask.sum())})",
                    )
                ax_pca.set_title(f"2D PCA Cluster Projection — {selected_clust}")
                ax_pca.set_xlabel("Principal Component 1")
                ax_pca.set_ylabel("Principal Component 2")
                ax_pca.legend(loc="best", fontsize=8.5)
                plt.tight_layout()
                st.pyplot(fig_pca)
                plt.close(fig_pca)

            with p_col2:
                st.markdown("**Cluster Size Breakdown**")
                st.dataframe(c_res["cluster_counts"], width="stretch", hide_index=True)

                fig_sz, ax_sz = plt.subplots(figsize=(4.8, 2.8))
                sns.barplot(
                    data=c_res["cluster_counts"],
                    x="Cluster",
                    y="Customers",
                    hue="Cluster",
                    legend=False,
                    palette="colorblind",
                    ax=ax_sz,
                )
                ax_sz.set_title("Customers per Cluster")
                plt.tight_layout()
                st.pyplot(fig_sz)
                plt.close(fig_sz)

            st.markdown("#### 3. Structural Diagnostic Plot")
            if selected_clust == "Agglomerative Hierarchical Clustering" and c_res["linkage_matrix"] is not None:
                Z = c_res["linkage_matrix"]
                k_cut = c_res["n_clusters"]
                cut_dist = (Z[-(k_cut - 1), 2] + Z[-k_cut, 2]) / 2.0 if k_cut > 1 else Z[-1, 2]

                fig_den, ax_den = plt.subplots(figsize=(10.5, 4.5))
                dendrogram(
                    Z,
                    truncate_mode="lastp",
                    p=int(c_params.get("p_leaves", 30)),
                    leaf_rotation=45,
                    leaf_font_size=9,
                    show_contracted=True,
                    color_threshold=cut_dist,
                    ax=ax_den,
                )
                ax_den.axhline(
                    y=cut_dist,
                    color="#DC2626",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"K = {k_cut} Cut Threshold ({cut_dist:.1f})",
                )
                ax_den.set_title(
                    f"Truncated Hierarchical Dendrogram ({c_params.get('linkage', 'ward').capitalize()} Linkage, n = 5,000)"
                )
                ax_den.set_xlabel("Cluster Size (in parentheses) or Sample Index")
                ax_den.set_ylabel("Linkage Distance")
                ax_den.legend(loc="upper right")
                plt.tight_layout()
                st.pyplot(fig_den)
                plt.close(fig_den)
            else:
                ks, inertias = compute_elbow_curve(clust_data["X_sample"])
                fig_elb, ax_elb = plt.subplots(figsize=(9.5, 4.0))
                ax_elb.plot(ks, inertias, marker="o", linewidth=2, color="#0F52BA")
                ax_elb.axvline(
                    x=c_res["n_clusters"],
                    color="#DC2626",
                    linestyle="--",
                    label=f"Selected K = {c_res['n_clusters']}",
                )
                ax_elb.set_title("K-Means Elbow Curve (Within-Cluster Sum of Squares vs. K)")
                ax_elb.set_xlabel("Number of Clusters (K)")
                ax_elb.set_ylabel("Inertia")
                ax_elb.set_xticks(ks)
                ax_elb.legend(loc="upper right")
                plt.tight_layout()
                st.pyplot(fig_elb)
                plt.close(fig_elb)

    # -----------------------------------------------------------------------
    # 1C. Regression Model Runner
    # -----------------------------------------------------------------------
    else:
        st.subheader("Regression Track — Interactive Model Runner")
        st.write(
            "Select any of the 10 regression algorithms, customize hyperparameters, and evaluate predictions on the held-out Health Insurance test split (`2,781` policyholders)."
        )

        ctrl_col, result_col = st.columns([1.05, 2.45], gap="large")

        with ctrl_col:
            st.markdown("#### 1. Select Regressor & Parameters")
            selected_reg = st.selectbox(
                "Regression Algorithm",
                options=REGRESSION_MODELS,
                index=REGRESSION_MODELS.index("Random Forest Regressor"),
            )

            r_params: dict = {}
            if selected_reg == "Ridge Regression":
                r_params["alpha"] = st.select_slider("L2 Penalty (alpha)", options=[0.01, 0.1, 1.0, 10.0, 50.0, 100.0], value=1.0)
            elif selected_reg == "Lasso Regression":
                r_params["alpha"] = st.select_slider("L1 Penalty (alpha)", options=[0.1, 1.0, 5.0, 10.0, 25.0, 50.0], value=10.0)
            elif selected_reg == "ElasticNet Regression":
                r_params["alpha"] = st.select_slider("Penalty (alpha)", options=[0.01, 0.05, 0.1, 0.5, 1.0], value=0.1)
                r_params["l1_ratio"] = st.slider("l1_ratio", 0.1, 0.9, 0.5, step=0.1)
            elif selected_reg == "Polynomial Regression":
                r_params["degree"] = st.slider("Polynomial Degree", 2, 3, 2)
            elif selected_reg == "Decision Tree Regressor":
                r_params["max_depth"] = st.slider("max_depth", 3, 25, 12)
                r_params["min_samples_split"] = st.slider("min_samples_split", 2, 20, 5)
            elif selected_reg == "Random Forest Regressor":
                r_params["n_estimators"] = st.slider("n_estimators", 50, 300, 200, step=50)
                r_params["max_depth"] = st.slider("max_depth", 5, 30, 18)
            elif selected_reg == "Gradient Boosting Regressor":
                r_params["n_estimators"] = st.slider("n_estimators", 50, 300, 150, step=50)
                r_params["learning_rate"] = st.select_slider("learning_rate", options=[0.05, 0.1, 0.2], value=0.1)
                r_params["max_depth"] = st.slider("max_depth", 2, 6, 3)
            elif selected_reg == "Support Vector Regressor (SVR)":
                r_params["C"] = st.select_slider("Regularization C", options=[1.0, 10.0, 50.0, 100.0, 500.0], value=100.0)
                r_params["kernel"] = st.selectbox("kernel", ["rbf", "linear"], index=0)
            elif selected_reg == "KNN Regressor":
                r_params["n_neighbors"] = st.slider("n_neighbors (k)", 3, 21, 5, step=2)
                r_params["weights"] = st.selectbox("weights", ["distance", "uniform"], index=0)
            else:
                st.info("Ordinary Least Squares baseline (`LinearRegression`) has no regularization hyperparameters.")

            run_reg_btn = st.button("Run Regressor on Test Split", type="primary", width="stretch")

        with result_col:
            reg_data = get_regression_data()
            reg_bundles = get_serialized_bundles()["regression"]["models"]
            r_state_key = f"reg_run_{selected_reg}"
            if run_reg_btn or r_state_key not in st.session_state:
                pretrained_reg = None
                if not run_reg_btn:
                    if selected_reg == "Random Forest Regressor":
                        pretrained_reg = reg_bundles.get("Random Forest Regressor (Best)")
                    elif selected_reg == "Gradient Boosting Regressor":
                        pretrained_reg = reg_bundles.get("Gradient Boosting Regressor")
                    elif selected_reg == "Linear Regression":
                        pretrained_reg = reg_bundles.get("Linear Regression (Baseline)")
                with st.spinner(f"Evaluating {selected_reg}..."):
                    st.session_state[r_state_key] = run_regression_model(
                        selected_reg, r_params, reg_data, pretrained_model=pretrained_reg
                    )

            r_res = st.session_state[r_state_key]
            rm = r_res["metrics"]

            st.markdown(f"#### 2. Evaluation Results — `{r_res['model_name']}`")
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("R² Score (Test Set)", f"{rm['R2']:.4f}")
            rc2.metric("RMSE (USD)", f"${rm['RMSE']:,.2f}")
            rc3.metric("MAE (USD)", f"${rm['MAE']:,.2f}")

            y_true = r_res["y_test"]
            y_hat = r_res["y_pred"]
            residuals = y_true - y_hat

            rp1, rp2 = st.columns(2)
            with rp1:
                fig_pa, ax_pa = plt.subplots(figsize=(5.5, 4.4))
                ax_pa.scatter(y_true, y_hat, alpha=0.35, s=16, color="#0F52BA")
                lims = [min(y_true.min(), y_hat.min()), max(y_true.max(), y_hat.max())]
                ax_pa.plot(lims, lims, "--", color="#DC2626", linewidth=1.5, label="Ideal Fit (y = x)")
                ax_pa.set_title("Predicted vs. Actual Claims")
                ax_pa.set_xlabel("Actual Claim (USD)")
                ax_pa.set_ylabel("Predicted Claim (USD)")
                ax_pa.legend(loc="upper left")
                plt.tight_layout()
                st.pyplot(fig_pa)
                plt.close(fig_pa)

            with rp2:
                fig_res, ax_res = plt.subplots(figsize=(5.5, 4.4))
                ax_res.scatter(y_hat, residuals, alpha=0.35, s=16, color="#0D9488")
                ax_res.axhline(0, linestyle="--", color="#DC2626", linewidth=1.5)
                ax_res.set_title("Residuals vs. Predicted Claims")
                ax_res.set_xlabel("Predicted Claim (USD)")
                ax_res.set_ylabel("Residual (Actual - Predicted)")
                plt.tight_layout()
                st.pyplot(fig_res)
                plt.close(fig_res)

            if r_res["feature_importance"] is not None:
                st.markdown("#### 3. Top Feature Importances / Coefficients")
                fig_rfi, ax_rfi = plt.subplots(figsize=(9, 3.8))
                sns.barplot(
                    data=r_res["feature_importance"],
                    x="Importance",
                    y="Feature",
                    hue="Feature",
                    legend=False,
                    palette="Blues_r",
                    ax=ax_rfi,
                )
                ax_rfi.set_title(f"Top 12 Predictors — {r_res['model_name']}")
                plt.tight_layout()
                st.pyplot(fig_rfi)
                plt.close(fig_rfi)


# ===========================================================================
# MODE 2: CONSOLIDATED COMPARISONS
# ===========================================================================

elif workspace_mode == "Consolidated Comparisons":
    comparisons = get_comparisons()

    tab_clf, tab_clust, tab_reg = st.tabs([
        "Classification (10 Models + A3 Tuning)",
        "Clustering (K-Means vs. Agglomerative & Linkages)",
        "Regression (10 Models)",
    ])

    # -----------------------------------------------------------------------
    # Tab 1: Classification Consolidated Comparison + A3 Tuning
    # -----------------------------------------------------------------------
    with tab_clf:
        st.subheader("Consolidated 10-Algorithm Classification Comparison")
        clf_df = comparisons["classification"]

        top_clf = clf_df.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Baseline Classifier", str(top_clf["Model"]))
        c2.metric("Baseline Weighted F1", f"{top_clf['Weighted F1-score']:.4f}")
        c3.metric("Tuned AdaBoost Weighted F1", "0.9571", "+0.0022 improvement")
        c4.metric("Tuned AdaBoost ROC-AUC", "0.9877", "+0.0009 improvement")

        sort_metric = st.selectbox(
            "Rank and Plot Classification Models by Metric",
            options=["Weighted F1-score", "Accuracy", "ROC-AUC", "Weighted Precision", "Weighted Recall"],
            index=0,
        )

        sorted_clf = clf_df.sort_values(by=sort_metric, ascending=False).reset_index(drop=True)
        sorted_clf.index = sorted_clf.index + 1
        sorted_clf.index.name = "Rank"

        col_tbl, col_chart = st.columns([1.25, 1.15], gap="large")
        with col_tbl:
            st.markdown("**10-Model Consolidated Evaluation Table**")
            st.dataframe(
                sorted_clf.style.format({
                    "Accuracy": "{:.4f}",
                    "Weighted Precision": "{:.4f}",
                    "Weighted Recall": "{:.4f}",
                    "Weighted F1-score": "{:.4f}",
                    "ROC-AUC": "{:.4f}",
                }).highlight_max(
                    subset=["Accuracy", "Weighted Precision", "Weighted Recall", "Weighted F1-score", "ROC-AUC"],
                    color="#DBEAFE",
                ),
                width="stretch",
            )
            st.download_button(
                "Download Classification Comparison CSV",
                data=sorted_clf.to_csv(index=True).encode("utf-8"),
                file_name="classification_comparison.csv",
                mime="text/csv",
            )

        with col_chart:
            fig_c, ax_c = plt.subplots(figsize=(6.2, 4.6))
            plot_df = sorted_clf.sort_values(by=sort_metric, ascending=True)
            y_pos = np.arange(len(plot_df))
            ax_c.barh(y_pos - 0.18, plot_df[sort_metric], height=0.35, label=sort_metric, color="#0F52BA")
            ax_c.barh(y_pos + 0.18, plot_df["ROC-AUC"], height=0.35, label="ROC-AUC", color="#0D9488")
            ax_c.set_yticks(y_pos)
            ax_c.set_yticklabels(plot_df["Model"])
            ax_c.set_xlim(0.70, 1.01)
            ax_c.set_xlabel("Score")
            ax_c.set_title(f"10 Classifiers Ranked by {sort_metric}")
            ax_c.legend(loc="lower right")
            plt.tight_layout()
            st.pyplot(fig_c)
            plt.close(fig_c)

        st.divider()
        st.subheader("Section A3 — Best Model Hyperparameter Tuning Comparison (AdaBoost)")
        tuning_df = comparisons["classification_tuning"]
        if not tuning_df.empty:
            t_col1, t_col2 = st.columns([1.25, 1.15], gap="large")
            with t_col1:
                st.dataframe(
                    tuning_df.style.format({
                        "Accuracy": "{:.4f}",
                        "Weighted Precision": "{:.4f}",
                        "Weighted Recall": "{:.4f}",
                        "Weighted F1-score": "{:.4f}",
                        "ROC-AUC": "{:.4f}",
                    }),
                    width="stretch",
                    hide_index=True,
                )
                st.markdown(
                    """
                    **Justification & Tuning Summary:**
                    - **Selected Model:** `AdaBoost Classifier` ranked #1 across all five evaluation metrics in the 10-model comparison.
                    - **Tuning Method:** 3-fold Stratified `GridSearchCV` over `n_estimators` (`[50, 100, 150, 200]`) and `learning_rate` (`[0.5, 1.0]`).
                    - **Result:** Increasing `n_estimators` from `100` to `200` improved every test-set metric (`Accuracy: 0.9589 -> 0.9610`, `Weighted F1: 0.9549 -> 0.9571`, `ROC-AUC: 0.9868 -> 0.9877`).
                    """
                )
            with t_col2:
                m_cols = ["Accuracy", "Weighted Precision", "Weighted Recall", "Weighted F1-score", "ROC-AUC"]
                b_vals = tuning_df.iloc[0][m_cols].to_numpy(dtype=float)
                t_vals = tuning_df.iloc[1][m_cols].to_numpy(dtype=float)
                x_i = np.arange(len(m_cols))
                fig_t, ax_t = plt.subplots(figsize=(6.0, 3.8))
                ax_t.bar(x_i - 0.18, b_vals, width=0.35, label="Baseline (n=100)", color="#64748B")
                ax_t.bar(x_i + 0.18, t_vals, width=0.35, label="Tuned (n=200)", color="#0F52BA")
                ax_t.set_ylim(0.94, 0.995)
                ax_t.set_xticks(x_i)
                ax_t.set_xticklabels(m_cols, rotation=20, ha="right")
                ax_t.set_ylabel("Score")
                ax_t.set_title("Baseline vs. Tuned AdaBoost on Held-Out Test Set")
                ax_t.legend(loc="lower right")
                plt.tight_layout()
                st.pyplot(fig_t)
                plt.close(fig_t)

    # -----------------------------------------------------------------------
    # Tab 2: Clustering Comparison
    # -----------------------------------------------------------------------
    with tab_clust:
        st.subheader("Clustering Track — K-Means vs. Agglomerative & Linkage Comparison")
        clust_comp = comparisons["clustering_comparison"]
        linkage_df = comparisons["clustering_linkage"]

        if not clust_comp.empty:
            st.markdown("#### 1. K-Means vs. Agglomerative Hierarchical Clustering ($K = 4$, $n = 5,000$)")
            st.dataframe(
                clust_comp.style.format({
                    "Silhouette Score": "{:.4f}",
                    "Davies-Bouldin Index": "{:.4f}",
                    "Calinski-Harabasz Index": "{:.2f}",
                }),
                width="stretch",
                hide_index=True,
            )

            # Side-by-side 2D PCA comparison from saved bundle
            cb = get_serialized_bundles()["clustering"]
            coords = cb["X_sample_pca"]
            km_lbl = cb["kmeans_sample_labels"]
            ag_lbl = cb["agg_sample_labels"]

            fig_dual, axes_dual = plt.subplots(1, 2, figsize=(11.5, 4.4))
            pal = sns.color_palette("colorblind", 4)
            for cid in range(4):
                m_km = km_lbl == cid
                axes_dual[0].scatter(coords[m_km, 0], coords[m_km, 1], s=12, alpha=0.6, color=pal[cid], label=f"Cluster {cid}")
                m_ag = ag_lbl == cid
                axes_dual[1].scatter(coords[m_ag, 0], coords[m_ag, 1], s=12, alpha=0.6, color=pal[cid], label=f"Cluster {cid}")

            axes_dual[0].set_title("K-Means Clustering (K = 4) — 2D PCA")
            axes_dual[0].set_xlabel("PC1")
            axes_dual[0].set_ylabel("PC2")
            axes_dual[0].legend(loc="best", fontsize=8)

            axes_dual[1].set_title("Agglomerative Clustering (Ward, K = 4) — 2D PCA")
            axes_dual[1].set_xlabel("PC1")
            axes_dual[1].set_ylabel("PC2")
            axes_dual[1].legend(loc="best", fontsize=8)

            plt.tight_layout()
            st.pyplot(fig_dual)
            plt.close(fig_dual)

        if not linkage_df.empty:
            st.divider()
            st.markdown("#### 2. Agglomerative Linkage Strategy Comparison (`Ward`, `Complete`, `Average`, `Single`)")
            l_col1, l_col2 = st.columns([1.2, 1.2], gap="large")
            with l_col1:
                st.dataframe(
                    linkage_df.style.format({
                        "Silhouette_Score": "{:.4f}",
                        "Davies_Bouldin_Index": "{:.4f}",
                        "Calinski_Harabasz_Index": "{:.2f}",
                    }),
                    width="stretch",
                    hide_index=True,
                )
                st.caption(
                    "Note: Complete, Average, and Single linkage place >96% of customers into a single cluster (Smallest_Cluster_Size = 1) due to outlier chaining, whereas Ward linkage achieves a 4.3x higher Calinski–Harabasz Index (828.39) with balanced segments."
                )
            with l_col2:
                fig_l, ax_l = plt.subplots(figsize=(5.8, 3.8))
                sns.barplot(
                    data=linkage_df,
                    x="Linkage",
                    y="Calinski_Harabasz_Index",
                    hue="Linkage",
                    legend=False,
                    palette="colorblind",
                    ax=ax_l,
                )
                ax_l.set_title("Calinski–Harabasz Index by Linkage Strategy")
                ax_l.set_ylabel("Calinski–Harabasz Index")
                plt.tight_layout()
                st.pyplot(fig_l)
                plt.close(fig_l)

    # -----------------------------------------------------------------------
    # Tab 3: Regression 10-Model Comparison
    # -----------------------------------------------------------------------
    with tab_reg:
        st.subheader("Consolidated 10-Algorithm Regression Comparison")
        reg_df = comparisons["regression"]

        top_reg = reg_df.iloc[0]
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Best Regression Model", str(top_reg["Model"]))
        r2.metric("Test R² Score", f"{top_reg['R2']:.4f}")
        r3.metric("Test RMSE", f"${top_reg['RMSE']:,.2f}")
        r4.metric("5-Fold CV R²", f"{top_reg['CV_R2']:.4f}")

        r_tbl, r_chart = st.columns([1.25, 1.15], gap="large")
        with r_tbl:
            st.dataframe(
                reg_df.style.format({
                    "R2": "{:.4f}",
                    "RMSE": "{:,.2f}",
                    "MAE": "{:,.2f}",
                    "CV_R2": "{:.4f}",
                }).highlight_max(subset=["R2", "CV_R2"], color="#DBEAFE")
                .highlight_min(subset=["RMSE", "MAE"], color="#DCFCE7"),
                width="stretch",
            )
            st.download_button(
                "Download Regression Comparison CSV",
                data=reg_df.to_csv(index=True).encode("utf-8"),
                file_name="regression_comparison.csv",
                mime="text/csv",
            )

        with r_chart:
            fig_r, ax_r = plt.subplots(figsize=(6.2, 4.6))
            r_plot = reg_df.sort_values(by="R2", ascending=True)
            sns.barplot(
                data=r_plot,
                x="R2",
                y="Model",
                hue="Model",
                legend=False,
                palette="Blues",
                ax=ax_r,
            )
            ax_r.set_xlim(0.55, 1.0)
            ax_r.set_title("10 Regression Models Ranked by Test R²")
            ax_r.set_xlabel("R² Score")
            plt.tight_layout()
            st.pyplot(fig_r)
            plt.close(fig_r)


# ===========================================================================
# MODE 3: LIVE PREDICTOR & HOSTED MODEL BUNDLES
# ===========================================================================

else:
    bundles = get_serialized_bundles()

    pred_tab1, pred_tab2, pred_tab3, pred_tab4 = st.tabs([
        "Diabetes Risk Predictor (Classification)",
        "Insurance Claim Estimator (Regression)",
        "Customer Segment Assigner (Clustering)",
        "Hosted Model Bundles (models/*.pkl)",
    ])

    # -----------------------------------------------------------------------
    # 3A. Live Diabetes Risk Predictor
    # -----------------------------------------------------------------------
    with pred_tab1:
        st.subheader("Live Patient Diabetes Risk Predictor")
        clf_bundle = bundles["classification"]
        num_defs = clf_bundle["numeric_defaults"]
        cat_opts = clf_bundle["categorical_options"]

        top_bar1, top_bar2 = st.columns([1.4, 1.6])
        with top_bar1:
            chosen_clf_bundle_model = st.selectbox(
                "Hosted Classification Model (`models/classification_bundle.pkl`)",
                options=list(clf_bundle["models"].keys()),
                index=0,
            )
        with top_bar2:
            patient_preset = st.radio(
                "Quick Patient Profile Preset",
                options=["Moderate Risk Adult", "High Risk Clinical Profile", "Low Risk Healthy Profile"],
                horizontal=True,
            )

        if patient_preset == "High Risk Clinical Profile":
            p_defaults = {"Fasting_Blood_Sugar": 175.0, "Postprandial_Blood_Sugar": 245.0, "HbA1c": 8.2, "BMI": 33.5, "Age": 58.0}
        elif patient_preset == "Low Risk Healthy Profile":
            p_defaults = {"Fasting_Blood_Sugar": 84.0, "Postprandial_Blood_Sugar": 112.0, "HbA1c": 4.9, "BMI": 21.5, "Age": 27.0}
        else:
            p_defaults = {"Fasting_Blood_Sugar": 114.0, "Postprandial_Blood_Sugar": 155.0, "HbA1c": 6.0, "BMI": 26.8, "Age": 44.0}

        with st.form("diabetes_live_form"):
            st.markdown("**Clinical & Lifestyle Inputs**")
            f_cols = st.columns(4)
            user_input: dict = {}

            for idx, (col_name, meta) in enumerate(num_defs.items()):
                default_val = p_defaults.get(col_name, meta["median"])
                with f_cols[idx % 4]:
                    user_input[col_name] = st.number_input(
                        col_name.replace("_", " "),
                        min_value=float(meta["min"]),
                        max_value=float(meta["max"]),
                        value=float(np.clip(default_val, meta["min"], meta["max"])),
                    )

            st.markdown("**Categorical Attributes**")
            c_cols = st.columns(4)
            for idx, (col_name, options) in enumerate(cat_opts.items()):
                with c_cols[idx % 4]:
                    user_input[col_name] = st.selectbox(
                        col_name.replace("_", " "),
                        options=options,
                        index=0,
                    )

            submitted_clf = st.form_submit_button("Predict Diabetes Risk", type="primary")

        # Engineer Pulse_Pressure and apply training BMI IQR bounds exactly as in notebook
        user_input["Pulse_Pressure"] = (
            user_input["Blood_Pressure_Systolic"] - user_input["Blood_Pressure_Diastolic"]
        )
        bmi_low, bmi_high = clf_bundle["bmi_bounds"]
        user_input["BMI"] = float(np.clip(user_input["BMI"], bmi_low, bmi_high))

        input_df = pd.DataFrame([user_input])
        X_user_proc = clf_bundle["preprocessor"].transform(input_df)

        active_clf_model = clf_bundle["models"][chosen_clf_bundle_model]
        pred_class = active_clf_model.predict(X_user_proc)[0]
        pred_proba = active_clf_model.predict_proba(X_user_proc)[0]
        classes = active_clf_model.classes_

        res_c1, res_c2 = st.columns([1.1, 1.4], gap="large")
        with res_c1:
            st.metric("Predicted Diabetes Risk Category", str(pred_class))
            st.metric("Engineered Pulse Pressure", f"{user_input['Pulse_Pressure']:.1f} mmHg")
            proba_table = pd.DataFrame({
                "Risk Class": classes,
                "Probability": pred_proba,
                "Percentage": [f"{p * 100:.2f}%" for p in pred_proba],
            })
            st.dataframe(proba_table, width="stretch", hide_index=True)

        with res_c2:
            fig_p, ax_p = plt.subplots(figsize=(5.8, 3.2))
            sns.barplot(
                data=proba_table,
                x="Probability",
                y="Risk Class",
                hue="Risk Class",
                legend=False,
                palette="colorblind",
                ax=ax_p,
            )
            ax_p.set_xlim(0, 1.05)
            ax_p.set_title(f"Predicted Class Probabilities — {chosen_clf_bundle_model}")
            plt.tight_layout()
            st.pyplot(fig_p)
            plt.close(fig_p)

    # -----------------------------------------------------------------------
    # 3B. Live Health Insurance Claim Estimator
    # -----------------------------------------------------------------------
    with pred_tab2:
        st.subheader("Live Health Insurance Claim Estimator")
        reg_bundle = bundles["regression"]
        r_num_defs = reg_bundle["numeric_defaults"]
        r_cat_opts = reg_bundle["categorical_options"]

        with st.form("insurance_live_form"):
            rc_cols = st.columns(4)
            r_input: dict = {}
            for idx, (col_name, meta) in enumerate(r_num_defs.items()):
                with rc_cols[idx % 4]:
                    r_input[col_name] = st.number_input(
                        col_name.replace("_", " ").title(),
                        min_value=float(meta["min"]),
                        max_value=float(meta["max"]),
                        value=float(meta["median"]),
                    )
            for idx, (col_name, options) in enumerate(r_cat_opts.items()):
                with rc_cols[(idx + len(r_num_defs)) % 4]:
                    r_input[col_name] = st.selectbox(
                        col_name.replace("_", " ").title(),
                        options=options,
                        index=0,
                    )
            st.form_submit_button("Estimate Insurance Claim", type="primary")

        r_input["age_bmi_interaction"] = r_input["age"] * r_input["bmi"]
        r_in_df = pd.DataFrame([r_input])
        X_r_proc = reg_bundle["preprocessor"].transform(r_in_df)

        pred_rows = []
        for m_label, m_obj in reg_bundle["models"].items():
            est_val = float(m_obj.predict(X_r_proc)[0])
            pred_rows.append({"Hosted Model": m_label, "Estimated Claim (USD)": est_val})

        pred_reg_df = pd.DataFrame(pred_rows)
        m_col1, m_col2, m_col3 = st.columns(3)
        for col_widget, row_data in zip((m_col1, m_col2, m_col3), pred_rows):
            col_widget.metric(row_data["Hosted Model"], f"${row_data['Estimated Claim (USD)']:,.2f}")

    # -----------------------------------------------------------------------
    # 3C. Live Bank Customer Segment Assigner
    # -----------------------------------------------------------------------
    with pred_tab3:
        st.subheader("Live Bank Customer Segment Assigner")
        clust_bundle = bundles["clustering"]
        c_num_defs = clust_bundle["numeric_defaults"]
        c_cat_opts = clust_bundle["categorical_options"]

        with st.form("clustering_live_form"):
            st.markdown("**Customer Demographic & Campaign Inputs**")
            cc_cols = st.columns(5)
            c_input: dict = {}
            for idx, (col_name, meta) in enumerate(c_num_defs.items()):
                with cc_cols[idx % 5]:
                    c_input[col_name] = st.number_input(
                        col_name,
                        min_value=float(meta["min"]),
                        max_value=float(meta["max"]),
                        value=float(meta["median"]),
                    )
            for idx, (col_name, options) in enumerate(c_cat_opts.items()):
                with cc_cols[idx % 5]:
                    c_input[col_name] = st.selectbox(col_name, options=options, index=0)
            st.form_submit_button("Assign Customer Cluster", type="primary")

        c_in_df = pd.DataFrame([c_input])
        X_c_proc = clust_bundle["preprocessor"].transform(c_in_df)
        km_cluster = int(clust_bundle["kmeans_model"].predict(X_c_proc)[0])
        agg_cluster = int(clust_bundle["agg_centroid_mapper"].predict(X_c_proc)[0])
        user_pca = clust_bundle["pca_2d"].transform(X_c_proc)[0]

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Assigned K-Means Cluster (K=4)", f"Cluster {km_cluster}")
        sc2.metric("Assigned Agglomerative Ward Cluster (K=4)", f"Cluster {agg_cluster}")
        sc3.metric("2D PCA Coordinates", f"({user_pca[0]:.2f}, {user_pca[1]:.2f})")

    # -----------------------------------------------------------------------
    # 3D. Hosted Model Bundles Inspector & Downloader
    # -----------------------------------------------------------------------
    with pred_tab4:
        st.subheader("Serialized Model Bundles (`models/*.pkl`)")
        st.write(
            "All preprocessors and best-performing models are serialized with `joblib` in the repository's `models/` folder so the application and models can be hosted together on **Streamlit Community Cloud**, **Hugging Face Spaces**, or **Render**."
        )

        bundle_files = [
            ("Classification Bundle (Preprocessor + Tuned & Baseline AdaBoost, GBM, Logistic)", MODELS_DIR / "classification_bundle.pkl"),
            ("Regression Bundle (Preprocessor + Random Forest, GBM, Linear Regression)", MODELS_DIR / "regression_bundle.pkl"),
            ("Clustering Bundle (Preprocessor + PCA + K-Means & Agglomerative Ward)", MODELS_DIR / "clustering_bundle.pkl"),
        ]

        for label, b_path in bundle_files:
            if b_path.exists():
                size_kb = b_path.stat().st_size / 1024
                b_col1, b_col2 = st.columns([3, 1])
                b_col1.markdown(f"**{label}** — `{b_path.name}` (`{size_kb:,.1f} KB`)")
                with b_col2:
                    st.download_button(
                        label=f"Download {b_path.name}",
                        data=b_path.read_bytes(),
                        file_name=b_path.name,
                        mime="application/octet-stream",
                        key=f"dl_{b_path.name}",
                    )
