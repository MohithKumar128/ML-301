"""Model hub and artifact builder for the ML-301 Capstone Streamlit Workbench.

Provides:
1. Pre-trained model bundle serialization (`models/*.pkl` via joblib) for fast cloud hosting.
2. Live training and evaluation runners for all 22 algorithms across Regression, Classification, and Clustering.
3. Single-record inference helpers for the Live Predictor tab.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
     accuracy_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, RobustScaler, StandardScaler, label_binarize
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

GLOBAL_SEED = 42


def get_repo_root() -> Path:
    """Locate the ML-301 repository root directory."""
    here = Path(__file__).resolve().parent
    for candidate in (here.parent, Path.cwd(), *Path.cwd().parents):
        if (candidate / "data" / "diabetes_risk_prediction_dataset.csv").exists():
            return candidate
    return here.parent


REPO_ROOT = get_repo_root()
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"


# ---------------------------------------------------------------------------
# 1. Data & Preprocessing Loaders (Matching Notebook Pipelines Exactly)
# ---------------------------------------------------------------------------

def prepare_regression_data() -> dict[str, Any]:
    """Prepare the Health Insurance dataset exactly as in notebooks/regression.ipynb."""
    raw_df = pd.read_csv(DATA_DIR / "health_insurance.csv")
    df = raw_df.drop_duplicates().copy()

    X = df.drop(columns="claim")
    y = df["claim"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=GLOBAL_SEED
    )

    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

    X_train = X_train.copy()
    X_test = X_test.copy()
    for frame in (X_train, X_test):
        frame["age_bmi_interaction"] = frame["age"] * frame["bmi"]
    numeric_features_ext = numeric_features + ["age_bmi_interaction"]

    preprocessor = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
            ]),
            numeric_features_ext,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
            ]),
            categorical_features,
        ),
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Top numeric features for Polynomial Regression (matching Review 1 notebook)
    corr_Order = (
        X_train[numeric_features]
        .apply(lambda col: col.fillna(col.median()))
        .corrwith(y_train)
        .abs()
        .sort_values(ascending=False)
    )
    top_poly_cols = corr_Order.head(4).index.tolist()
    poly_idx = [numeric_features_ext.index(c) for c in top_poly_cols]

    # Schema defaults for Live Predictor UI
    raw_numeric_defaults = {
        col: {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
        }
        for col in numeric_features
    }
    raw_categorical_options = {
        col: sorted(df[col].dropna().astype(str).unique().tolist())
        for col in categorical_features
    }

    return {
        "df": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_processed": X_train_processed,
        "X_test_processed": X_test_processed,
        "preprocessor": preprocessor,
        "feature_names": preprocessor.get_feature_names_out().tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "poly_idx": poly_idx,
        "numeric_defaults": raw_numeric_defaults,
        "categorical_options": raw_categorical_options,
    }


def prepare_classification_data() -> dict[str, Any]:
    """Prepare the Diabetes Risk dataset exactly as in notebooks/classification.ipynb."""
    df = pd.read_csv(DATA_DIR / "diabetes_risk_prediction_dataset.csv")
    target_col = "Diabetes_Risk"

    df["Diabetes_Risk_Code"] = df[target_col].map({"Low": 0, "Moderate": 1, "High": 2})
    df["Pulse_Pressure"] = df["Blood_Pressure_Systolic"] - df["Blood_Pressure_Diastolic"]

    drop_cols = [
        target_col,
        "Patient_ID",
        "Diabetes_Risk_Code",
        "Diabetes_Risk_Score",
        "AI_Health_Recommendation",
        "Doctor_Consultation_Needed",
    ]
    X = df.drop(columns=drop_cols)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=GLOBAL_SEED, stratify=y
    )
    X_train = X_train.copy()
    X_test = X_test.copy()

    # BMI IQR capping fitted on training split only
    q1 = float(X_train["BMI"].quantile(0.25))
    q3 = float(X_train["BMI"].quantile(0.75))
    iqr = q3 - q1
    bmi_lower = q1 - 1.5 * iqr
    bmi_upper = q3 + 1.5 * iqr
    X_train["BMI"] = X_train["BMI"].clip(lower=bmi_lower, upper=bmi_upper)
    X_test["BMI"] = X_test["BMI"].clip(lower=bmi_lower, upper=bmi_upper)

    num_features = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_features = [c for c in X_train.columns if c not in num_features]

    preprocessor = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            num_features,
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cat_features,
        ),
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    raw_input_num = [c for c in num_features if c != "Pulse_Pressure"]
    numeric_defaults = {
        col: {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
        }
        for col in raw_input_num
    }
    categorical_options = {
        col: sorted(df[col].dropna().astype(str).unique().tolist())
        for col in cat_features
    }

    return {
        "df": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_processed": X_train_processed,
        "X_test_processed": X_test_processed,
        "preprocessor": preprocessor,
        "feature_names": preprocessor.get_feature_names_out().tolist(),
        "num_features": num_features,
        "cat_features": cat_features,
        "bmi_bounds": (bmi_lower, bmi_upper),
        "numeric_defaults": numeric_defaults,
        "categorical_options": categorical_options,
    }


def prepare_clustering_data() -> dict[str, Any]:
    """Prepare the Bank Marketing dataset exactly as in notebooks/clustering.ipynb."""
    df = pd.read_csv(DATA_DIR / "bank-additional-full.csv", sep=";")
    df_clean = df.drop_duplicates().copy().reset_index(drop=True)
    X = df_clean.drop(columns=["y"], errors="ignore")

    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numerical_cols,
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_cols,
        ),
    ])

    X_processed = preprocessor.fit_transform(X)

    rng = np.random.RandomState(GLOBAL_SEED)
    sample_indices = np.sort(rng.choice(len(X_processed), size=5000, replace=False))
    X_sample = X_processed[sample_indices]
    df_sample = df_clean.iloc[sample_indices].reset_index(drop=True)

    pca_2d = PCA(n_components=2, random_state=GLOBAL_SEED)
    X_sample_pca = pca_2d.fit_transform(X_sample)

    numeric_defaults = {
        col: {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "median": float(X[col].median()),
        }
        for col in numerical_cols
    }
    categorical_options = {
        col: sorted(X[col].dropna().astype(str).unique().tolist())
        for col in categorical_cols
    }

    return {
        "df_clean": df_clean,
        "df_sample": df_sample,
        "X": X,
        "X_processed": X_processed,
        "sample_indices": sample_indices,
        "X_sample": X_sample,
        "X_sample_pca": X_sample_pca,
        "pca_2d": pca_2d,
        "preprocessor": preprocessor,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "numeric_defaults": numeric_defaults,
        "categorical_options": categorical_options,
    }


# ---------------------------------------------------------------------------
# 2. Pre-trained Bundle Builder & Loader (Saved in models/*.pkl)
# ---------------------------------------------------------------------------

def build_and_save_model_bundles() -> dict[str, Path]:
    """Train and serialize lightweight deployment bundles into models/*.pkl."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Regression bundle
    reg = prepare_regression_data()
    rf_reg = RandomForestRegressor(
        n_estimators=150, max_depth=18, random_state=GLOBAL_SEED, n_jobs=-1
    )
    rf_reg.fit(reg["X_train_processed"], reg["y_train"])

    gbr_reg = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.1, max_depth=4, random_state=GLOBAL_SEED
    )
    gbr_reg.fit(reg["X_train_processed"], reg["y_train"])

    lin_reg = LinearRegression()
    lin_reg.fit(reg["X_train_processed"], reg["y_train"])

    reg_bundle_path = MODELS_DIR / "regression_bundle.pkl"
    joblib.dump(
        {
            "preprocessor": reg["preprocessor"],
            "feature_names": reg["feature_names"],
            "numeric_features": reg["numeric_features"],
            "categorical_features": reg["categorical_features"],
            "numeric_defaults": reg["numeric_defaults"],
            "categorical_options": reg["categorical_options"],
            "models": {
                "Random Forest Regressor (Best)": rf_reg,
                "Gradient Boosting Regressor": gbr_reg,
                "Linear Regression (Baseline)": lin_reg,
            },
        },
        reg_bundle_path,
        compress=3,
    )

    # 2. Classification bundle
    clf = prepare_classification_data()
    ada_tuned = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=GLOBAL_SEED),
        n_estimators=200,
        learning_rate=1.0,
        random_state=GLOBAL_SEED,
    )
    ada_tuned.fit(clf["X_train_processed"], clf["y_train"])

    ada_baseline = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=GLOBAL_SEED),
        n_estimators=100,
        learning_rate=1.0,
        random_state=GLOBAL_SEED,
    )
    ada_baseline.fit(clf["X_train_processed"], clf["y_train"])

    gb_clf = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=GLOBAL_SEED
    )
    gb_clf.fit(clf["X_train_processed"], clf["y_train"])

    log_clf = LogisticRegression(max_iter=1000, random_state=GLOBAL_SEED)
    log_clf.fit(clf["X_train_processed"], clf["y_train"])

    clf_bundle_path = MODELS_DIR / "classification_bundle.pkl"
    joblib.dump(
        {
            "preprocessor": clf["preprocessor"],
            "feature_names": clf["feature_names"],
            "num_features": clf["num_features"],
            "cat_features": clf["cat_features"],
            "bmi_bounds": clf["bmi_bounds"],
            "numeric_defaults": clf["numeric_defaults"],
            "categorical_options": clf["categorical_options"],
            "models": {
                "AdaBoost (Tuned — Best Model)": ada_tuned,
                "AdaBoost (Baseline)": ada_baseline,
                "Gradient Boosting Classifier": gb_clf,
                "Logistic Regression (Baseline)": log_clf,
            },
        },
        clf_bundle_path,
        compress=3,
    )

    # 3. Clustering bundle
    clust = prepare_clustering_data()
    kmeans_model = KMeans(n_clusters=4, random_state=GLOBAL_SEED, n_init=10)
    kmeans_full_labels = kmeans_model.fit_predict(clust["X_processed"])
    kmeans_sample_labels = kmeans_full_labels[clust["sample_indices"]]

    agg_model = AgglomerativeClustering(n_clusters=4, metric="euclidean", linkage="ward")
    agg_sample_labels = agg_model.fit_predict(clust["X_sample"])

    agg_centroid_mapper = NearestCentroid()
    agg_centroid_mapper.fit(clust["X_sample"], agg_sample_labels)

    # Precompute K-Means metrics on the same 5,000-customer sample and full dataset
    kmeans_metrics_sample = {
        "Silhouette_Score": float(silhouette_score(clust["X_sample"], kmeans_sample_labels)),
        "Davies_Bouldin_Index": float(davies_bouldin_score(clust["X_sample"], kmeans_sample_labels)),
        "Calinski_Harabasz_Index": float(calinski_harabasz_score(clust["X_sample"], kmeans_sample_labels)),
    }
    agg_metrics_sample = {
        "Silhouette_Score": float(silhouette_score(clust["X_sample"], agg_sample_labels)),
        "Davies_Bouldin_Index": float(davies_bouldin_score(clust["X_sample"], agg_sample_labels)),
        "Calinski_Harabasz_Index": float(calinski_harabasz_score(clust["X_sample"], agg_sample_labels)),
    }

    # Build customer segment profiles on the sample
    df_prof = clust["df_sample"].copy()
    df_prof["KMeans_Cluster"] = kmeans_sample_labels
    df_prof["Agg_Cluster"] = agg_sample_labels

    clust_bundle_path = MODELS_DIR / "clustering_bundle.pkl"
    joblib.dump(
        {
            "preprocessor": clust["preprocessor"],
            "pca_2d": clust["pca_2d"],
            "X_sample_pca": clust["X_sample_pca"],
            "kmeans_model": kmeans_model,
            "agg_centroid_mapper": agg_centroid_mapper,
            "kmeans_sample_labels": kmeans_sample_labels,
            "agg_sample_labels": agg_sample_labels,
            "kmeans_metrics_sample": kmeans_metrics_sample,
            "agg_metrics_sample": agg_metrics_sample,
            "numerical_cols": clust["numerical_cols"],
            "categorical_cols": clust["categorical_cols"],
            "numeric_defaults": clust["numeric_defaults"],
            "categorical_options": clust["categorical_options"],
            "profile_df": df_prof[["age", "duration", "campaign", "euribor3m", "nr.employed", "job", "marital", "education", "y", "KMeans_Cluster", "Agg_Cluster"]],
        },
        clust_bundle_path,
        compress=3,
    )

    return {
        "regression": reg_bundle_path,
        "classification": clf_bundle_path,
        "clustering": clust_bundle_path,
    }


# ---------------------------------------------------------------------------
# 3. Interactive Model Runners (Train & Evaluate Any Algorithm on Demand)
# ---------------------------------------------------------------------------

REGRESSION_MODELS = [
    "Linear Regression",
    "Ridge Regression",
    "Lasso Regression",
    "ElasticNet Regression",
    "Polynomial Regression",
    "Decision Tree Regressor",
    "Random Forest Regressor",
    "Gradient Boosting Regressor",
    "Support Vector Regressor (SVR)",
    "KNN Regressor",
]

CLASSIFICATION_MODELS = [
    "Logistic Regression",
    "K-Nearest Neighbors (KNN)",
    "Gaussian Naive Bayes",
    "Decision Tree Classifier",
    "Support Vector Machine (SVC)",
    "Random Forest Classifier",
    "AdaBoost Classifier",
    "Gradient Boosting Classifier",
    "Bagging Classifier (Decision Tree)",
    "MLP Classifier (Neural Network)",
]

CLUSTERING_MODELS = [
    "K-Means Clustering",
    "Agglomerative Hierarchical Clustering",
]


def run_regression_model(
    model_name: str,
    params: dict[str, Any],
    data: dict[str, Any],
    pretrained_model: Any | None = None,
) -> dict[str, Any]:
    """Train (or use pre-trained bundle model) and evaluate the selected regression model."""
    X_train = data["X_train_processed"]
    X_test = data["X_test_processed"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]

    if pretrained_model is not None:
        model = pretrained_model
    elif model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Ridge Regression":
        model = Ridge(alpha=float(params.get("alpha", 1.0)), random_state=GLOBAL_SEED)
    elif model_name == "Lasso Regression":
        model = Lasso(alpha=float(params.get("alpha", 10.0)), max_iter=3000, random_state=GLOBAL_SEED)
    elif model_name == "ElasticNet Regression":
        model = ElasticNet(
            alpha=float(params.get("alpha", 0.1)),
            l1_ratio=float(params.get("l1_ratio", 0.5)),
            max_iter=3000,
            random_state=GLOBAL_SEED,
        )
    elif model_name == "Polynomial Regression":
        degree = int(params.get("degree", 2))
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_idx = data["poly_idx"]
        X_train_poly = np.hstack([X_train, poly.fit_transform(X_train[:, poly_idx])])
        X_test_poly = np.hstack([X_test, poly.transform(X_test[:, poly_idx])])
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        y_pred = model.predict(X_test_poly)
        r2 = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        return {
            "model_name": f"Polynomial Regression (degree={degree})",
            "metrics": {"R2": r2, "RMSE": rmse, "MAE": mae},
            "y_test": y_test.to_numpy(),
            "y_pred": y_pred,
            "feature_importance": None,
        }
    elif model_name == "Decision Tree Regressor":
        model = DecisionTreeRegressor(
            max_depth=int(params.get("max_depth", 10)),
            min_samples_split=int(params.get("min_samples_split", 5)),
            random_state=GLOBAL_SEED,
        )
    elif model_name == "Random Forest Regressor":
        model = RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 150)),
            max_depth=int(params.get("max_depth", 18)),
            random_state=GLOBAL_SEED,
            n_jobs=-1,
        )
    elif model_name == "Gradient Boosting Regressor":
        model = GradientBoostingRegressor(
            n_estimators=int(params.get("n_estimators", 150)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 3)),
            random_state=GLOBAL_SEED,
        )
    elif model_name == "Support Vector Regressor (SVR)":
        model = SVR(
            C=float(params.get("C", 100.0)),
            kernel=str(params.get("kernel", "rbf")),
        )
    elif model_name == "KNN Regressor":
        model = KNeighborsRegressor(
            n_neighbors=int(params.get("n_neighbors", 5)),
            weights=str(params.get("weights", "distance")),
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown regression model: {model_name}")

    if pretrained_model is None:
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    fi_df = None
    if hasattr(model, "feature_importances_"):
        fi_df = (
            pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
            .sort_values(by="Importance", ascending=False)
            .head(12)
        )
    elif hasattr(model, "coef_"):
        coefs = np.ravel(model.coef_)
        if len(coefs) == len(feature_names):
            fi_df = (
                pd.DataFrame({"Feature": feature_names, "Importance": np.abs(coefs)})
                .sort_values(by="Importance", ascending=False)
                .head(12)
            )

    return {
        "model_name": model_name,
        "metrics": {"R2": r2, "RMSE": rmse, "MAE": mae},
        "y_test": y_test.to_numpy(),
        "y_pred": y_pred,
        "feature_importance": fi_df,
    }


def run_classification_model(
    model_name: str,
    params: dict[str, Any],
    data: dict[str, Any],
    pretrained_model: Any | None = None,
) -> dict[str, Any]:
    """Train (or use pre-trained bundle model) and evaluate the selected classification model."""
    X_train = data["X_train_processed"]
    X_test = data["X_test_processed"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]

    if pretrained_model is not None:
        model = pretrained_model
    elif model_name == "Logistic Regression":
        model = LogisticRegression(
            C=float(params.get("C", 1.0)),
            max_iter=int(params.get("max_iter", 1000)),
            random_state=GLOBAL_SEED,
        )
    elif model_name == "K-Nearest Neighbors (KNN)":
        model = KNeighborsClassifier(
            n_neighbors=int(params.get("n_neighbors", 7)),
            weights=str(params.get("weights", "uniform")),
            metric=str(params.get("metric", "minkowski")),
            n_jobs=-1,
        )
    elif model_name == "Gaussian Naive Bayes":
        model = GaussianNB(var_smoothing=float(params.get("var_smoothing", 1e-9)))
    elif model_name == "Decision Tree Classifier":
        model = DecisionTreeClassifier(
            max_depth=int(params.get("max_depth", 8)),
            criterion=str(params.get("criterion", "gini")),
            random_state=GLOBAL_SEED,
        )
    elif model_name == "Support Vector Machine (SVC)":
        # Subsample option for responsive interactive SVC runs on 40k rows
        train_limit = int(params.get("train_limit", 12000))
        if train_limit < len(X_train):
            X_tr_sub, _, y_tr_sub, _ = train_test_split(
                X_train, y_train, train_size=train_limit, stratify=y_train, random_state=GLOBAL_SEED
            )
        else:
            X_tr_sub, y_tr_sub = X_train, y_train
        model = SVC(
            C=float(params.get("C", 1.0)),
            kernel=str(params.get("kernel", "rbf")),
            probability=True,
            random_state=GLOBAL_SEED,
        )
        model.fit(X_tr_sub, y_tr_sub)
    elif model_name == "Random Forest Classifier":
        model = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=params.get("max_depth", None),
            random_state=GLOBAL_SEED,
            n_jobs=-1,
        )
    elif model_name == "AdaBoost Classifier":
        model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=int(params.get("base_max_depth", 1)),
                random_state=GLOBAL_SEED,
            ),
            n_estimators=int(params.get("n_estimators", 200)),
            learning_rate=float(params.get("learning_rate", 1.0)),
            random_state=GLOBAL_SEED,
        )
    elif model_name == "Gradient Boosting Classifier":
        model = GradientBoostingClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 3)),
            random_state=GLOBAL_SEED,
        )
    elif model_name == "Bagging Classifier (Decision Tree)":
        model = BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=GLOBAL_SEED),
            n_estimators=int(params.get("n_estimators", 50)),
            max_samples=float(params.get("max_samples", 0.8)),
            random_state=GLOBAL_SEED,
            n_jobs=-1,
        )
    elif model_name == "MLP Classifier (Neural Network)":
        hidden_units = int(params.get("hidden_units", 100))
        model = MLPClassifier(
            hidden_layer_sizes=(hidden_units,),
            activation=str(params.get("activation", "relu")),
            solver="adam",
            max_iter=int(params.get("max_iter", 100)),
            early_stopping=True,
            random_state=GLOBAL_SEED,
        )
    else:
        raise ValueError(f"Unknown classification model: {model_name}")

    if pretrained_model is None and model_name != "Support Vector Machine (SVC)":
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    classes = model.classes_.tolist()

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    auc = float(roc_auc_score(y_test, y_proba, labels=classes, multi_class="ovr", average="weighted"))
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    # Multi-class OvR ROC curves
    y_test_bin = label_binarize(y_test, classes=classes)
    roc_curves = {}
    for idx, cls_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, idx], y_proba[:, idx])
        roc_curves[cls_name] = (fpr, tpr)

    fi_df = None
    if hasattr(model, "feature_importances_"):
        fi_df = (
            pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
            .sort_values(by="Importance", ascending=False)
            .head(12)
        )
    elif hasattr(model, "coef_"):
        coefs = np.abs(model.coef_).mean(axis=0)
        if len(coefs) == len(feature_names):
            fi_df = (
                pd.DataFrame({"Feature": feature_names, "Importance": coefs})
                .sort_values(by="Importance", ascending=False)
                .head(12)
            )

    return {
        "model_name": model_name,
        "metrics": {
            "Accuracy": acc,
            "Weighted Precision": prec,
            "Weighted Recall": rec,
            "Weighted F1-score": f1,
            "ROC-AUC": auc,
        },
        "classes": classes,
        "confusion_matrix": cm,
        "roc_curves": roc_curves,
        "feature_importance": fi_df,
    }


def run_clustering_model(model_name: str, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """Run K-Means or Agglomerative Clustering and compute metrics & visualization data."""
    X_sample = data["X_sample"]
    X_sample_pca = data["X_sample_pca"]
    n_clusters = int(params.get("n_clusters", 4))

    if model_name == "K-Means Clustering":
        n_init = int(params.get("n_init", 10))
        model = KMeans(n_clusters=n_clusters, random_state=GLOBAL_SEED, n_init=n_init)
        labels = model.fit_predict(X_sample)
        linkage_matrix = None
    elif model_name == "Agglomerative Hierarchical Clustering":
        linkage_method = str(params.get("linkage", "ward"))
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        labels = model.fit_predict(X_sample)
        linkage_matrix = linkage(X_sample, method=linkage_method) if params.get("compute_dendrogram", True) else None
    else:
        raise ValueError(f"Unknown clustering model: {model_name}")

    sil = float(silhouette_score(X_sample, labels))
    dbi = float(davies_bouldin_score(X_sample, labels))
    chi = float(calinski_harabasz_score(X_sample, labels))

    cluster_counts = (
        pd.Series(labels, name="Cluster")
        .value_counts()
        .sort_index()
        .reset_index()
    )
    cluster_counts.columns = ["Cluster", "Customers"]
    cluster_counts["Share (%)"] = (cluster_counts["Customers"] / len(labels) * 100).round(2)

    return {
        "model_name": model_name,
        "metrics": {
            "Silhouette Score": sil,
            "Davies-Bouldin Index": dbi,
            "Calinski-Harabasz Index": chi,
        },
        "labels": labels,
        "pca_coords": X_sample_pca,
        "cluster_counts": cluster_counts,
        "linkage_matrix": linkage_matrix,
        "n_clusters": n_clusters,
    }


def load_saved_comparisons() -> dict[str, pd.DataFrame]:
    """Load the saved evaluation CSV tables across all three tracks."""
    # 1. Regression 10-model table
    reg_dir = RESULTS_DIR / "regression"
    reg_files = sorted(
        p for p in reg_dir.glob("*.csv") if not p.name.startswith("00_")
    )
    reg_df = pd.concat([pd.read_csv(p) for p in reg_files], ignore_index=True)
    reg_df = reg_df.sort_values(by="R2", ascending=False).reset_index(drop=True)
    reg_df.index = reg_df.index + 1
    reg_df.index.name = "Rank"

    # 2. Classification 10-model table + A3 tuning table
    clf_dir = RESULTS_DIR / "classification_results"
    clf_comp_path = clf_dir / "classification_comparison.csv"
    if clf_comp_path.exists():
        clf_df = pd.read_csv(clf_comp_path)
    else:
        clf_files = sorted(
            p for p in clf_dir.glob("*.csv") if p.name[:2].isdigit() and int(p.name[:2]) <= 10
        )
        clf_df = pd.concat([pd.read_csv(p) for p in clf_files], ignore_index=True).rename(
            columns={
                "Precision_weighted": "Weighted Precision",
                "Recall_weighted": "Weighted Recall",
                "F1_weighted": "Weighted F1-score",
                "ROC_AUC": "ROC-AUC",
            }
        )
    clf_ranked = clf_df.sort_values(by="Weighted F1-score", ascending=False).reset_index(drop=True)
    clf_ranked.index = clf_ranked.index + 1
    clf_ranked.index.name = "Rank"

    tuning_path = clf_dir / "11_adaboost_tuning_comparison.csv"
    tuning_df = pd.read_csv(tuning_path) if tuning_path.exists() else pd.DataFrame()

    # 3. Clustering comparison tables
    clust_dir = RESULTS_DIR / "clustering"
    linkage_path = clust_dir / "agglomerative_linkage_comparison.csv"
    linkage_df = pd.read_csv(linkage_path) if linkage_path.exists() else pd.DataFrame()

    # Combine K-Means and Agglomerative metrics from clustering bundle if available
    clust_bundle_path = MODELS_DIR / "clustering_bundle.pkl"
    if clust_bundle_path.exists():
        cb = joblib.load(clust_bundle_path)
        km_m = cb["kmeans_metrics_sample"]
        ag_m = cb["agg_metrics_sample"]
        clust_comp_df = pd.DataFrame([
            {
                "Algorithm": "K-Means Clustering (K=4)",
                "Sample Size": 5000,
                "Clusters (K)": 4,
                "Silhouette Score": km_m["Silhouette_Score"],
                "Davies-Bouldin Index": km_m["Davies_Bouldin_Index"],
                "Calinski-Harabasz Index": km_m["Calinski_Harabasz_Index"],
            },
            {
                "Algorithm": "Agglomerative Clustering (Ward, K=4)",
                "Sample Size": 5000,
                "Clusters (K)": 4,
                "Silhouette Score": ag_m["Silhouette_Score"],
                "Davies-Bouldin Index": ag_m["Davies_Bouldin_Index"],
                "Calinski-Harabasz Index": ag_m["Calinski_Harabasz_Index"],
            },
        ])
    else:
        clust_comp_df = pd.DataFrame()

    return {
        "regression": reg_df,
        "classification": clf_ranked,
        "classification_raw": clf_df,
        "classification_tuning": tuning_df,
        "clustering_comparison": clust_comp_df,
        "clustering_linkage": linkage_df,
    }


if __name__ == "__main__":
    paths = build_and_save_model_bundles()
    for track, p in paths.items():
        size_kb = p.stat().st_size / 1024
        print(f"Saved {track:15s} bundle -> {p} ({size_kb:.1f} KB)")
