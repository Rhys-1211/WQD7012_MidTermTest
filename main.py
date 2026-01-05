import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix


# -----------------------------
# 0) Load data (use project-relative path)
# -----------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_DIR, "data", "Set05.csv")

df = pd.read_csv(DATA_PATH)

print("\n====================")
print("DATA LOADED SUCCESS")
print("Path:", DATA_PATH)
print("Shape:", df.shape)
print("====================\n")


# -----------------------------
# Q1) Current Condition / Basic Checks
# -----------------------------
print("=== Q1: Data condition ===")
print(df.head(5))
print("\nInfo:")
print(df.info())

missing = df.isna().sum().sort_values(ascending=False)
dups = df.duplicated().sum()

print("\nMissing values (top 15):")
print(missing.head(15))
print("\nDuplicate rows:", dups)

print("\nDescribe (numeric):")
print(df.describe().T)

# If these columns exist (based on your earlier Set05 format)
for col in ["state", "quarter", "year"]:
    if col in df.columns:
        if col == "year":
            print(f"\nYear range: {df['year'].min()} - {df['year'].max()}")
        else:
            print(f"\n{col} unique:", df[col].nunique())
            print(df[col].value_counts().head(10))


# -----------------------------
# Q2) KMeans - best K + comparison (no preprocessing vs preprocessing)
# -----------------------------
print("\n=== Q2: KMeans clustering ===")

cat_cols = [c for c in ["state", "quarter"] if c in df.columns]
num_cols = [c for c in df.columns if c not in cat_cols]

# A) Without preprocessing: label encode categorical + raw scale
df_raw = df.copy()
for c in cat_cols:
    le = LabelEncoder()
    df_raw[c] = le.fit_transform(df_raw[c].astype(str))
X_raw = df_raw.values

# B) With preprocessing: impute + standardize numeric, onehot categorical
preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    sparse_threshold=0.0
)
X_proc = preprocess.fit_transform(df)


def kmeans_scan(X, k_range=range(2, 9), sample_size=200, random_state=42):
    inertias, sils = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sils.append(
            silhouette_score(
                X, labels,
                sample_size=min(sample_size, len(X)),
                random_state=random_state
            )
        )
    return pd.DataFrame({"k": list(k_range), "inertia": inertias, "silhouette": sils})


raw_metrics = kmeans_scan(X_raw)
proc_metrics = kmeans_scan(X_proc)

print("\n--- Without preprocessing metrics ---")
print(raw_metrics)
print("\n--- With preprocessing metrics ---")
print(proc_metrics)

# Plot and save figures (good for report screenshots)
fig_dir = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(fig_dir, exist_ok=True)

def plot_k(metrics, title_prefix, filename_prefix):
    plt.figure()
    plt.plot(metrics["k"], metrics["inertia"], marker="o")
    plt.xlabel("k")
    plt.ylabel("Inertia (SSE)")
    plt.title(f"{title_prefix} - Elbow (Inertia)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{filename_prefix}_elbow.png"), dpi=200)
    plt.show()

    plt.figure()
    plt.plot(metrics["k"], metrics["silhouette"], marker="o")
    plt.xlabel("k")
    plt.ylabel("Silhouette (sampled)")
    plt.title(f"{title_prefix} - Silhouette")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{filename_prefix}_silhouette.png"), dpi=200)
    plt.show()

plot_k(raw_metrics, "Without preprocessing", "kmeans_raw")
plot_k(proc_metrics, "With preprocessing", "kmeans_proc")

BEST_K = int(proc_metrics.sort_values("silhouette", ascending=False).iloc[0]["k"])
print("\nSuggested BEST_K (preprocessed, highest silhouette):", BEST_K)

kmeans_proc = KMeans(n_clusters=BEST_K, n_init=20, random_state=42)
labels_proc = kmeans_proc.fit_predict(X_proc)

df_clustered = df.copy()
df_clustered["cluster"] = labels_proc

print("\nCluster counts:")
print(df_clustered["cluster"].value_counts().sort_index())

print("\nCluster profile (mean of numeric columns):")
cluster_profile = df_clustered.groupby("cluster")[num_cols].mean().round(3)
print(cluster_profile)

cluster_profile.to_csv(os.path.join(fig_dir, "cluster_profile.csv"), index=True)


# -----------------------------
# Q3) Linear Regression (choose a numeric target) + compare preproc vs no preproc
# -----------------------------
print("\n=== Q3: Linear Regression ===")

# Pick a sensible default target if exists:
candidate_targets = ["state_gdp_rm_million", "mean_household_income", "life_expectancy"]
target = None
for t in candidate_targets:
    if t in df.columns:
        target = t
        break

if target is None:
    # fallback: use last numeric column
    target = num_cols[-1]

print("Regression target:", target)

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# (A) Without preprocessing: label encode cat + raw scale
X_le = X.copy()
for c in cat_cols:
    le = LabelEncoder()
    X_le[c] = le.fit_transform(X_le[c].astype(str))

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_le, y, test_size=0.2, random_state=42)

lr_raw = LinearRegression()
lr_raw.fit(X_train2, y_train2)
pred_raw = lr_raw.predict(X_test2)

mae_raw = mean_absolute_error(y_test2, pred_raw)
rmse_raw = np.sqrt(mean_squared_error(y_test2, pred_raw))
r2_raw = r2_score(y_test2, pred_raw)

print("\n[WITHOUT preprocessing]")
print("MAE :", mae_raw)
print("RMSE:", rmse_raw)
print("R2  :", r2_raw)

# (B) With preprocessing pipeline
num_cols_X = [c for c in X.columns if c not in cat_cols]
preprocess_reg = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols_X),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    sparse_threshold=0.0
)

lr_proc = Pipeline([
    ("prep", preprocess_reg),
    ("lr", LinearRegression())
])

lr_proc.fit(X_train, y_train)
pred_proc = lr_proc.predict(X_test)

mae_proc = mean_absolute_error(y_test, pred_proc)
rmse_proc = np.sqrt(mean_squared_error(y_test, pred_proc))
r2_proc = r2_score(y_test, pred_proc)

print("\n[WITH preprocessing]")
print("MAE :", mae_proc)
print("RMSE:", rmse_proc)
print("R2  :", r2_proc)


# -----------------------------
# Q4) SVM Classification (choose another target -> binarize) + compare
# -----------------------------
print("\n=== Q4: SVM Classification ===")

candidate_cls_targets = ["gini_coefficient", "unemployment_rate", "infant_mortality_rate"]
target_cls = None
for t in candidate_cls_targets:
    if t in df.columns and t != target:
        target_cls = t
        break

if target_cls is None:
    # fallback: choose a different numeric column
    target_cls = [c for c in num_cols if c != target][0]

print("Classification base target:", target_cls)
y_cls = (df[target_cls] > df[target_cls].median()).astype(int)
X_cls = df.drop(columns=[target_cls])

X_train, X_test, y_train, y_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

# (A) Without preprocessing
X_le = X_cls.copy()
for c in cat_cols:
    le = LabelEncoder()
    X_le[c] = le.fit_transform(X_le[c].astype(str))

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_le, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

svm_raw = SVC(kernel="linear", C=1, probability=True, random_state=42)
svm_raw.fit(X_train2, y_train2)
pred = svm_raw.predict(X_test2)
proba = svm_raw.predict_proba(X_test2)[:, 1]

acc = accuracy_score(y_test2, pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test2, pred, average="binary")
auc = roc_auc_score(y_test2, proba)

print("\n[WITHOUT preprocessing]")
print("ACC:", acc, "Precision:", prec, "Recall:", rec, "F1:", f1, "AUC:", auc)
print("Confusion matrix:\n", confusion_matrix(y_test2, pred))

# (B) With preprocessing
num_cols_X = [c for c in X_cls.columns if c not in cat_cols]
preprocess_cls = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols_X),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    sparse_threshold=0.0
)

svm_proc = Pipeline([
    ("prep", preprocess_cls),
    ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42))
])

svm_proc.fit(X_train, y_train)
pred = svm_proc.predict(X_test)
proba = svm_proc.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary")
auc = roc_auc_score(y_test, proba)

print("\n[WITH preprocessing]")
print("ACC:", acc, "Precision:", prec, "Recall:", rec, "F1:", f1, "AUC:", auc)
print("Confusion matrix:\n", confusion_matrix(y_test, pred))


# -----------------------------
# Q5) Improvement: GridSearchCV tuning SVM
# -----------------------------
print("\n=== Q5: Improvement (GridSearchCV for SVM) ===")

param_grid = {
    "svm__kernel": ["linear", "rbf"],
    "svm__C": [0.1, 1, 10, 50],
    "svm__gamma": ["scale", 0.01, 0.1, 1]
}

svm_tune = Pipeline([
    ("prep", preprocess_cls),
    ("svm", SVC(probability=True, class_weight="balanced", random_state=42))
])

grid = GridSearchCV(
    svm_tune,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV score (F1):", grid.best_score_)

best_model = grid.best_estimator_
pred = best_model.predict(X_test)
proba = best_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary")
auc = roc_auc_score(y_test, proba)

print("\n[IMPROVED MODEL - tuned]")
print("ACC:", acc, "Precision:", prec, "Recall:", rec, "F1:", f1, "AUC:", auc)
print("Confusion matrix:\n", confusion_matrix(y_test, pred))

print("\nAll outputs saved to:", fig_dir)
