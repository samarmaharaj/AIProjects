"""
=============================================================================
 TASK 3 - CLASSIFICATION  (predict Patient Outcome)
 CS-415 Lab Assignment 7 : Data Mining in Hospital Management
=============================================================================
 Run:  python 02_classification.py

 Target attribute : Outcome  (Recovered / Stable / Deceased)
 Algorithms       : 1. Decision Tree (C4.5  ==  WEKA's J48)
                    2. Naive Bayes
                    3. Random Forest
                    4. K-Nearest Neighbours
 Evaluation       : stratified 10-fold cross-validation  (the WEKA default)
 Outputs          : printed report + output_*.png + classification_results.csv
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # headless -> saves PNG instead of a window
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

pd.set_option("display.width", 200)
RANDOM_STATE = 42

# =============================================================================
print("=" * 78)
print("STEP 1 : LOAD PREPROCESSED DATA")
print("=" * 78)

df = pd.read_csv("hospital_clean.csv")
print(f"Loaded hospital_clean.csv : {df.shape[0]} rows x {df.shape[1]} columns")

# ---- Feature engineering (same encoding logic as in 01_preprocess.py) -------
FEATURES_NUM = ["Age", "Blood_Pressure", "Length_of_Stay", "Hospital_Charges"]
FEATURES_CAT = ["Gender", "Disease", "Diabetes", "Admission_Type", "Treatment"]
TARGET = "Outcome"

X_raw = df[FEATURES_NUM + FEATURES_CAT].copy()
y_raw = df[TARGET].copy()

X = pd.get_dummies(X_raw, columns=FEATURES_CAT)

le = LabelEncoder()
y = le.fit_transform(y_raw)               # Recovered / Stable / Deceased -> 0/1/2

print(f"\nFeatures used ({X.shape[1]} after one-hot encoding):")
print(f"  numeric     : {FEATURES_NUM}")
print(f"  categorical : {FEATURES_CAT}")
print(f"\nClass distribution of target '{TARGET}':")
for cls, cnt in zip(*np.unique(y_raw, return_counts=True)):
    print(f"  {cls:10s} {cnt:4d}  ({cnt/len(y_raw)*100:5.1f}%)")

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

# =============================================================================
print("\n" + "=" * 78)
print("STEP 2 : DEFINE THE CLASSIFIERS")
print("=" * 78)

models = {
    # C4.5 in WEKA is 'J48'. scikit-learn's DecisionTreeClassifier uses the
    # CART algorithm with entropy criterion -> statistically the closest twin.
    "Decision Tree (J48/C4.5)": DecisionTreeClassifier(
        criterion="entropy", max_depth=6, min_samples_leaf=5,
        random_state=RANDOM_STATE),

    # Naive Bayes: assumes features are independent given the class.
    "Naive Bayes": GaussianNB(),

    # Random Forest: 200 decorrelated trees -> usually the most accurate here.
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE),

    # KNN: distance based, so features must be standardised first (pipeline).
    "K-Nearest Neighbours (k=7)": Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=7)),
    ]),
}

for name in models:
    print(f"  - {name}")

# =============================================================================
print("\n" + "=" * 78)
print("STEP 3 : TRAIN + EVALUATE WITH STRATIFIED 10-FOLD CROSS-VALIDATION")
print("=" * 78)
print("Why cross-validation instead of one split? With only 400 patients, a")
print("single 70/30 split depends heavily on which rows land in the test set.")
print("10-fold CV trains and tests the model 10 times on 10 different splits")
print("and averages the results -> a far more reliable accuracy estimate.\n")

results = []
predictions = {}

for name, model in models.items():
    # cross_val_predict gives one out-of-fold prediction per patient,
    # i.e. every prediction comes from a model that never saw that patient.
    y_pred = cross_val_predict(model, X, y, cv=cv)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y, y_pred, average="weighted", zero_division=0)

    results.append({"Algorithm": name, "Accuracy": acc,
                    "Precision": prec, "Recall": rec, "F1": f1})
    predictions[name] = y_pred

results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)

print("=" * 78)
print("CLASSIFICATION ACCURACY COMPARISON  (10-fold CV)")
print("=" * 78)
print(results_df.to_string(index=False,
                           float_format=lambda v: f"{v:.4f}"))
print("=" * 78)
best = results_df.iloc[0]
print(f"BEST MODEL: {best['Algorithm']}  with accuracy {best['Accuracy']*100:.2f}%")

# ---- Detailed report for the two required algorithms ------------------------
for name in ["Decision Tree (J48/C4.5)", "Naive Bayes"]:
    print("\n" + "-" * 78)
    print(f"DETAILED REPORT : {name}")
    print("-" * 78)
    print(classification_report(y, predictions[name],
                                target_names=le.classes_, digits=4,
                                zero_division=0))

# =============================================================================
print("=" * 78)
print("STEP 4 : CONFUSION MATRICES")
print("=" * 78)
for name in ["Decision Tree (J48/C4.5)", "Naive Bayes"]:
    cm = confusion_matrix(y, predictions[name])
    print(f"\n{name}")
    print("                 " + "".join(f"{c:>12s}" for c in le.classes_) + "   <- predicted")
    for i, cls in enumerate(le.classes_):
        print(f"  {cls:12s} " + "".join(f"{v:12d}" for v in cm[i]))
    print("  (rows = actual class)")

# =============================================================================
print("\n" + "=" * 78)
print("STEP 5 : THE DECISION TREE RULES  (what a doctor can read)")
print("=" * 78)
dt = DecisionTreeClassifier(criterion="entropy", max_depth=6,
                            min_samples_leaf=5, random_state=RANDOM_STATE)
dt.fit(X, y)
rules = export_text(dt, feature_names=list(X.columns), class_names=list(le.classes_),
                    decimals=2)
print(rules)

# =============================================================================
print("=" * 78)
print("STEP 6 : FEATURE IMPORTANCE  (which attributes actually drive Outcome?)")
print("=" * 78)
rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
rf.fit(X, y)
imp = (pd.Series(rf.feature_importances_, index=X.columns)
       .sort_values(ascending=False))
print(imp.head(12).round(4).to_string())

# =============================================================================
print("\n" + "=" * 78)
print("STEP 7 : SAVING RESULTS AND FIGURES")
print("=" * 78)

results_df.to_csv("outputs/classification_results.csv", index=False)
print("  outputs/classification_results.csv")

with open("outputs/classification_report.txt", "w") as f:
    f.write("HOSPITAL OUTCOME PREDICTION - CLASSIFICATION RESULTS\n")
    f.write("10-fold stratified cross-validation\n\n")
    f.write(results_df.to_string(index=False) + "\n\n")
    for name in models:
        f.write("=" * 70 + f"\n{name}\n" + "=" * 70 + "\n")
        f.write(classification_report(y, predictions[name],
                                      target_names=le.classes_, digits=4,
                                      zero_division=0) + "\n")
    f.write("=" * 70 + "\nDECISION TREE RULES\n" + "=" * 70 + "\n")
    f.write(rules + "\n")
    f.write("=" * 70 + "\nFEATURE IMPORTANCE (Random Forest)\n" + "=" * 70 + "\n")
    f.write(imp.round(4).to_string() + "\n")
print("  outputs/classification_report.txt")

# ---- FIGURE 1 : accuracy comparison bar chart -------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
bars = ax.barh(results_df["Algorithm"], results_df["Accuracy"],
               color=colors[:len(results_df)], edgecolor="black")
ax.set_xlabel("Accuracy (10-fold CV)")
ax.set_title("Algorithm Comparison - Predicting Patient Outcome", fontweight="bold")
ax.set_xlim(0, 1.05)
for b, v in zip(bars, results_df["Accuracy"]):
    ax.text(v + 0.01, b.get_y() + b.get_height() / 2,
            f"{v*100:.2f}%", va="center", fontweight="bold")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("outputs/fig1_algorithm_comparison.png", dpi=140)
plt.close()

# ---- FIGURE 2 : confusion matrices side by side -----------------------------
top2 = results_df["Algorithm"].head(2).tolist()
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, name in zip(axes, top2):
    cm = confusion_matrix(y, predictions[name])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{name}\nAccuracy = {accuracy_score(y, predictions[name])*100:.2f}%",
                 fontweight="bold", fontsize=11)
    ax.set_xticks(range(len(le.classes_)))
    ax.set_yticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_, rotation=30)
    ax.set_yticklabels(le.classes_)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig("outputs/fig2_confusion_matrices.png", dpi=140)
plt.close()

# ---- FIGURE 3 : feature importance -----------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
top = imp.head(10)
ax.barh(top.index[::-1], top.values[::-1], color="#2E86AB", edgecolor="black")
ax.set_xlabel("Importance")
ax.set_title("Top 10 Most Important Attributes (Random Forest)", fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/fig3_feature_importance.png", dpi=140)
plt.close()

# ---- FIGURE 4 : the decision tree itself ------------------------------------
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(dt, feature_names=list(X.columns), class_names=list(le.classes_),
          filled=True, rounded=True, fontsize=8, max_depth=3, ax=ax)
ax.set_title("Decision Tree (top 3 levels) - Predicting Patient Outcome",
             fontweight="bold", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/fig4_decision_tree.png", dpi=120)
plt.close()

print("  outputs/fig1_algorithm_comparison.png")
print("  outputs/fig2_confusion_matrices.png")
print("  outputs/fig3_feature_importance.png")
print("  outputs/fig4_decision_tree.png")

print("\n" + "=" * 78)
print("CLASSIFICATION COMPLETE  ->  next: python 03_clustering.py")
print("=" * 78)
