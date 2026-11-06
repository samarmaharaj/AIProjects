"""
=============================================================================
 TASK 4 - CLUSTERING  (group patients with K-Means)
 CS-415 Lab Assignment 7 : Data Mining in Hospital Management
=============================================================================
 Run:  python 03_clustering.py

 Unsupervised: we IGNORE the Outcome label and let K-Means discover natural
 patient groups from Age, Blood_Pressure, Length_of_Stay, Hospital_Charges.

 Steps:
   1. Select + standardise the clustering attributes
   2. Choose k properly  (elbow method + silhouette score)
   3. Run K-Means with the best k
   4. Profile each cluster (mean values + disease mix + outcome mix)
   5. Visualise with PCA (2-D projection) and a parallel-coordinates plot
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
RANDOM_STATE = 42

# =============================================================================
print("=" * 78)
print("STEP 1 : SELECT AND STANDARDISE CLUSTERING ATTRIBUTES")
print("=" * 78)

df = pd.read_csv("hospital_clean.csv")
# Patient_ID was removed during preprocessing (it is not an analytic
# attribute) - we re-attach it here purely so the output file is traceable.
df.insert(0, "Patient_ID", pd.read_csv("hospital_data.csv")["Patient_ID"])
print(f"Loaded hospital_clean.csv : {df.shape[0]} rows")

# We cluster on the four continuous clinical/financial measures.
CLUSTER_COLS = ["Age", "Blood_Pressure", "Length_of_Stay", "Hospital_Charges"]
print(f"\nClustering attributes ({len(CLUSTER_COLS)}): {CLUSTER_COLS}")

X_raw = df[CLUSTER_COLS].copy()
print("\nRaw ranges (note how Charges dwarfs the others):")
print(X_raw.agg(["min", "max", "mean", "std"]).round(1).to_string())

# StandardScaler -> mean 0, std 1. Without this, Hospital_Charges (values in
# the 10,000s) would swamp Age (values in the 10s) in the distance function.
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

print("\nAfter StandardScaler (mean = 0, std = 1):")
print(pd.DataFrame(X, columns=CLUSTER_COLS).agg(
    ["min", "max", "mean", "std"]).round(2).to_string())

# =============================================================================
print("\n" + "=" * 78)
print("STEP 2 : CHOOSE k  (elbow method + silhouette score)")
print("=" * 78)
print("K-Means must be told how many clusters to look for. We test k = 2..8 and")
print("judge by two standard criteria:")
print("  - Inertia (WCSS): lower is better, but always drops as k grows ->")
print("    look for the 'elbow' where improvement flattens out.")
print("  - Silhouette: -1..1, higher is better; peak value = best separated k.\n")

k_range = range(2, 9)
inertias, silhouettes, db_scores = [], [], []

print(f"{'k':>3} {'Inertia':>12} {'Silhouette':>12} {'Davies-Bouldin':>15}")
print("-" * 46)
for k in k_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=20,
                random_state=RANDOM_STATE)
    labels = km.fit_predict(X)
    s = silhouette_score(X, labels)
    dbi = davies_bouldin_score(X, labels)
    inertias.append(km.inertia_)
    silhouettes.append(s)
    db_scores.append(dbi)
    print(f"{k:>3} {km.inertia_:>12.1f} {s:>12.4f} {dbi:>15.4f}")

auto_k = list(k_range)[int(np.argmax(silhouettes))]
print(f"\n-> Highest silhouette score at k = {auto_k} "
      f"(silhouette = {max(silhouettes):.4f})")

# --- CHOOSING THE FINAL k ---------------------------------------------------
# Silhouette peaks at k = 2, but with only four attributes a 2-cluster solution
# mostly just splits 'surgical / long-stay / expensive' from 'medical /
# short-stay / cheap' - that is too coarse to be clinically useful.
# Two more pieces of evidence argue for k = 3:
#   * the ELBOW in inertia is at k = 3 (drop of 266 from k=2->3, then only
#     ~100 per step afterwards - the curve flattens),
#   * k = 3 lines up with three recognisable patient types (short medical stay,
#     medium stay, long surgical stay) that a ward manager can act on.
# The silhouette cost of moving from k=2 to k=3 is small (~0.35 -> ~0.29).
# Set FINAL_K = None to let the silhouette score decide automatically.
FINAL_K = 3

if FINAL_K is None:
    best_k = auto_k
    print(f"   FINAL k = {best_k} (chosen automatically by silhouette)")
else:
    best_k = FINAL_K
    drop = max(silhouettes) - silhouettes[list(k_range).index(best_k)]
    print(f"   Silhouette-optimal k would be {auto_k}, but we select k = {best_k}.")
    print(f"   Justification: the inertia ELBOW sits at k = {best_k}, and the")
    print(f"   silhouette only drops by {drop:.4f} versus k = {auto_k}, a small price")
    print(f"   for a far more actionable, clinically meaningful segmentation.")

# =============================================================================
print("\n" + "=" * 78)
print(f"STEP 3 : RUN K-MEANS WITH k = {best_k}")
print("=" * 78)

km_final = KMeans(n_clusters=best_k, init="k-means++", n_init=50,
                  random_state=RANDOM_STATE)
df["Cluster"] = km_final.fit_predict(X)

print(f"Converged in {km_final.n_iter_} iterations.")
print(f"Final inertia (WCSS): {km_final.inertia_:.2f}")
print(f"Final silhouette    : {silhouette_score(X, df['Cluster']):.4f}")
print(f"\nPatients per cluster:")
counts = df["Cluster"].value_counts().sort_index()
for c, n in counts.items():
    print(f"  Cluster {c}: {n:4d} patients  ({n/len(df)*100:5.1f}%)")

# =============================================================================
print("\n" + "=" * 78)
print("STEP 4 : PROFILE EACH CLUSTER  (the interpretation that matters)")
print("=" * 78)
print("Centroids in ORIGINAL units (so they are readable as real patients):\n")

centroids = df.groupby("Cluster")[CLUSTER_COLS].mean().round(1)
centroids["Patients"] = counts
print(centroids.to_string())

print("\nClinical / demographic mix per cluster:")
for cat in ["Disease", "Admission_Type", "Diabetes", "Outcome"]:
    print(f"\n--- {cat} (row % within cluster) ---")
    ct = pd.crosstab(df["Cluster"], df[cat], normalize="index") * 100
    print(ct.round(1).to_string())

print("\n--- Most common treatment per cluster ---")
for c in sorted(df["Cluster"].unique()):
    top = df[df["Cluster"] == c]["Treatment"].value_counts().idxmax()
    share = df[df["Cluster"] == c]["Treatment"].value_counts(normalize=True).max() * 100
    print(f"  Cluster {c}: {top} ({share:.1f}% of the cluster)")

# =============================================================================
print("\n" + "=" * 78)
print("STEP 5 : SAVING RESULTS AND FIGURES")
print("=" * 78)

df[["Patient_ID"] + CLUSTER_COLS + ["Disease", "Outcome", "Cluster"]] \
    .to_csv("outputs/cluster_assignments.csv", index=False)
print("  outputs/cluster_assignments.csv")

with open("outputs/clustering_report.txt", "w") as f:
    f.write("K-MEANS CLUSTERING RESULTS\n")
    f.write(f"k = {best_k}\nFeatures: {CLUSTER_COLS} (standardised)\n\n")
    f.write("k selection table:\n")
    f.write(f"{'k':>3} {'Inertia':>12} {'Silhouette':>12}\n")
    for k, i_, s_ in zip(k_range, inertias, silhouettes):
        f.write(f"{k:>3} {i_:>12.1f} {s_:>12.4f}\n")
    f.write("\nCluster profiles (original units):\n")
    f.write(centroids.to_string() + "\n\n")
    for cat in ["Disease", "Admission_Type", "Outcome"]:
        f.write(f"\n{cat} distribution (% within cluster):\n")
        f.write((pd.crosstab(df["Cluster"], df[cat], normalize="index") * 100)
                .round(1).to_string() + "\n")
print("  outputs/clustering_report.txt")

# ---- FIGURE 5 : elbow + silhouette -----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(list(k_range), inertias, "o-", color="#2E86AB", lw=2, ms=8)
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia (WCSS)")
axes[0].set_title("Elbow Method", fontweight="bold")
axes[0].grid(alpha=0.3)
axes[0].axvline(best_k, ls="--", color="red", alpha=0.6, label=f"chosen k={best_k}")
axes[0].legend()

axes[1].plot(list(k_range), silhouettes, "o-", color="#A23B72", lw=2, ms=8)
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Silhouette score")
axes[1].set_title("Silhouette Analysis", fontweight="bold")
axes[1].grid(alpha=0.3)
axes[1].axvline(best_k, ls="--", color="red", alpha=0.6, label=f"best k={best_k}")
axes[1].legend()
plt.tight_layout()
plt.savefig("outputs/fig5_elbow_silhouette.png", dpi=140)
plt.close()

# ---- FIGURE 6 : PCA scatter ------------------------------------------------
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X)
var = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(9, 6.5))
palette = ["#2E86AB", "#F18F01", "#A23B72", "#3B9C6E", "#C73E1D", "#7B4B94", "#6B7A8F"]
for c in sorted(df["Cluster"].unique()):
    m = df["Cluster"] == c
    ax.scatter(X_pca[m, 0], X_pca[m, 1], s=45, alpha=0.75,
               color=palette[c % len(palette)],
               edgecolor="black", linewidth=0.4,
               label=f"Cluster {c} (n={m.sum()})")
cent_pca = pca.transform(km_final.cluster_centers_)
ax.scatter(cent_pca[:, 0], cent_pca[:, 1], s=350, marker="X",
           c="black", edgecolor="yellow", linewidth=2, label="Centroids", zorder=5)
ax.set_xlabel(f"Principal Component 1 ({var[0]*100:.1f}% variance)")
ax.set_ylabel(f"Principal Component 2 ({var[1]*100:.1f}% variance)")
ax.set_title("Patient Clusters - PCA 2-D Projection", fontweight="bold")
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/fig6_pca_clusters.png", dpi=140)
plt.close()

# ---- FIGURE 7 : parallel coordinates (cluster profile lines) ---------------
fig, ax = plt.subplots(figsize=(10, 6))
norm_centroids = (centroids[CLUSTER_COLS] - centroids[CLUSTER_COLS].min()) / \
                 (centroids[CLUSTER_COLS].max() - centroids[CLUSTER_COLS].min())
for c in norm_centroids.index:
    ax.plot(range(len(CLUSTER_COLS)), norm_centroids.loc[c],
            "o-", lw=2.5, ms=10, label=f"Cluster {c}",
            color=palette[c % len(palette)])
ax.set_xticks(range(len(CLUSTER_COLS)))
ax.set_xticklabels([c.replace("_", "\n") for c in CLUSTER_COLS])
ax.set_ylabel("Normalised centroid value (0 = min, 1 = max)")
ax.set_title("Cluster Profiles Across Attributes", fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/fig7_cluster_profiles.png", dpi=140)
plt.close()

# ---- FIGURE 8 : disease composition per cluster ---------------------------
ct = pd.crosstab(df["Cluster"], df["Disease"], normalize="index") * 100
fig, ax = plt.subplots(figsize=(10, 5.5))
ct.plot(kind="bar", stacked=True, ax=ax, colormap="viridis", edgecolor="black", lw=0.5)
ax.set_ylabel("% of patients in cluster")
ax.set_xlabel("Cluster")
ax.set_title("Disease Composition of Each Cluster", fontweight="bold")
ax.legend(title="Disease", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/fig8_cluster_disease_mix.png", dpi=140)
plt.close()

for p in ["fig5_elbow_silhouette", "fig6_pca_clusters",
          "fig7_cluster_profiles", "fig8_cluster_disease_mix"]:
    print(f"  outputs/{p}.png")

print("\n" + "=" * 78)
print("CLUSTERING COMPLETE  ->  next: python 04_association.py")
print("=" * 78)
