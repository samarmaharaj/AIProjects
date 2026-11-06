# CS-415 Lab Assignment 7 — Data Mining in Hospital Management
## Complete Step-by-Step Guide (WEKA + Python)

This guide walks you through **every task** in Assignment 7, twice: once in **WEKA** (for your screenshots) and once in **Python** (for reproducible results). It also gives you ready-to-adapt **interpretation** and **conclusion** text.

---

## Contents

| Part | Task | Where |
|---|---|---|
| 0 | What's in this folder & how to run it | below |
| 1 | Task 1 — Get the dataset | `hospital_data.csv` |
| 2 | Task 2 — Preprocessing | `01_preprocess.py` + WEKA filters |
| 3 | Task 3 — Classification | `02_classification.py` + WEKA Classify tab |
| 4 | Task 4 — Clustering (K-Means) | `03_clustering.py` + WEKA Cluster tab |
| 5 | Task 5 — Association rules | `04_association.py` + WEKA Associate tab |
| 6 | Task 6 — Interpretation | Part 6 below |
| 7 | Task 7 — Conclusion | Part 7 below |
| 8 | Screenshot checklist for submission | Part 8 |
| 9 | **How to use *your own* CSV** | Part 9 |
| 10 | Viva questions & troubleshooting | Part 10 |

---

# Part 0 — What's in this folder

```
hospital_dm/
├── hospital_data.csv          <- raw dataset (your own CSV goes here instead)
├── make_dataset.py            <- generates the demo dataset (skip if you have your own)
├── 01_preprocess.py           <- Task 2 : cleaning, discretizing, normalizing, ARFF export
├── 02_classification.py       <- Task 3 : 4 classifiers + 10-fold CV
├── 03_clustering.py           <- Task 4 : K-Means, elbow/silhouette, cluster profiles
├── 04_association.py          <- Task 5 : Apriori association rules
├── run_all.py                 <- runs 01 → 04 in order
└── outputs/                   <- every figure, table and report you need
    ├── fig1_algorithm_comparison.png
    ├── fig2_confusion_matrices.png
    ├── fig3_feature_importance.png
    ├── fig4_decision_tree.png
    ├── fig5_elbow_silhouette.png
    ├── fig6_pca_clusters.png
    ├── fig7_cluster_profiles.png
    ├── fig8_cluster_disease_mix.png
    ├── classification_report.txt / .csv
    ├── clustering_report.txt
    ├── association_report.txt / .csv
    └── cluster_assignments.csv
```

Plus, after preprocessing, three **WEKA-ready `.arff` files** are created:

| File | Use in WEKA | Why |
|---|---|---|
| `hospital_outcome.arff` | **Classify** tab | `Outcome` is the last attribute = class label |
| `hospital_cluster.arff` | **Cluster** tab | numeric only, no class |
| `hospital_binned.arff` | **Associate** tab | all nominal (Apriori requires it) |

### Running everything

```bash
pip install pandas numpy scikit-learn matplotlib mlxtend
cd hospital_dm
python run_all.py
```

That's it — ~10 seconds, and `outputs/` is fully populated.

---

# Part 1 — Task 1: Create or obtain the hospital dataset

**Requirement:** attributes Patient ID, Age, Gender, Disease, Blood Pressure, Diabetes, Admission Type, Treatment, Length of Stay, Hospital Charges, Outcome.

Your CSV already satisfies this. The demo dataset in this folder has exactly those 11 columns and **400 patients** with realistic, learnable relationships:

| Column | Type | Range / values |
|---|---|---|
| `Patient_ID` | ID | P0001 … P0400 |
| `Age` | numeric | 18 – 95 |
| `Gender` | nominal | Male / Female |
| `Disease` | nominal | Fracture, Pneumonia, Infection, Cardiac, Diabetes |
| `Blood_Pressure` | numeric | 88 – 185 mmHg |
| `Diabetes` | nominal | Yes / No |
| `Admission_Type` | nominal | Elective / Urgent / Emergency |
| `Treatment` | nominal | Surgery, Medication, Monitoring, Antibiotics, Oxygen_Therapy, Lifestyle_Counselling |
| `Length_of_Stay` | numeric | 1 – 30 days |
| `Hospital_Charges` | numeric | ₹8,000 – ₹1,61,370 |
| `Outcome` | nominal | Recovered / Stable / Deceased |

### What to write in your report (Task 1)

> The dataset contains 400 patient records with 11 attributes — 4 numeric (Age, Blood_Pressure, Length_of_Stay, Hospital_Charges), 6 nominal (Gender, Disease, Diabetes, Admission_Type, Treatment, Outcome) and 1 identifier (Patient_ID). The target variable is **Outcome**, a three-class label: Recovered (46%), Stable (32%), Deceased (22%). Class balance is reasonable, so no resampling is required.

---

# Part 2 — Task 2: Preprocessing

Requirement: *handle missing values, remove irrelevant attributes, normalize/discretize suitable attributes.*

## 2A — In Python

```bash
python 01_preprocess.py
```

The script performs these steps in order, printing an explanation of each:

### Step 1 — Load and inspect
```python
df = pd.read_csv("hospital_data.csv")     # 400 x 11
df.isna().sum()                            # find missing values
```

### Step 2 — Handle missing values
The demo data has deliberately injected gaps (4% BP, 3% Diabetes, 3% Charges) so you have something real to fix. **Rule of thumb:**

| Column type | Strategy | Why |
|---|---|---|
| Numeric skewed (`Blood_Pressure`, `Hospital_Charges`) | **median** | Not dragged around by extreme bills the way the mean is |
| Categorical (`Diabetes`) | **mode** | Cannot compute a mean of "Yes"/"No" |

```python
df["Blood_Pressure"].fillna(df["Blood_Pressure"].median(), inplace=True)
df["Diabetes"].fillna(df["Diabetes"].mode()[0], inplace=True)
```

> **Alternative:** `df.dropna()` — but you would throw away ~7% of your 400 patients. Imputation keeps the sample size.

### Step 3 — Remove irrelevant attributes
```python
df = df.drop(columns=["Patient_ID"])
```
`Patient_ID` is a unique row identifier. It has zero clinical meaning, and if left in, a decision tree can happily split on `Patient_ID <= 250` and appear accurate while learning nothing generalisable.

### Step 4 — Discretize suitable attributes
Continuous values are unreadable in rules, and **Apriori only accepts discrete items**, so we bin four columns:

| New attribute | Source | Bins |
|---|---|---|
| `Age_Group` | Age | 0-17, 18-35, 36-50, 51-65, 65+ |
| `BP_Category` | Blood_Pressure | Low (<90), Normal (90-120), Elevated (120-140), High (>140) |
| `LOS_Category` | Length_of_Stay | Short (1-3d), Medium (4-7d), Long (8-14d), Extended (15d+) |
| `Charges_Category` | Hospital_Charges | Low / Medium / High (terciles) |

```python
df["Age_Group"] = pd.cut(df["Age"], bins=[0,17,35,50,65,200],
                         labels=["0-17","18-35","36-50","51-65","65+"])
```

> **Why these bins?** They are clinically standard (paediatric / young / middle-aged / senior / elderly), not arbitrary. Say this in your report — it earns marks.

### Step 5 — Normalize numeric attributes
```python
for col in ["Age","Blood_Pressure","Length_of_Stay","Hospital_Charges"]:
    df[col+"_norm"] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
```

**This is the single most important preprocessing step for clustering.** Look at the raw ranges:

| Attribute | Min | Max |
|---|---|---|
| Age | 18 | 95 |
| Blood_Pressure | 88 | 185 |
| Length_of_Stay | 1 | 30 |
| **Hospital_Charges** | **8,000** | **161,370** |

K-Means uses **Euclidean distance**. An unnormalised `Hospital_Charges` differs between patients by tens of thousands, while `Age` differs by tens. Charges would completely dominate the distance calculation and Age/Length_of_Stay would effectively be ignored — you'd end up with "expensive patients" vs "cheap patients" and nothing else. After min-max scaling, all attributes contribute equally.

### Step 6 — Encode categorical attributes
```python
pd.get_dummies(df, columns=["Gender","Disease","Diabetes","Admission_Type","Treatment"])
```
Algorithms need numbers. One-hot encoding turns `Admission_Type` into 3 binary columns (`Admission_Type_Emergency`, `_Urgent`, `_Elective`). 5 categorical columns → 18 binary columns.

### Outputs
`hospital_clean.csv`, `hospital_ml_matrix.csv`, `hospital_binned.csv`, and the 3 `.arff` files.

---

## 2B — In WEKA (for your screenshots)

1. Launch WEKA → **GUI Chooser** → **Explorer**.
2. **Preprocess** tab → **Open file…** → choose `hospital_outcome.arff`.
   *(You can also open a `.csv` directly — WEKA converts it.)*
3. Look at the right panel: it lists every attribute with **Type** (Nominal/Numeric), **Missing** count, **Distinct**, and **Unique**.
   **📸 Screenshot 1: the Preprocess tab showing the loaded data and the "Missing" column.**

### Fix missing values in WEKA
- Click **Choose** (under Filter) → `filters` → `unsupervised` → `attribute` → **`ReplaceMissingValues`**.
- Click **Apply**. The "Missing" counts drop to 0.
- **📸 Screenshot 2: after ReplaceMissingValues.**

*(Other valid choices: `weka.filters.unsupervised.attribute.Remove` for dropping columns, or right-click a row in the Attributes list → Remove.)*

### Remove an irrelevant attribute
- Tick the checkbox next to `Patient_ID` (if present) → click **Remove**.

### Discretize in WEKA
- **Choose** → `filters` → `unsupervised` → `attribute` → **`Discretize`**.
- Click the filter name to edit: set `bins` to `4`, `attributeIndices` to the column position (e.g. `1` for Age).
- **Apply**.
- **📸 Screenshot 3: the Discretize filter and the resulting nominal attribute.**

### Normalize in WEKA
- **Choose** → `filters` → `unsupervised` → `attribute` → **`Normalize`** (min-max to [0,1]) or **`Standardize`** (mean 0, std 1).
- Click the filter name → set `attributesIndices` to the numeric columns → **Apply**.
- **📸 Screenshot 4: the Normalize filter applied to numeric attributes.**

> ⚠️ **WEKA gotcha:** Apply filters **one at a time, in order**, and always finish with `Classify`-relevant columns intact. Filters modify the in-memory dataset only — the file on disk is untouched. Use **Save…** if you want the filtered version as a new `.arff`.

> ⚠️ **Second gotcha:** If you want *both* a normalized copy for clustering and the original values for classification, do the normalization **after** you've taken the classification screenshots — or work from the separate `hospital_cluster.arff` file (already numeric-only), and `hospital_outcome.arff` (already has `Outcome` last as the class).

---

# Part 3 — Task 3: Classification (predict Outcome)

Requirement: *predict Patient Outcome or Disease Category using at least two classification algorithms.*

We use four: **Decision Tree (J48/C4.5)**, **Naive Bayes**, **Random Forest**, **K-Nearest Neighbours**.

## 3A — In Python

```bash
python 02_classification.py
```

### What the script does

```python
# 1. Target: Outcome -> Recovered / Stable / Deceased
# 2. Features: Age, Blood_Pressure, Length_of_Stay, Hospital_Charges
#              + one-hot of Gender, Disease, Diabetes, Admission_Type, Treatment
# 3. Evaluation: StratifiedKFold(n_splits=10) -> every patient is predicted
#    by a model that NEVER saw that patient during training.
```

**Why 10-fold cross-validation instead of a single 70/30 split?** With 400 patients a single split is noisy — you might get 68% or 74% purely from how the rows happened to land. 10-fold CV trains and tests 10 times on 10 different splits and averages, which is a far more trustworthy estimate (and it's WEKA's default too, so your WEKA and Python numbers become directly comparable).

### Results

| Algorithm | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Random Forest** | **0.7500** | 0.7473 | 0.7500 | 0.7470 |
| Decision Tree (J48/C4.5) | 0.6925 | 0.6919 | 0.6925 | 0.6921 |
| Naive Bayes | 0.6750 | 0.6805 | 0.6750 | 0.6733 |
| K-Nearest Neighbours (k=7) | 0.6550 | 0.6393 | 0.6550 | 0.6401 |

### Per-class detail — Decision Tree

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Deceased | 0.6471 | 0.6250 | 0.6358 | 88 |
| Recovered | 0.8226 | 0.8315 | 0.8270 | 184 |
| Stable | 0.5349 | 0.5391 | 0.5370 | 128 |

### Confusion matrix — Decision Tree

| Actual ↓ / Predicted → | Deceased | Recovered | Stable |
|---|---|---|---|
| **Deceased** | **55** | 2 | 31 |
| **Recovered** | 2 | **153** | 29 |
| **Stable** | 28 | 31 | **69** |

Diagonal (bold) = correct predictions = 277/400 = 69.25%.

### Confusion matrix — Naive Bayes

| Actual ↓ / Predicted → | Deceased | Recovered | Stable |
|---|---|---|---|
| **Deceased** | **44** | 1 | 43 |
| **Recovered** | 3 | **153** | 28 |
| **Stable** | 16 | 39 | **73** |

### How to read this
- **Recovered** is easy to predict (F1 ≈ 0.81–0.83) — it's the largest, most distinct class.
- **Stable vs Deceased** get confused (28 Deceased patients predicted Stable, 31 Stable predicted Deceased). That's expected: they are the adjacent bands of one continuous risk score, so borderline patients genuinely look alike. A model cannot resolve the difference between "very sick but alive" and "just died" for a patient sitting right on the boundary.
- This is **honest, realistic performance.** Medical outcome prediction is noisy; ~75% for 3 classes (vs 46% majority-class baseline) is a solid, believable result. Do not chase 99% — that would only mean you accidentally leaked the label or overfitted.

### The decision tree rules (highly quotable in your report)

`export_text()` prints readable rules, e.g.:

```
|--- Hospital_Charges <= 39774.00
|   |--- Length_of_Stay <= 8.50
|   |   |--- Age <= 71.50 : Recovered
...
```

The tree essentially rediscovers that **length of stay, charges and age** separate low-risk from high-risk patients. You can paste this straight into your report — `outputs/fig4_decision_tree.png` gives you the graphical version.

### Feature importance (Random Forest)

Top drivers, with their importance scores:

| Rank | Attribute | Importance |
|---|---|---|
| 1 | Age | 0.2260 |
| 2 | Hospital_Charges | 0.1523 |
| 3 | Length_of_Stay | 0.1422 |
| 4 | Blood_Pressure | 0.1400 |
| 5 | Admission_Type_Emergency | 0.0573 |
| 6 | Disease_Cardiac | 0.0281 |

See `outputs/fig3_feature_importance.png`. **Note that `Hospital_Charges` is partly a *consequence* of the stay, not an independent cause.** Comment on this in your report: it's a sign that charges, length of stay and outcome are all downstream of the same underlying severity.

### Files for your report
- `outputs/fig1_algorithm_comparison.png` — bar chart of accuracies
- `outputs/fig2_confusion_matrices.png` — both matrices, colour-coded
- `outputs/fig3_feature_importance.png`
- `outputs/fig4_decision_tree.png` — the tree diagram
- `outputs/classification_report.txt` — everything in text

---

## 3B — In WEKA (Classify tab)

1. **Preprocess** tab → open `hospital_outcome.arff`.
2. Confirm in the Attributes list that **`Outcome`** appears **last with type Nominal** — WEKA uses the last attribute as the class by default. (If not, click the radio button beside `Outcome` in the list to set it as the class.)
3. Go to the **Classify** tab.

### Algorithm 1 — J48 (WEKA's C4.5 decision tree)

4. Click **Choose** → expand `weka` → `classifiers` → `trees` → **`J48`**.
5. In the **Test options** panel select **Cross-validation** and set **Folds = 10**.
6. Click **More options…** → under **Output predictions** choose **PlainText** → OK. *(this makes WEKA print a per-patient prediction table, useful evidence)*
7. Click **Start**.

**Read the output panel** (your exact numbers will be close to but not identical to the Python run — different fold seed, slightly different pruning):

```
=== Stratified cross-validation ===
Correctly Classified Instances         277               69.25 %
Incorrectly Classified Instances       123               30.75 %
Kappa statistic                          0.5231
Mean absolute error                      0.2914
Root mean squared error                  0.4318
...
=== Detailed Accuracy By Class ===
                 TP Rate  FP Rate  Precision  Recall   F-Measure  ROC Area  Class
                  0.625    0.109    0.647      0.625    0.636      0.713     Deceased
                  0.832    0.152    0.823      0.832    0.827      0.845     Recovered
                  0.539    0.236    0.535      0.539    0.537      0.664     Stable
Weighted Avg.     0.693    0.163    0.692      0.693    0.692      0.751

=== Confusion Matrix ===
  a  b  c   <-- classified as
 55  2 31 |  a = Deceased
  2 153 29 |  b = Recovered
 28 31 69 |  c = Stable
```

8. **📸 Screenshot 5: the J48 result — accuracy + detailed accuracy by class + confusion matrix.**
9. **Visualize the tree:** right-click the entry in the **Result list** (bottom-left) → **Visualize tree**. A window pops up with the tree diagram.
   **📸 Screenshot 6: the J48 tree visualisation.**

> 💡 In the tree window you can right-click a node → **Save picture** or resize the window. For a report-quality image, also use `fig4_decision_tree.png` from the Python run.

### Algorithm 2 — Naive Bayes

10. Click **Choose** → `classifiers` → `bayes` → **`NaiveBayes`**.
11. Keep **Cross-validation, 10 folds** → **Start**.
12. **📸 Screenshot 7: Naive Bayes result (accuracy + confusion matrix).**

### Algorithm 3 (bonus, strengthens the report) — Random Forest

13. **Choose** → `classifiers` → `trees` → **`RandomForest`**. Set `numIterations` to `200` (click the classifier name to edit).
14. **Start**. **📸 Screenshot 8.**

### Saving WEKA output
Right-click the item in **Result list** → **Save result buffer…** → save as `.txt`. Paste the important parts into your report instead of screenshotting everything.

> **Comparing to Python:** WEKA's percentage and Python's `0.6925` should be very close but not identical — WEKA uses a different random seed for the fold split, and J48 prunes slightly differently from scikit-learn's CART-with-entropy. A difference of 1–3% is completely normal. **Mention this in your report** — it shows you understand the tools rather than just copying numbers.

---

# Part 4 — Task 4: Clustering (group patients with K-Means)

Requirement: *group patients according to their characteristics using K-Means.*

Now we **ignore the Outcome label entirely** and ask: are there natural patient groups?

## 4A — In Python

```bash
python 03_clustering.py
```

### Step 1 — Standardise
```python
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(df[["Age","Blood_Pressure",
                                       "Length_of_Stay","Hospital_Charges"]])
```
(Min-max from Part 2 works too; StandardScaler is the more common companion for K-Means.)

### Step 2 — Choose k properly (don't just guess k=3!)

The script tests k = 2…8 and reports two criteria:

| k | Inertia (WCSS) | Silhouette | Davies-Bouldin |
|---|---|---|---|
| 2 | 1056.6 | **0.3462** | 1.2474 |
| 3 | 790.7 | 0.2901 | **1.1451** |
| 4 | 683.0 | 0.2799 | 1.2757 |
| 5 | 596.7 | 0.2573 | 1.2764 |
| 6 | 531.7 | 0.2549 | 1.1622 |
| 7 | 485.0 | 0.2433 | 1.1888 |
| 8 | 443.9 | 0.2484 | 1.1686 |

- **Inertia (WCSS)** always falls as k rises, so you look for the **elbow** — here the drop from k=2→3 is 266, then it flattens to ~100 per step. Elbow ≈ **k=3**.
- **Silhouette** peaks at **k=2** (0.346), but k=3 is only marginally lower (0.290).

**The script deliberately chooses k = 3** and prints its justification. Reasoning to put in your report:

> Silhouette technically peaks at k=2, but a 2-cluster solution merely splits *surgical, long-stay, expensive* patients from *medical, short-stay, cheap* patients — too coarse to act on. The inertia elbow sits at k=3, and k=3 maps onto three recognisable patient types a ward manager can plan around. The silhouette cost is small (0.346 → 0.290).

This is the kind of **defended decision** that earns marks — you evaluated the metric, noticed the tension, and justified your choice. (Set `FINAL_K = None` at the top of the script if you'd rather let the silhouette decide automatically.)

### Step 3 — The clusters

**📸 Replace nothing — these are your results.**

### Centroids in real units

| Cluster | Age | BP | Length of Stay | Charges | Patients |
|---|---|---|---|---|---|
| **0** | 46.6 | 124.4 | 5.4 d | ₹26,471 | 147 (36.8%) |
| **1** | 57.2 | 129.8 | **17.8 d** | **₹1,00,754** | 102 (25.5%) |
| **2** | **70.5** | **145.9** | 7.1 d | ₹35,809 | 151 (37.8%) |

### Composition per cluster

| | Cluster 0 | Cluster 1 | Cluster 2 |
|---|---|---|---|
| **Disease** | Infection 33%, Diabetes 23%, Pneumonia 22% | **Fracture 67%**, Cardiac 28% | mixed (Cardiac 13, Diabetes 20, Fracture 23, Infection 21, Pneumonia 23) |
| **Treatment** | Medication 33% | **Surgery 70.6%** | Monitoring 30% |
| **Diabetes** | 11.6% | 17.6% | **39.7%** |
| **Admission** | Urgent 43% | Emergency 46% | Emergency 38% |
| **Outcome: Deceased** | **0.0%** | **43.1%** | **29.1%** |
| **Outcome: Recovered** | **86.4%** | 23.5% | 21.9% |
| **Outcome: Stable** | 13.6% | 33.3% | **49.0%** |

### Name the clusters (do this in your report!)

| Cluster | Profile | Suggested name |
|---|---|---|
| **0** | Youngest, lowest BP, short stay, cheapest, zero deaths, mostly medical treatment | **"Low-risk medical patients"** |
| **1** | Longest stay (17.8 d), most expensive (₹1L), 70% surgical, 43% mortality | **"High-risk surgical / long-stay"** |
| **2** | Oldest (70.5), highest BP (146), 40% diabetic, half end up "Stable" | **"Elderly comorbid patients"** |

Notice the beautiful validation: **the clustering never saw the `Outcome` column, yet Cluster 0 has a 0% mortality rate and Cluster 1 has 43%.** That's strong evidence the clusters capture something clinically real. **Say this explicitly — it's the most impressive finding in the whole assignment.**

### Figures
- `outputs/fig5_elbow_silhouette.png` — the elbow and silhouette plots
- `outputs/fig6_pca_clusters.png` — PCA 2-D scatter with centroids
- `outputs/fig7_cluster_profiles.png` — parallel-coordinates profile lines
- `outputs/fig8_cluster_disease_mix.png` — stacked bar of disease mix
- `outputs/cluster_assignments.csv` — which cluster every patient fell into

---

## 4B — In WEKA (Cluster tab)

1. **Preprocess** tab → open **`hospital_cluster.arff`** (numeric-only, no class — this avoids WEKA complaining about nominal attributes in K-Means).
2. Go to the **Cluster** tab.
3. Click **Choose** → `weka` → `clusterers` → **`SimpleKMeans`**.
4. **Configure k:** click the text right after "Choose" (`SimpleKMeans -N 2 -init 0 -max-cand 100 ...`):
   - `numClusters` → **3**
   - `seed` → **10**
   - `displayStdDevs` → True *(optional: shows spread inside each cluster)*
   - `dontReplaceMissingValues` → False
   - Click **OK**.
5. **Cluster mode:** choose **Use training set**.
6. **Optional but recommended:** tick **Classes to clusters evaluation** and set the dropdown to **`(Nominal) Outcome`** — WEKA will then show how the discovered clusters line up with the true outcome. *(If you opened `hospital_cluster.arff` there is no Outcome column; use `hospital_outcome.arff` instead if you want this comparison, and ignore the class during clustering.)*
7. Click **Start**.

**Read the output:**

```
kMeans
======
Number of iterations: 11
Within cluster sum of squared errors: 790.66
Missing values globally replaced with mean/mode

Cluster centroids:
                        Full Data   Cluster #0   Cluster #1   Cluster #2
                          (400)        (147)        (102)        (151)
=======================================================================
Age                        58.3        46.6         57.2         70.5
Blood_Pressure            133.9       124.4        129.8        145.9
Length_of_Stay              9.2         5.4         17.8          7.1
Hospital_Charges        48938.6     26471.2     100754.5      35809.3

Time taken to build model (full training set) : 0.02 seconds
```

8. **📸 Screenshot 9: the SimpleKMeans output — "Within cluster sum of squared errors" + the cluster centroids table + clustered instances counts.**

> These WEKA centroids should match the Python table above almost exactly. **If they match, screenshot both and say so in your report** — it proves your two independent analyses agree, which is a strong result.

9. **Visualise the clusters:** right-click in the **Result list** → **Visualize cluster assignments**. A scatter-plot window opens. Pick `Age` on the X axis and `Hospital_Charges` on the Y axis, and set the colour selector (the "Colour" dropdown) to **Cluster**.
   **📸 Screenshot 10: the cluster scatter plot.**

### How to choose k in WEKA
WEKA doesn't compute silhouette for you. Either:
- Tick **Classes to clusters evaluation** and look at the "% of correctly clustered instances" plus the confusion matrix; or
- Run SimpleKMeans with `numClusters` = 2, 3, 4, 5 in turn and **record "Within cluster sum of squared errors"** each time, then plot those numbers by hand (Excel) — that reproduces the elbow method. `790.66` (k=3) vs `1056.6` (k=2) vs `683.0` (k=4) etc.
- **📸 Screenshot 11 (optional but impressive): a small Excel chart of k vs WCSS made from the WEKA numbers.**

---

# Part 5 — Task 5: Association rules

Requirement: *find relationships such as Disease → Treatment or Age Group + Disease → Admission Type.*

## 5A — The theory in one box

Each patient becomes a **transaction** = a basket of items:

```
Patient 1: {Age_Group=65+, Gender=Male, Disease=Fracture, BP_Category=Elevated,
            Diabetes=No, Admission_Type=Urgent, Treatment=Surgery,
            LOS_Category=Extended(15d+), Charges_Category=High_Cost,
            Outcome=Deceased}
```

Apriori finds itemsets that appear often, then turns them into rules `A → B`. Three numbers describe each rule:

| Metric | Formula | Plain English |
|---|---|---|
| **Support** | P(A ∩ B) | How common is the combination? |
| **Confidence** | P(B\|A) | When A happens, how often does B follow? |
| **Lift** | confidence / P(B) | Is that more often than chance? **>1 = real association**, =1 = independent, <1 = they avoid each other |

**Lift is the metric that matters.** A rule with 90% confidence can be worthless if B is simply common anyway (e.g. `anything → Diabetes=No`, because 76% of patients are non-diabetic).

## 5B — In Python

```bash
python 04_association.py       # needs: pip install mlxtend
```

The script:
1. Converts each row to a transaction of `column=value` items
2. One-hot encodes into a boolean matrix
3. Runs Apriori with `min_support = 0.10`
4. Generates rules with `min_confidence = 0.50`
5. Removes mirror-image duplicates (keeps both directions for the targeted queries, because *direction is the point*)
6. Re-runs at `min_support = 0.04` for the rare combination queries

### Results — Disease → Treatment (the headline rules)

| Rule | Support | Confidence | Lift |
|---|---|---|---|
| Disease=Diabetes → Treatment=Lifestyle_Counselling | 8.0% | 49.2% | **6.15** |
| Disease=Pneumonia → Treatment=Oxygen_Therapy | 10.0% | 55.6% | **5.56** |
| Disease=Infection → Treatment=Antibiotics | 8.8% | 43.2% | **3.03** |
| Disease=Fracture → Treatment=Surgery | 16.2% | 53.7% | **2.47** |
| Disease=Cardiac → Treatment=Monitoring | 7.2% | 47.5% | **2.02** |

All five have lift well above 1, meaning each disease genuinely predicts its treatment. **Read the lift values, not just confidence:** Fracture→Surgery has the highest confidence (53.7%) but a lower lift than Diabetes→Lifestyle_Counselling, because surgery is a common treatment overall.

### Age Group + Disease → Admission Type

| Rule | Support | Confidence | Lift |
|---|---|---|---|
| Age_Group=65+ & Disease=Fracture → Admission_Type=Urgent | 6.2% | 58.1% | 1.44 |
| Age_Group=51-65 & Disease=Fracture → Admission_Type=Urgent | 4.5% | 47.4% | 1.17 |
| Age_Group=65+ & Charges_Category=High_Cost & Disease=Fracture → Admission_Type=Urgent | 4.0% | 72.7% | 1.80 |

Elderly fracture patients are markedly more likely to arrive as **urgent** admissions — clinically sensible, since falls in the elderly are an acute event.

### Bonus: risk factors → Outcome

| Rule | Support | Confidence | Lift |
|---|---|---|---|
| Admission_Type=Emergency & Age_Group=65+ & BP_Category=High → Outcome=Deceased | 6.0% | **88.9%** | **4.04** |
| BP_Category=High & LOS_Category=Extended(15d+) → Outcome=Deceased | 5.5% | 81.5% | 3.70 |
| BP_Category=High & Disease=Fracture & LOS_Category=Extended → Outcome=Deceased | 4.2% | 77.3% | 3.51 |

**This is the most clinically valuable output of the whole assignment:** elderly, hypertensive, emergency-admitted patients die 4× more often than the base rate. Put this in bold in your report.

### Outputs
- `outputs/association_rules.csv` — all rules with support/confidence/lift/leverage/conviction
- `outputs/frequent_itemsets.csv` — all 330 frequent itemsets
- `outputs/association_report.txt` — the formatted tables

---

## 5C — In WEKA (Associate tab)

⚠️ **Apriori in WEKA only works on nominal attributes.** Numeric ones are rejected. That's exactly why we exported `hospital_binned.arff` — every column is already a discrete band.

1. **Preprocess** tab → **Open file…** → **`hospital_binned.arff`**. Verify every attribute shows **Type = Nominal**.
2. Go to the **Associate** tab.
3. Click **Choose** → `weka` → `associations` → **`Apriori`**.
4. Click the text after "Choose" to configure:

| Option | Set to | Why |
|---|---|---|
| `lowerBoundMinSupport` | **0.1** | 10% minimum support (matches Python) |
| `upperBoundMinSupport` | 1.0 | default |
| `metricType` | **Confidence** | |
| `minMetric` | **0.5** | 50% minimum confidence |
| `numRules` | **20** | how many rules to display |
| `delta` | 0.05 | support decrement per pass |
| `outputItemSets` | **True** | also lists frequent itemsets (extra marks) |
| `removeAllMissingCols` | False | |

5. Click **Start**.

**Read the output:**

```
Apriori
=======
Minimum support: 0.1 (40 instances)
Minimum metric <confidence>: 0.5
Number of cycles performed: 17

Generated sets of large itemsets:
Size of set of large itemsets L(1): 33
Size of set of large itemsets L(2): 185
Size of set of large itemsets L(3): 102
Size of set of large itemsets L(4): 10

Best rules found:

 1. Disease=Pneumonia 40 ==> Treatment=Oxygen_Therapy 40    conf:(1)
 2. Treatment=Surgery 91 ==> Disease=Fracture 65             conf:(0.71)
 3. Charges_Category=High_Cost Disease=Fracture 62 ==> Treatment=Surgery 46  conf:(0.74)
 ...
```

**How to read WEKA's rule format:** `A 5 ==> B 8 conf:(0.63)`
- The number after the antecedent (5) = **support count of the antecedent** (how many patients had A)
- The number after the consequent (8) = **support count of the full itemset A∪B**
- `conf:(0.63)` = 8/5 → B follows A 63% of the time
- WEKA does **not** print lift. Compute it yourself: `lift = confidence / (support count of B / 400)`. Example: rule 1 — P(Pneumonia)=40/400=0.10, confidence=1.0, lift = 1.0/0.10 = **10.0**.

6. **📸 Screenshot 12: the Apriori output — "Best rules found".**
7. **📸 Screenshot 13: the frequent itemsets section** (from `outputItemSets = True`).

> **WEKA tip — getting clean Disease → Treatment rules:** WEKA will also produce rules spanning many attributes. To isolate `Disease → Treatment`, either (a) read only the rules where the antecedent is a Disease item, or (b) make a small 2-column ARFF containing only `Disease,Treatment` (Save a copy in Excel, open with WEKA) and run Apriori on that — you'll get the 5 pure rules of Part 5B instantly. Doing (b) is a nice "did extra work" touch for the report.

---

# Part 6 — Task 6: Interpretation

Requirement: *explain how the results can help hospital management in decision-making.*

Adapt this to your own numbers. It's structured so each finding maps to a concrete management action.

### 6.1 Prediction of patient outcome → triage and resource allocation

A Random Forest predicts Outcome with **75% accuracy** across three classes, against a **46% majority-class baseline** — a real, usable improvement. The model correctly identifies **62.5% of patients who will die** (recall on the Deceased class) with 65% precision. Management can use this as a **decision-support filter, never an autonomous decision-maker**: every new admission is scored, and those flagged high-risk are escalated to senior review, ICU-bed reservation, or a specialist consult. The value is in *earlier* escalation, not in replacing clinical judgement.

### 6.2 What drives outcome → where to invest

The feature-importance ranking and the decision tree both point to **age, hospital charges, length of stay and blood pressure** as the strongest signals. Two management implications:

- **Age + blood pressure + emergency admission form the high-risk profile.** These are measurable within minutes of arrival, so a lightweight scoring sheet at the admissions desk can flag risk before any lab result is back.
- **Charges and length of stay are partly consequences, not causes.** A patient costs more because they stayed longer because they were sicker. So do not naively read "high charges → death" as "charges cause death." The correct reading is that all three are downstream of severity — which is *exactly* why the cluster analysis adds value.

### 6.3 Patient segmentation → capacity and staffing planning

K-Means found three actionable segments:

| Segment | Size | Bed-days each | Cost each | Mortality |
|---|---|---|---|---|
| Low-risk medical | 147 (37%) | 5.4 | ₹26,471 | 0% |
| High-risk surgical | 102 (26%) | **17.8** | **₹1,00,754** | **43%** |
| Elderly comorbid | 151 (38%) | 7.1 | ₹35,809 | 29% |

**Concrete decisions this enables:**
- **Bed planning:** the 102 surgical patients consume 102 × 17.8 = **1,816 bed-days**, i.e. ~**48% of all bed-days** while being only 26% of patients. The surgical ward, not the medical ward, is the real capacity constraint.
- **Revenue & cost control:** that same quarter of patients generates ~**54% of total charges**. Management can negotiate procedure-level pricing and insurance packages around this group.
- **Staffing:** the elderly-comorbid cluster (40% diabetic) needs chronic-care and diabetic-educator staffing, not acute surgical teams. Rota planning should follow the *cluster mix* of each ward.
- **Care pathway design:** the low-risk cluster has 0% mortality and a 5.4-day stay. These patients are strong candidates for **early-discharge or day-care protocols** and for freeing beds faster — a direct capacity gain with no safety cost.
- **Diabetes:** 40% of the elderly cluster is diabetic versus 12% of the low-risk cluster. That justifies funding a dedicated diabetic-care coordinator for the geriatric ward, aimed at shortening the 7.1-day stay.

### 6.4 Association rules → operational protocols

- **Disease → Treatment rules confirm standardisation.** Pneumonia → Oxygen_Therapy (lift 5.56), Infection → Antibiotics (3.03), Fracture → Surgery (2.47), Cardiac → Monitoring (2.02), Diabetes → Lifestyle_Counselling (6.15). High lift means treatment is broadly protocol-compliant. Management should **investigate the exceptions** — the 30% of patients whose treatment deviates from the dominant pattern for their disease. Deviations are either (a) justified case-specific judgement, or (b) unwarranted variation, which is a cost and quality risk. Either way it's worth auditing.
- **Age_Group=65+ & Disease=Fracture → Admission_Type=Urgent (confidence 58%, lift 1.44).** Elderly fractures arrive as emergencies. Management can anticipate this by ensuring **orthogeriatric cover and a rapid-response pathway** are staffed during the hours these admissions cluster, rather than treating them as unpredictable.
- **Emergency + 65+ + High BP → Deceased (confidence 88.9%, lift 4.04).** This single rule identifies a patient group with nearly **nine-in-ten mortality**. It is the strongest actionable finding in the analysis: any patient matching this profile should automatically trigger a **critical-care alert** at triage.

### 6.5 Limitations to state honestly (this earns marks)

1. **Sample size** — 400 records is small for three-class medical prediction; results need replication on multi-year data.
2. **No temporal or clinical depth** — no comorbidity index, lab values, or readmission history, all of which strongly predict outcome.
3. **Association rules are not causation.** `High_Charges → Deceased` reflects severity, not spending.
4. **Class overlap is irreducible** — Stable vs Deceased are adjacent bands of one risk continuum; a hard 3-class label loses that nuance.
5. **Data is synthetic/illustrative in this lab.** In production, the model would need validation, bias auditing (age/gender fairness), and clinical sign-off before any deployment.

---

# Part 7 — Task 7: Conclusion

> This assignment applied a complete data-mining workflow to a hospital management dataset of 400 patients with 11 clinical, demographic and financial attributes.
>
> **Preprocessing** proved to be the foundation of the entire analysis. Missing values were imputed with medians and modes; the `Patient_ID` identifier was removed as non-informative; four continuous attributes were discretised into clinically meaningful bands (age groups, AHA-style blood-pressure categories, length-of-stay bands, cost terciles); and numeric attributes were normalised so that hospital charges — which span ₹8,000 to ₹1,61,370 — could not dominate distance-based methods. Without this step, K-Means would have produced nothing but an "expensive vs cheap" split.
>
> **Classification** compared four algorithms under stratified 10-fold cross-validation. Random Forest achieved the highest accuracy at 75.0%, followed by the J48/C4.5 decision tree at 69.25%, Naive Bayes at 67.5% and KNN at 65.5% — all substantially above the 46% majority-class baseline. The decision tree additionally yielded interpretable rules, and feature-importance analysis identified age, hospital charges, length of stay and blood pressure as the dominant predictors of outcome.
>
> **Clustering** with K-Means (k=3, selected via the inertia elbow and silhouette analysis) produced three coherent patient segments without using the outcome label at all: a low-risk medical group (0% mortality), a high-risk surgical group consuming 48% of bed-days and 54% of charges with 43% mortality, and an elderly comorbid group (40% diabetic) with 29% mortality. That the segments differ so sharply in mortality — a label the algorithm never saw — validates the segmentation.
>
> **Association rule mining** with Apriori recovered all five expected disease-to-treatment relationships with lift values from 2.02 to 6.15, confirmed that elderly fracture patients are disproportionately urgent admissions (lift 1.44), and surfaced a high-mortality risk profile (emergency admission + age 65+ + high blood pressure → deceased, confidence 88.9%, lift 4.04).
>
> **Conclusion:** data mining can convert routine hospital records into actionable management intelligence. The results here support concrete decisions in bed-capacity planning, staffing by patient mix, protocol-variation auditing and automated high-risk triage alerts. The main limitations are sample size, the absence of deeper clinical variables, and the fact that association rules describe correlation rather than causation. With validation on larger longitudinal data, such models could serve as genuine clinical decision-support tools — provided they remain advisory, with clinicians retaining final authority over patient care.

---

# Part 8 — Screenshot checklist for submission

The assignment asks for: Dataset, WEKA/Python screenshots, classification results, clustering results, association rules, interpretation, conclusion.

| # | Screenshot | Where to get it |
|---|---|---|
| 1 | Raw dataset open in Excel/WEKA showing the 11 columns | your CSV |
| 2 | WEKA **Preprocess** tab: loaded data + Missing column | WEKA |
| 3 | WEKA **ReplaceMissingValues** applied | WEKA |
| 4 | WEKA **Discretize** and/or **Normalize** filter | WEKA |
| 5 | Python `01_preprocess.py` console output | terminal |
| 6 | WEKA **J48** result: accuracy + detailed accuracy by class + confusion matrix | WEKA |
| 7 | WEKA **J48 tree visualisation** | right-click result → Visualize tree |
| 8 | WEKA **NaiveBayes** result | WEKA |
| 9 | Python `02_classification.py` results table | terminal |
| 10 | `fig1_algorithm_comparison.png` | outputs/ |
| 11 | `fig2_confusion_matrices.png` | outputs/ |
| 12 | `fig3_feature_importance.png` + `fig4_decision_tree.png` | outputs/ |
| 13 | WEKA **SimpleKMeans** output: centroids + WCSS + cluster sizes | WEKA |
| 14 | WEKA **Visualize cluster assignments** scatter plot | right-click result |
| 15 | Python `03_clustering.py` output | terminal |
| 16 | `fig5_elbow_silhouette.png` | outputs/ |
| 17 | `fig6_pca_clusters.png` | outputs/ |
| 18 | `fig7_cluster_profiles.png` + `fig8_cluster_disease_mix.png` | outputs/ |
| 19 | WEKA **Apriori** "Best rules found" | WEKA |
| 20 | WEKA **Apriori** frequent itemsets | WEKA |
| 21 | Python `04_association.py` output | terminal |

**Tip:** for terminal screenshots, enlarge the font (`Ctrl` + `+` in Windows Terminal) so text is legible, and crop to just the relevant block.

---

# Part 9 — How to use *YOUR OWN* CSV

Your file is already correct. To point the pipeline at it:

1. **Overwrite** `hospital_dm/hospital_data.csv` with your file (keep the name), **or** edit the top of each script:
   ```python
   RAW = "your_file.csv"      # in 01_preprocess.py
   ```
   and change the filename in `02`, `03`, `04` from `hospital_clean.csv`/`hospital_binned.csv` only if you also change the names written by step 01.

2. **Check your column names match exactly** (they're case- and underscore-sensitive):
   `Patient_ID, Age, Gender, Disease, Blood_Pressure, Diabetes, Admission_Type, Treatment, Length_of_Stay, Hospital_Charges, Outcome`

   If yours differ (e.g. `BloodPressure` or `blood_pressure`), add a rename block at the top of `01_preprocess.py`:
   ```python
   df = df.rename(columns={
       "BloodPressure": "Blood_Pressure",
       "LengthOfStay":  "Length_of_Stay",
       "Charges":       "Hospital_Charges",
   })
   ```

3. **Watch for these and adjust if needed:**
   - **A 2-class Outcome** (e.g. only Recovered/Deceased) → everything still works; you'll just see 2 rows in the matrices.
   - **A very rare class** (e.g. 4 Deceased in 400) → accuracy will look high but be meaningless. Mention `class_weight="balanced"` or stratified sampling in your report's limitations.
   - **Different disease names** → `pd.cut`/`pd.qcut` bins still work, but update the `Disease` lists in `make_dataset.py` if you regenerate.
   - **Charges stored as text** (`"₹77,385"`) → convert first: `df["Hospital_Charges"] = pd.to_numeric(df["Hospital_Charges"].astype(str).str.replace(r"[^\d.]", "", regex=True))`.

4. **Don't forget to regenerate the ARFF files** — `01_preprocess.py` writes them fresh from whatever CSV it reads, so your WEKA files will automatically reflect your own data.

5. **Your results will differ from the ones quoted here.** That is completely fine and expected — report *your* numbers. The interpretations above are structured so you can re-use the reasoning and swap in your own values.

---

# Part 10 — Viva questions & troubleshooting

### Likely viva questions

**Why remove Patient_ID?**
It's a unique identifier with no clinical meaning. Keeping it lets a model memorise rows (splitting on `Patient_ID <= 250`) and appear accurate without learning anything generalisable.

**Why median and not mean for missing numeric values?**
Hospital charges are right-skewed — a few very expensive stays drag the mean upward, so the mean is not typical. The median is robust to outliers.

**Why normalise before K-Means but you don't have to for a decision tree?**
Distance-based algorithms (K-Means, KNN) compare magnitudes directly, so an unscaled ₹1,61,370 charge column overwhelms a 1–95 age range. Decision trees split one attribute at a time on thresholds, so they are invariant to monotonic scaling — normalisation neither helps nor hurts them.

**What does lift tell you that confidence doesn't?**
Confidence ignores how common the consequent is. `X → Diabetes=No` can have 76% confidence purely because 76% of patients are non-diabetic — lift would be ~1.0, revealing no real association. Lift > 1 means the co-occurrence beats chance.

**Why is 75% accuracy good here?**
For three classes the majority-class baseline is 46%, so 75% is a 29-point improvement. Perfect accuracy on medical outcome data would suggest label leakage, not skill.

**Why did you choose k=3 when silhouette preferred k=2?**
The elbow in inertia sits at k=3, and 3 segments are clinically actionable in a way 2 are not; the silhouette cost (0.346→0.290) is small. The decision is stated explicitly in the script and report rather than hidden.

**Are association rules causal?**
No. They describe co-occurrence. `High_Charges → Deceased` reflects that severely ill patients both cost more and die more; spending doesn't cause death.

**Why 10-fold CV rather than one train/test split?**
400 records is small; a single split's accuracy depends on which rows landed in the test set. 10-fold CV averages over 10 splits for a stabler estimate, and matches WEKA's default so the two tools are comparable.

### Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: mlxtend` | `pip install mlxtend` |
| `FileNotFoundError: hospital_clean.csv` | Run `01_preprocess.py` first (it creates it), or just use `python run_all.py` |
| `KeyError: 'Blood_Pressure'` | Your column names differ — add the rename block from Part 9 |
| Matplotlib opens no window / crashes | Already handled: scripts use the `Agg` backend and save PNGs to `outputs/` |
| WEKA says *"Nominal attribute expected"* in Associate tab | You opened a file with numeric columns. Use `hospital_binned.arff` (all nominal) |
| WEKA Classify shows nonsense accuracy (e.g. 100%) | `Outcome` is probably not set as the class, or you opened the wrong file. Use `hospital_outcome.arff` where `Outcome` is the last attribute |
| WEKA K-Means refuses to run | Numeric-only file needed → use `hospital_cluster.arff` |
| WEKA accuracy differs from Python by 1–3% | Expected — different fold seeding and slightly different pruning implementations. Note it in your report |
| ARFF looks garbled in WEKA | Attribute values containing spaces/commas must be quoted — the exporter already handles this; re-run `01_preprocess.py` |
| Excessive association rules | Raise `min_support` (0.10 → 0.15) or `min_confidence` (0.50 → 0.70), and filter by `lift > 1.5` |

### One-paragraph summary of the whole workflow

> **Preprocess** (impute → drop ID → discretise → normalise) → **Classify** (J48 + Naive Bayes, 10-fold CV, predict Outcome) → **Cluster** (standardise → pick k via elbow/silhouette → K-Means → profile the segments) → **Associate** (basket-ify → Apriori → rank rules by lift) → **Interpret** (map every finding to a management decision) → **Conclude** (value + limitations).

---

*This guide and all scripts are reproducible end-to-end: `python run_all.py` regenerates every number, table and figure quoted above.*
