"""
=============================================================================
 TASK 2 - DATA PREPROCESSING
 CS-415 Lab Assignment 7 : Data Mining in Hospital Management
=============================================================================
 Run:  python 01_preprocess.py

 What this script does (maps 1:1 to the assignment's Preprocessing task):
   1. Load the raw hospital CSV and inspect it
   2. Handle missing values      -> median (numeric) / mode (categorical)
   3. Remove irrelevant attributes -> Patient_ID is just an ID, no predictive power
   4. Normalize numeric attributes -> Min-Max scaling into [0, 1]
   5. Discretize suitable attributes -> Age_Group, BP_Category,
      LOS_Category, Charges_Category  (needed for association rules & WEKA)
   6. Encode categorical attributes -> one-hot (for Python models)
   7. Save: hospital_clean.csv, hospital_clean.arff (for WEKA),
            hospital_binned.csv  (for association rules),
            Weka buckets: hospital_outcome.arff (labels kept as-is)
=============================================================================
"""

import io
import numpy as np
import pandas as pd

RAW = "hospital_data.csv"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

print("=" * 78)
print("STEP 1 : LOAD RAW DATA")
print("=" * 78)
df = pd.read_csv(RAW)
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head().to_string())
print("\nColumn data types:")
print(df.dtypes.to_string())

# =============================================================================
print("\n" + "=" * 78)
print("STEP 2 : HANDLE MISSING VALUES")
# =============================================================================
print("=" * 78)
missing = df.isna().sum()
print("Missing value count per column:")
print(missing.to_string())
print(f"\nTotal missing cells: {missing.sum()}  "
      f"({missing.sum() / df.size * 100:.2f}% of all cells)")

# --- Numeric -> median (robust to outliers / skew, unlike the mean)
for col in ["Blood_Pressure", "Hospital_Charges"]:
    if df[col].isna().any():
        med = df[col].median()
        n = df[col].isna().sum()
        df[col] = df[col].fillna(med)
        print(f"  [median] {col:18s} : filled {n:3d} missing with {med:.2f}")

# --- Categorical -> mode (most frequent value)
for col in ["Diabetes"]:
    if df[col].isna().any():
        mode = df[col].mode()[0]
        n = df[col].isna().sum()
        df[col] = df[col].fillna(mode)
        print(f"  [mode  ] {col:18s} : filled {n:3d} missing with '{mode}'")

print(f"\nRemaining missing cells after imputation: {df.isna().sum().sum()}")

# =============================================================================
print("\n" + "=" * 78)
print("STEP 3 : REMOVE IRRELEVANT ATTRIBUTES")
# =============================================================================
print("=" * 78)
print("Dropping 'Patient_ID'.")
print("  Reason: it is a unique row identifier. It carries zero clinical")
print("  information, and if left in, a classifier may memorise it (overfit)")
print("  and the model would be meaningless on new patients.")

df_model = df.drop(columns=["Patient_ID"]).copy()
print(f"Columns kept ({len(df_model.columns)}): {list(df_model.columns)}")

# =============================================================================
print("\n" + "=" * 78)
print("STEP 4 : DISCRETIZE SUITABLE ATTRIBUTES")
# =============================================================================
print("=" * 78)
print("Continuous values (age 67, BP 138, bill 64,177) are hard to reason about")
print("in rules. Binning them into clinically meaningful bands makes the output")
print("readable AND is required for Apriori (which only accepts discrete items).")

# --- Age -> Age_Group (clinically standard bands)
df_model["Age_Group"] = pd.cut(
    df_model["Age"],
    bins=[0, 17, 35, 50, 65, 200],
    labels=["0-17", "18-35", "36-50", "51-65", "65+"],
)
print("\nAge -> Age_Group")
print(df_model["Age_Group"].value_counts().sort_index().to_string())

# --- Blood Pressure -> BP_Category (AHA-style ranges)
df_model["BP_Category"] = pd.cut(
    df_model["Blood_Pressure"],
    bins=[0, 90, 120, 140, 1000],
    labels=["Low", "Normal", "Elevated", "High"],
)
print("\nBlood_Pressure -> BP_Category")
print(df_model["BP_Category"].value_counts().sort_index().to_string())

# --- Length of stay -> LOS_Category
df_model["LOS_Category"] = pd.cut(
    df_model["Length_of_Stay"],
    bins=[0, 3, 7, 14, 1000],
    labels=["Short(1-3d)", "Medium(4-7d)", "Long(8-14d)", "Extended(15d+)"],
)
print("\nLength_of_Stay -> LOS_Category")
print(df_model["LOS_Category"].value_counts().sort_index().to_string())

# --- Charges -> Charges_Category (tercile split = low / medium / high cost)
df_model["Charges_Category"] = pd.qcut(
    df_model["Hospital_Charges"], q=3, labels=["Low_Cost", "Medium_Cost", "High_Cost"]
)
print("\nHospital_Charges -> Charges_Category")
print(df_model["Charges_Category"].value_counts().sort_index().to_string())

# =============================================================================
print("\n" + "=" * 78)
print("STEP 5 : NORMALIZE NUMERIC ATTRIBUTES  (Min-Max, range [0,1])")
# =============================================================================
print("=" * 78)
print("Why: Hospital_Charges reaches ~150,000 while Age is ~18-95 and")
print("Length_of_Stay is 1-30. K-Means uses Euclidean distance, so an")
print("unnormalised charge column would completely dominate the clustering")
print("and age/length-of-stay would be ignored. Normalising puts every")
print("attribute on the same footing.")

numeric_cols = ["Age", "Blood_Pressure", "Length_of_Stay", "Hospital_Charges"]
print("\nBefore normalization:")
print(df_model[numeric_cols].agg(["min", "max", "mean"]).round(2).to_string())

df_norm = df_model.copy()
for col in numeric_cols:
    lo, hi = df_model[col].min(), df_model[col].max()
    df_norm[col + "_norm"] = (df_model[col] - lo) / (hi - lo)

print("\nAfter normalization (the *_norm columns):")
print(df_norm[[c + "_norm" for c in numeric_cols]]
      .agg(["min", "max", "mean"]).round(3).to_string())

# =============================================================================
print("\n" + "=" * 78)
print("STEP 6 : ENCODE CATEGORICAL ATTRIBUTES  (one-hot encoding)")
# =============================================================================
print("=" * 78)
print("ML algorithms need numbers, not text. One-hot encoding turns")
print("'Admission_Type = Emergency/Urgent/Elective' into 3 binary columns.")

cat_cols = ["Gender", "Disease", "Diabetes", "Admission_Type", "Treatment"]
encoded = pd.get_dummies(df_model[cat_cols], columns=cat_cols, drop_first=False)
print(f"\n{len(cat_cols)} categorical columns -> {encoded.shape[1]} binary columns:")
print(list(encoded.columns))

# =============================================================================
print("\n" + "=" * 78)
print("STEP 7 : SAVE OUTPUT FILES")
# =============================================================================
print("=" * 78)

# (a) Cleaned + enriched dataset (human/WEKA readable)
df_model.to_csv("hospital_clean.csv", index=False)
print("  hospital_clean.csv      : cleaned data + discretized band columns")

# (b) Fully numeric matrix (ready for scikit-learn)
ml_matrix = pd.concat(
    [df_norm[[c + "_norm" for c in numeric_cols]], encoded,
     df_model[["Age_Group", "BP_Category", "LOS_Category", "Charges_Category"]],
     df_model["Outcome"]],
    axis=1,
)
ml_matrix.to_csv("hospital_ml_matrix.csv", index=False)
print(f"  hospital_ml_matrix.csv  : {ml_matrix.shape[0]} x {ml_matrix.shape[1]} "
      "numeric matrix for scikit-learn")

# (c) Binned transaction table (only discrete items -> for Apriori)
binned = df_model[["Age_Group", "Gender", "Disease", "BP_Category", "Diabetes",
                   "Admission_Type", "Treatment", "LOS_Category",
                   "Charges_Category", "Outcome"]].copy()
binned = binned.astype(str)
binned.to_csv("hospital_binned.csv", index=False)
print("  hospital_binned.csv     : all-discrete table for association rules")


# -----------------------------------------------------------------------------
# ARFF writer - lets you open the SAME data in WEKA with zero fiddling
# -----------------------------------------------------------------------------
def to_arff(df: pd.DataFrame, name: str, path: str):
    """Write a DataFrame to a WEKA .arff file (nominal cols auto-detected)."""
    buf = io.StringIO()
    buf.write(f"@RELATION {name}\n\n")
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            buf.write(f"@ATTRIBUTE {col} NUMERIC\n")
        else:
            vals = sorted(s.astype(str).unique())
            buf.write("@ATTRIBUTE %s {%s}\n" % (col, ",".join(
                "'" + v + "'" if (" " in v or "-" in v or "(" in v) else v
                for v in vals)))
    buf.write("\n@DATA\n")
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            v = row[col]
            if pd.api.types.is_numeric_dtype(df[col]):
                cells.append(str(v))
            else:
                v = str(v)
                cells.append("'" + v + "'" if (" " in v or "-" in v or "(" in v) else v)
        buf.write(",".join(cells) + "\n")
    with open(path, "w") as f:
        f.write(buf.getvalue())
    print(f"  {path:24s} : {df.shape[0]} instances, {df.shape[1]} attributes")


# WEKA file for CLASSIFICATION: keep Outcome as the label, drop the derived
# _norm duplicates (WEKA can normalise/standardise on its own).
weka_cls = df_model[["Age", "Gender", "Disease", "Blood_Pressure", "Diabetes",
                     "Admission_Type", "Treatment", "Length_of_Stay",
                     "Hospital_Charges", "Outcome"]].astype(object)
# move Outcome to last position (WEKA treats the last attribute as the class)
weka_cls = weka_cls[[c for c in weka_cls.columns if c != "Outcome"] + ["Outcome"]]
to_arff(weka_cls, "hospital_outcome", "hospital_outcome.arff")

# WEKA file for CLUSTERING: no class label, numeric only
weka_clu = df_model[["Age", "Blood_Pressure", "Length_of_Stay", "Hospital_Charges"]]
to_arff(weka_clu, "hospital_cluster", "hospital_cluster.arff")

# WEKA file for ASSOCIATION: all nominal, no ID
to_arff(binned.astype(str), "hospital_binned", "hospital_binned.arff")

print("\n" + "=" * 78)
print("PREPROCESSING COMPLETE")
print("=" * 78)
print("""
Files produced:
  hospital_clean.csv       - cleaned + discretized  (open in Excel)
  hospital_ml_matrix.csv   - numeric matrix          (for Python models)
  hospital_binned.csv      - all-discrete             (for Apriori)
  hospital_outcome.arff    - WEKA classification file
  hospital_cluster.arff    - WEKA clustering file
  hospital_binned.arff     - WEKA association file

Next:  python 02_classification.py
""")
