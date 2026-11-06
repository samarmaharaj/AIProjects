"""
make_dataset.py
---------------
Creates a hospital dataset with EXACTLY the schema shown in the assignment:
Patient_ID, Age, Gender, Disease, Blood_Pressure, Diabetes, Admission_Type,
Treatment, Length_of_Stay, Hospital_Charges, Outcome

NOTE: If you already have your own CSV, you can SKIP this file entirely and
point the other scripts at your file. This generator exists only so the whole
pipeline is runnable/reproducible.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 400

# ---------------------------------------------------------------- demographics
age = np.random.normal(58, 18, N).clip(18, 95).round().astype(int)
age = np.clip(age, 18, 95)
gender = np.random.choice(["Male", "Female"], N, p=[0.53, 0.47])

diseases = ["Fracture", "Pneumonia", "Infection", "Cardiac", "Diabetes"]
disease_p = [0.28, 0.22, 0.20, 0.16, 0.14]
disease = np.random.choice(diseases, N, p=disease_p)

# ------------------------------------------------------------------ diabetes
# older + diabetic patients are more likely to be 'Cardiac' / 'Diabetes' cases
diab_prob = np.where(age > 60, 0.34, 0.15)
diabetes = np.where(np.random.rand(N) < diab_prob, "Yes", "No")

# ------------------------------------------------------------- admission type
def admission_for(d, a):
    if d in ("Cardiac", "Pneumonia"):
        return np.random.choice(["Emergency", "Urgent", "Elective"], p=[0.55, 0.30, 0.15])
    if d == "Fracture":
        return np.random.choice(["Urgent", "Emergency", "Elective"], p=[0.50, 0.35, 0.15])
    return np.random.choice(["Elective", "Urgent", "Emergency"], p=[0.45, 0.35, 0.20])

admission = np.array([admission_for(d, a) for d, a in zip(disease, age)])

# ------------------------------------------------------------------ treatment
TREAT = {
    "Fracture":  ["Surgery", "Medication", "Monitoring"],
    "Pneumonia": ["Oxygen_Therapy", "Antibiotics", "Medication"],
    "Infection": ["Antibiotics", "Monitoring", "Medication"],
    "Cardiac":   ["Monitoring", "Surgery", "Medication"],
    "Diabetes":  ["Lifestyle_Counselling", "Medication", "Monitoring"],
}
treatment = np.array([
    np.random.choice(TREAT[d], p=[0.5, 0.3, 0.2]) for d in disease
])

# -------------------------------------------------------------- blood pressure
bp_base = np.random.normal(126, 16, N)
bp_base = np.where(age > 60, bp_base + 10, bp_base)
bp_base = np.where(diabetes == "Yes", bp_base + 7, bp_base)
bp_base = np.where(admission == "Emergency", bp_base + 5, bp_base)
blood_pressure = bp_base.clip(88, 185).round().astype(int)

# ------------------------------------------------------------ length of stay
los = np.random.gamma(3.0, 1.9, N)
los = np.where(disease == "Fracture", los * 1.9, los)     # surgery -> long stay
los = np.where(disease == "Cardiac", los * 1.6, los)
los = np.where(disease == "Infection", los * 0.85, los)
los = np.where(treatment == "Surgery", los + 4, los)
los = np.where(treatment == "Lifestyle_Counselling", los * 0.4, los)
los = np.where(admission == "Emergency", los + 1.5, los)
los = np.where(age > 75, los + 1.2, los)
length_of_stay = np.clip(los.round(), 1, 30).astype(int)

# ----------------------------------------------------------- hospital charges
charges = (length_of_stay * np.random.normal(3400, 700, N)
           + 6000
           + np.where(treatment == "Surgery", 42000, 0)
           + np.where(disease == "Cardiac", 9000, 0)
           + np.where(admission == "Emergency", 4000, 0)
           + np.random.normal(0, 4000, N))
hospital_charges = charges.clip(8000, 165000).round(0)

# -------------------------------------------------------------------- outcome
# A single continuous clinical RISK SCORE drives the outcome. Every risk
# factor pushes it up; the three outcome bands are then cut from that score.
# This makes the label genuinely learnable (as it would be in real data):
# older + emergency + cardiac + long stay + high BP  ->  worse outcome.
risk = (0.058 * (age - 55)
        + 0.145 * (length_of_stay - 6)
        + 0.90  * (admission == "Emergency")
        + 1.05  * (disease == "Cardiac")
        + 0.62  * (disease == "Pneumonia")
        + 0.50  * (diabetes == "Yes")
        + 0.022 * (blood_pressure - 126)
        + np.random.normal(0, 0.55, N))       # irreducible clinical noise

# Cut points are set on the risk-score distribution so the three outcome
# classes come out realistically balanced (~46% Recovered / 32% Stable /
# 22% Deceased) - a badly skewed target (e.g. 4 deceased patients) would make
# any classifier meaningless.
q_recovered = np.quantile(risk, 0.46)
q_stable    = np.quantile(risk, 0.78)

outcome = np.where(risk > q_stable,    "Deceased",
           np.where(risk > q_recovered, "Stable", "Recovered"))

df = pd.DataFrame({
    "Patient_ID":       [f"P{i+1:04d}" for i in range(N)],
    "Age":              age,
    "Gender":           gender,
    "Disease":          disease,
    "Blood_Pressure":   blood_pressure,
    "Diabetes":         diabetes,
    "Admission_Type":   admission,
    "Treatment":        treatment,
    "Length_of_Stay":   length_of_stay,
    "Hospital_Charges": hospital_charges,
    "Outcome":          outcome,
})

# ------------------------------------------------- inject realistic missingness
def punch(col, frac, rng):
    idx = rng.choice(df.index, int(len(df) * frac), replace=False)
    df.loc[idx, col] = np.nan

rng = np.random.default_rng(7)
punch("Blood_Pressure", 0.04, rng)
punch("Diabetes", 0.03, rng)
punch("Hospital_Charges", 0.03, rng)

df = df.sample(frac=1, random_state=1).reset_index(drop=True)   # shuffle rows
df.to_csv("hospital_data.csv", index=False)

print(f"Wrote hospital_data.csv  ->  {df.shape[0]} rows x {df.shape[1]} cols")
print("\nMissing values per column:")
print(df.isna().sum().to_string())
print("\nOutcome distribution:")
print(df["Outcome"].value_counts().to_string())
print("\nDisease x Treatment (top combinations):")
print(df.groupby(["Disease", "Treatment"]).size().sort_values(ascending=False).head(8).to_string())
