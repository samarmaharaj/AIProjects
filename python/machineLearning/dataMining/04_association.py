"""
=============================================================================
 TASK 5 - ASSOCIATION RULE MINING  (Apriori)
 CS-415 Lab Assignment 7 : Data Mining in Hospital Management
=============================================================================
 Run:  python 04_association.py     (needs: pip install mlxtend)

 The assignment asks for relationships such as:
     Disease -> Treatment
     Age Group + Disease -> Admission Type

 Apriori needs DISCRETE "items", so every record is turned into a basket of
 items such as {Disease_Fracture, Treatment_Surgery, Age_Group_51-65,
 Admission_Type_Urgent, Outcome_Recovered, ...}.

 Key metrics (all reported below):
   Support    = how often the itemset appears      (P(A and B))
   Confidence = P(B | A)                           (reliability of the rule)
   Lift       = confidence / support(B)            (> 1  =  genuine association,
                                                    1  =  independent,
                                                    <1  =  items avoid each other)
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_colwidth", 70)

# =============================================================================
print("=" * 78)
print("STEP 1 : BUILD TRANSACTIONS FROM THE BINNED TABLE")
print("=" * 78)

df = pd.read_csv("hospital_binned.csv")
print(f"Loaded hospital_binned.csv : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# Each patient becomes one "transaction" = the set of attribute=value items.
transactions = []
for _, row in df.iterrows():
    items = [f"{col}={row[col]}" for col in df.columns]
    transactions.append(items)

print(f"\nBuilt {len(transactions)} transactions.")
print("\nExample - first 3 patient baskets:")
for i in range(3):
    print(f"  Patient {i+1}: {{{', '.join(transactions[i])}}}")

# One-hot encode the baskets into the boolean matrix Apriori expects
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
basket = pd.DataFrame(te_array, columns=te.columns_)

print(f"\nOne-hot basket matrix: {basket.shape[0]} transactions x "
      f"{basket.shape[1]} distinct items")
print(f"\nAll distinct items ({len(te.columns_)}):")
for item in sorted(te.columns_):
    print(f"   {item}")

# =============================================================================
print("\n" + "=" * 78)
print("STEP 2 : FREQUENT ITEMSETS WITH APRIORI")
print("=" * 78)
print("min_support = 0.10  -> an itemset must appear in at least 10% of the")
print("400 patients (>= 40 patients) to be considered 'frequent'.\n")

freq_items = apriori(basket, min_support=0.10, use_colnames=True)
freq_items["length"] = freq_items["itemsets"].apply(len)
freq_items = freq_items.sort_values("support", ascending=False)

print(f"Found {len(freq_items)} frequent itemsets.")
print(f"  size 1 (single items): {(freq_items['length'] == 1).sum()}")
print(f"  size 2 (pairs)       : {(freq_items['length'] == 2).sum()}")
print(f"  size 3 (triples)     : {(freq_items['length'] == 3).sum()}")
print(f"  size 4+              : {(freq_items['length'] >= 4).sum()}")

print("\nTop 15 most frequent itemsets:")
top_freq = freq_items.head(15).copy()
top_freq["itemset"] = top_freq["itemsets"].apply(lambda s: ", ".join(sorted(s)))
print(top_freq[["support", "itemset"]]
      .assign(support=lambda d: (d["support"] * 100).round(1))
      .to_string(index=False))

# =============================================================================
print("\n" + "=" * 78)
print("STEP 3 : GENERATE ASSOCIATION RULES")
print("=" * 78)
print("min_threshold = 0.50  -> a rule must be right at least 50% of the time.\n")

rules = association_rules(freq_items, metric="confidence", min_threshold=0.50)
rules = rules.sort_values(["lift", "confidence"], ascending=False)

rules["antecedent"] = rules["antecedents"].apply(lambda s: " + ".join(sorted(s)))
rules["consequent"] = rules["consequents"].apply(lambda s: " + ".join(sorted(s)))

# Apriori emits every rule in BOTH directions (A->B and B->A). They have the
# same support and lift, so keeping both just clutters the table. We keep the
# direction with the higher confidence.
def drop_mirror_images(r):
    keep = []
    seen = set()
    for idx, row in r.sort_values("confidence", ascending=False).iterrows():
        key = frozenset([frozenset(row["antecedents"]), frozenset(row["consequents"])])
        if key in seen:
            continue
        seen.add(key)
        keep.append(idx)
    return r.loc[keep]

rules_all = rules.copy()
rules = drop_mirror_images(rules)
print(f"Generated {len(rules_all)} rules with confidence >= 50%; "
      f"{len(rules)} after removing mirror-image duplicates.")

strong = rules[(rules["lift"] > 1.2) & (rules["confidence"] >= 0.60)]
print(f"Of these, {len(strong)} are STRONG: lift > 1.2 AND confidence >= 60%")

# =============================================================================
print("\n" + "=" * 78)
print("STEP 4 : TOP RULES OVERALL  (ranked by lift)")
print("=" * 78)

show = rules.head(20)[["antecedent", "consequent", "support",
                       "confidence", "lift"]].copy()
show["support"] = (show["support"] * 100).round(1).astype(str) + "%"
show["confidence"] = (show["confidence"] * 100).round(1).astype(str) + "%"
show["lift"] = show["lift"].round(2)
print(show.to_string(index=False, max_colwidth=45))

# =============================================================================
print("\n" + "=" * 78)
print("STEP 5 : THE TWO RELATIONSHIPS THE ASSIGNMENT ASKS FOR")
print("=" * 78)
print("Targeted questions need a LOWER support threshold: a rule like")
print("'Age_Group=65+ & Disease=Fracture -> Admission_Type=Urgent' only covers")
print("~6% of patients, so it can never survive a 10% support cut. We re-run")
print("Apriori with min_support=0.04 / min_confidence=0.40 for this section.")
print("(Lower support = rarer patterns = must be read with more caution.)")

freq_fine = apriori(basket, min_support=0.04, use_colnames=True)
rules_fine = association_rules(freq_fine, metric="confidence", min_threshold=0.40)
rules_fine["antecedent"] = rules_fine["antecedents"].apply(lambda x: " + ".join(sorted(x)))
rules_fine["consequent"] = rules_fine["consequents"].apply(lambda x: " + ".join(sorted(x)))
rules_fine = rules_fine.drop_duplicates(subset=["antecedent", "consequent"])

# IMPORTANT: mirror-image removal must NOT be applied here. A->B and B->A are
# genuinely different rules, and it is the DIRECTION we care about: for the
# assignment we need 'Disease=Fracture -> Treatment=Surgery', not the
# backwards 'Treatment=Surgery -> Disease=Fracture'.
rules_dir = rules_fine[rules_fine["lift"] > 1.0].copy()
print(f"\nFine-grained rule set: {len(rules_dir)} rules "
      f"(support >= 4%, confidence >= 40%, lift > 1)")

def one_consequent(r, prefix):
    """Keep only rules with ONE consequent item starting with `prefix`."""
    return r[(r["consequents"].apply(len) == 1) &
             (r["consequent"].str.startswith(prefix))]

def show_rules(r, n=None, sort_by="lift"):
    t = r.sort_values(sort_by, ascending=False)
    if n:
        t = t.head(n)
    t = t[["antecedent", "consequent", "support", "confidence", "lift"]].copy()
    t["support"] = (t["support"] * 100).round(1).astype(str) + "%"
    t["confidence"] = (t["confidence"] * 100).round(1).astype(str) + "%"
    t["lift"] = t["lift"].round(2)
    print(t.to_string(index=False, max_colwidth=48))


# ---- (A) Disease  ->  Treatment --------------------------------------------
print("\n" + "-" * 78)
print("(A)  DISEASE  ->  TREATMENT")
print("-" * 78)
a_all = rules_dir[rules_dir["antecedent"].str.startswith("Disease=")]
a_all = one_consequent(a_all, "Treatment=")

# (A1) the pure rules:  Disease -> Treatment
a_pure = a_all[a_all["antecedents"].apply(len) == 1]
print("\n(A1) Pure rules - Disease -> Treatment:\n")
show_rules(a_pure)
print(f"\n-> {len(a_pure)} pure disease->treatment rules.")

# (A2) the same, refined with one supporting condition
a_rich = a_all[a_all["antecedents"].apply(len) == 2]
print("\n(A2) Refined with one extra patient characteristic:\n")
show_rules(a_rich, n=10)
a = a_all

# ---- (B) Age Group + Disease  ->  Admission Type ---------------------------
print("\n" + "-" * 78)
print("(B)  AGE GROUP + DISEASE  ->  ADMISSION TYPE")
print("-" * 78)
b = rules_dir[(rules_dir["antecedent"].str.contains("Age_Group=")) &
              (rules_dir["antecedent"].str.contains("Disease=")) &
              (rules_dir["consequents"].apply(len) == 1) &
              (rules_dir["consequent"].str.startswith("Admission_Type="))]
print(f"\n-> {len(b)} such rules found (single-item consequent).\n")
show_rules(b, n=15)
b_all = b

# ---- (C) Bonus: risk factors -> bad outcome --------------------------------
print("\n" + "-" * 78)
print("(C)  CLINICAL RISK FACTORS  ->  OUTCOME  (bonus, very useful clinically)")
print("-" * 78)
c = rules_dir[(rules_dir["consequents"].apply(len) == 1) &
              (rules_dir["consequent"].str.startswith("Outcome=")) &
              (~rules_dir["antecedent"].str.contains("Outcome="))]
print(f"\n-> {len(c)} such rules; top 12 by lift shown.\n")
show_rules(c, n=12)
a, b, c = a, b_all, c

# =============================================================================
print("\n" + "=" * 78)
print("STEP 6 : SAVING RESULTS")
print("=" * 78)

out = rules[["antecedent", "consequent", "support", "confidence", "lift",
             "leverage", "conviction"]].round(4)
out.to_csv("outputs/association_rules.csv", index=False)
print("  outputs/association_rules.csv")

freq_out = freq_items.copy()
freq_out["itemset"] = freq_out["itemsets"].apply(lambda s: ", ".join(sorted(s)))
freq_out[["itemset", "support", "length"]].sort_values("support", ascending=False) \
    .round(4).to_csv("outputs/frequent_itemsets.csv", index=False)
print("  outputs/frequent_itemsets.csv")

with open("outputs/association_report.txt", "w") as f:
    f.write("ASSOCIATION RULE MINING (APRIORI)\n")
    f.write("min_support = 0.10, min_confidence = 0.50\n\n")
    f.write(f"Frequent itemsets: {len(freq_items)}\n")
    f.write(f"Rules generated  : {len(rules)}\n\n")
    f.write("=" * 70 + "\nTOP 30 RULES BY LIFT\n" + "=" * 70 + "\n")
    f.write(rules.head(30)[["antecedent", "consequent", "support",
                            "confidence", "lift"]].round(4).to_string(index=False))
    f.write("\n")

    for title, sub in [("DISEASE -> TREATMENT", a),
                       ("AGE GROUP + DISEASE -> ADMISSION TYPE", b),
                       ("RISK FACTORS -> OUTCOME", c)]:
        if len(sub):
            f.write("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70 + "\n")
            f.write(sub[["antecedent", "consequent", "support", "confidence",
                         "lift"]].round(4).to_string(index=False) + "\n")
print("  outputs/association_report.txt")

print("\n" + "=" * 78)
print("ASSOCIATION COMPLETE  ->  all four tasks finished.")
print("=" * 78)
