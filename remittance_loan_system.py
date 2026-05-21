import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import warnings
import os

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

np.random.seed(42)
N = 500

print("\n" + "═"*60)
print("  STEP 1: Generating Dummy Remittance Customer Data")
print("═"*60)

DESTINATION_COUNTRIES = ["Qatar", "UAE", "Saudi Arabia", "Malaysia", "South Korea", "Kuwait", "Japan"]
DISTRICTS = ["Kathmandu", "Lalitpur", "Bhaktapur", "Pokhara", "Chitwan", "Butwal", "Dharan"]
JOBS_ABROAD = ["Construction", "Security Guard", "Factory Worker", "Domestic Worker", "Driver", "Technician"]
REMITTANCE_CHANNELS = ["Bank Transfer", "Hundi", "Western Union", "IME", "Prabhu Money"]

dest_weights = [0.28, 0.25, 0.18, 0.12, 0.08, 0.05, 0.04]
chan_weights = [0.40, 0.20, 0.15, 0.15, 0.10]

customer_ids = [f"NRM-{1000+i}" for i in range(N)]
age = np.random.randint(22, 58, N)
family_size = np.random.randint(2, 8, N)
district = np.random.choice(DISTRICTS, N)
dest_country = np.random.choice(DESTINATION_COUNTRIES, N, p=dest_weights)
job_type = np.random.choice(JOBS_ABROAD, N)
years_abroad = np.round(np.random.uniform(0.5, 12, N), 1)
remit_channel = np.random.choice(REMITTANCE_CHANNELS, N, p=chan_weights)

base_salary_map = {
    "Construction": 35000,
    "Security Guard": 28000,
    "Factory Worker": 32000,
    "Domestic Worker": 22000,
    "Driver": 38000,
    "Technician": 52000
}

monthly_remit = np.array([
    int(base_salary_map[j] * np.random.uniform(0.55, 0.80) + years_abroad[i] * 800)
    for i, j in enumerate(job_type)
])

remit_frequency_days = np.random.choice([15, 30, 45, 60], N, p=[0.10, 0.55, 0.25, 0.10])
days_since_last = np.random.randint(1, 90, N)
cash_gap_days = np.clip(remit_frequency_days - days_since_last, 0, 90)

monthly_household_expense = np.array([
    int(family_size[i] * np.random.randint(3500, 6500))
    for i in range(N)
])

existing_loans = np.random.choice([0, 1, 2], N, p=[0.55, 0.35, 0.10])
savings_balance = np.random.randint(0, 80000, N)
bank_account_age = np.random.randint(1, 120, N)

print("\n  Building original engineered features...")

monthly_surplus = monthly_remit - monthly_household_expense

remit_consistency = np.clip(
    1 - (np.random.uniform(0, 0.3, N) * (1 / years_abroad)),
    0.4,
    1.0
)

debt_to_income = np.round(
    existing_loans * 15000 / np.clip(monthly_remit, 1, None),
    2
)

financial_stress = np.clip(
    (days_since_last / remit_frequency_days) * 0.4 +
    (monthly_household_expense / np.clip(monthly_remit, 1, None)) * 0.4 +
    (existing_loans / 3) * 0.2,
    0,
    1
)

financial_stress = np.round(financial_stress, 3)

print("\n" + "═"*60)
print("  STEP 2: Enhanced Feature Engineering")
print("═"*60)

remit_per_year = monthly_remit / np.clip(years_abroad, 0.5, None)
remit_momentum = np.round(np.log1p(remit_per_year) / 10, 3)

savings_coverage_months = np.round(
    savings_balance / np.clip(monthly_household_expense, 1, None),
    2
)

channel_trust_map = {
    "Bank Transfer": 1.00,
    "IME": 0.85,
    "Prabhu Money": 0.80,
    "Western Union": 0.70,
    "Hundi": 0.40
}

channel_trust_score = np.array([
    channel_trust_map[c] for c in remit_channel
])

reliability_index = np.round(
    remit_consistency * 0.50 +
    channel_trust_score * 0.30 +
    (1 - remit_frequency_days / 60) * 0.20,
    3
)

estimated_emi = monthly_remit * 0.15

affordability = np.round(
    np.clip(
        (monthly_surplus - estimated_emi) /
        np.clip(monthly_remit, 1, None),
        -1,
        1
    ),
    3
)

vulnerability_flag = (
    (financial_stress > 0.65) &
    (savings_balance < 10000) &
    (monthly_surplus < 5000)
).astype(int)

experience_tier = pd.cut(
    years_abroad,
    bins=[0, 1, 3, 7, 13],
    labels=["New (<1yr)", "Mid (1-3yr)", "Senior (3-7yr)", "Veteran (7+yr)"]
).astype(str)

country_stability_map = {
    "South Korea": 0.95,
    "Japan": 0.93,
    "UAE": 0.85,
    "Qatar": 0.80,
    "Kuwait": 0.78,
    "Malaysia": 0.72,
    "Saudi Arabia": 0.68
}

dest_stability_score = np.array([
    country_stability_map[c] for c in dest_country
])

remit_expense_ratio = np.round(
    monthly_remit / np.clip(monthly_household_expense, 1, None),
    3
)

overdue_risk = np.round(
    np.clip(days_since_last / remit_frequency_days, 0, 1),
    3
)

creditworthiness_score = np.round(
    reliability_index * 25 +
    affordability.clip(0, 1) * 20 +
    dest_stability_score * 15 +
    remit_momentum * 15 +
    (1 - financial_stress) * 15 +
    savings_coverage_months.clip(0, 1) * 10,
    1
)

print(f"  ✓ remit_momentum            mean: {remit_momentum.mean():.3f}")
print(f"  ✓ savings_coverage_months   mean: {savings_coverage_months.mean():.2f} months")
print(f"  ✓ reliability_index         mean: {reliability_index.mean():.3f}")
print(f"  ✓ affordability_score       mean: {affordability.mean():.3f}")
print(f"  ✓ remit_expense_ratio       mean: {remit_expense_ratio.mean():.3f}")
print(f"  ✓ overdue_risk_score        mean: {overdue_risk.mean():.3f}")
print(f"  ✓ dest_stability_score      mean: {dest_stability_score.mean():.3f}")
print(f"  ✓ vulnerability_flag        flagged: {vulnerability_flag.sum()} customers")
print(f"  ✓ creditworthiness_score    mean: {creditworthiness_score.mean():.1f} / 100")

print("\n  Building repayment labels...")

def repayment_label(i):
    score = (
        0.30 * remit_consistency[i] +
        0.25 * (1 - debt_to_income[i] / 3) +
        0.20 * (monthly_surplus[i] / 30000) +
        0.15 * (years_abroad[i] / 12) +
        0.10 * (savings_balance[i] / 80000)
    )

    noise = np.random.normal(0, 0.05)

    return 1 if (score + noise) > 0.38 else 0

repayment_prob_label = np.array([
    repayment_label(i) for i in range(N)
])

recommended_loan = np.where(
    repayment_prob_label == 1,
    np.clip(
        monthly_remit * np.random.uniform(0.8, 1.5, N),
        5000,
        80000
    ).astype(int),
    np.zeros(N, dtype=int)
)

recommended_emi_months = np.where(
    recommended_loan > 40000,
    3,
    np.where(recommended_loan > 20000, 2, 1)
)

zero_interest_eligible = (
    (remit_consistency > 0.80) &
    (existing_loans == 0) &
    (bank_account_age > 12)
).astype(int)

df = pd.DataFrame({
    "customer_id": customer_ids,
    "age": age,
    "district": district,
    "family_size": family_size,
    "destination_country": dest_country,
    "job_type": job_type,
    "years_abroad": years_abroad,
    "remittance_channel": remit_channel,
    "monthly_remit_npr": monthly_remit,
    "remit_frequency_days": remit_frequency_days,
    "days_since_last_transfer": days_since_last,
    "cash_gap_days": cash_gap_days,
    "monthly_household_expense": monthly_household_expense,
    "monthly_surplus_npr": monthly_surplus,
    "existing_loans": existing_loans,
    "savings_balance_npr": savings_balance,
    "bank_account_age_months": bank_account_age,
    "remit_consistency_score": np.round(remit_consistency, 3),
    "debt_to_income_ratio": debt_to_income,
    "financial_stress_score": financial_stress,
    "remit_momentum": remit_momentum,
    "savings_coverage_months": savings_coverage_months,
    "reliability_index": reliability_index,
    "affordability_score": affordability,
    "remit_expense_ratio": remit_expense_ratio,
    "overdue_risk_score": overdue_risk,
    "dest_stability_score": dest_stability_score,
    "experience_tier": experience_tier,
    "vulnerability_flag": vulnerability_flag,
    "creditworthiness_score": creditworthiness_score,
    "will_repay": repayment_prob_label,
    "recommended_loan_npr": recommended_loan,
    "recommended_emi_months": recommended_emi_months,
    "zero_interest_eligible": zero_interest_eligible,
})

print(f"\n  ✓ Generated {N} customer records with {df.shape[1]} features")
print(f"  ✓ Repayment approval rate : {repayment_prob_label.mean()*100:.1f}%")
print(f"  ✓ Zero-interest eligible  : {zero_interest_eligible.sum()} customers")

df.to_csv(os.path.join(OUTPUT_DIR, "remittance_customers.csv"), index=False)

print("  ✓ Saved → remittance_customers.csv")

print("\n" + "═"*60)
print("  STEP 3: Predictive Analytics — Repayment Model")
print("═"*60)

cat_cols = [
    "district",
    "destination_country",
    "job_type",
    "remittance_channel",
    "experience_tier"
]

encoders = {}
df_model = df.copy()

for col in cat_cols:
    le = LabelEncoder()
    df_model[col + "_enc"] = le.fit_transform(df_model[col])
    encoders[col] = le

FEATURES = [
    "age",
    "family_size",
    "years_abroad",
    "monthly_remit_npr",
    "remit_frequency_days",
    "days_since_last_transfer",
    "cash_gap_days",
    "monthly_household_expense",
    "monthly_surplus_npr",
    "existing_loans",
    "savings_balance_npr",
    "bank_account_age_months",
    "remit_consistency_score",
    "debt_to_income_ratio",
    "financial_stress_score",
    "district_enc",
    "destination_country_enc",
    "job_type_enc",
    "remittance_channel_enc",
    "experience_tier_enc",
    "remit_momentum",
    "savings_coverage_months",
    "reliability_index",
    "affordability_score",
    "remit_expense_ratio",
    "overdue_risk_score",
    "dest_stability_score",
    "vulnerability_flag",
    "creditworthiness_score",
]

X = df_model[FEATURES]
y = df_model["will_repay"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

df["repayment_probability"] = np.round(
    clf.predict_proba(X)[:, 1],
    3
)

print(f"\n  Model          : Random Forest Classifier")
print(f"  Features used  : {len(FEATURES)}")
print(f"  Accuracy       : {acc*100:.1f}%")
print(f"  Train samples  : {len(X_train)}   |   Test samples: {len(X_test)}")

print(f"\n  Classification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Will Default", "Will Repay"]
    )
)

importances = pd.Series(
    clf.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

print("  Top 5 Predictive Features:")

for feat, score in importances.head(5).items():
    bar = "█" * int(score * 200)

    print(f"    {feat:<35} {bar} {score:.3f}")

print("\n" + "═"*60)
print("  STEP 4: Prescriptive Analytics — Loan Recommendations")
print("═"*60)

approved = df[df["will_repay"] == 1].copy()
stress_high = df[df["financial_stress_score"] > 0.65]
vulnerable = df[df["vulnerability_flag"] == 1]

print(f"\n  Customers approved for loan     : {len(approved)}")
print(f"  Avg recommended loan (NPR)      : {approved['recommended_loan_npr'].mean():,.0f}")
print(f"  Avg repayment period            : {approved['recommended_emi_months'].mean():.1f} months")
print(f"  Zero-interest offers            : {approved['zero_interest_eligible'].sum()}")
print(f"  Avg financial stress score      : {approved['financial_stress_score'].mean():.3f}")
print(f"  Avg creditworthiness score      : {approved['creditworthiness_score'].mean():.1f} / 100")

print(f"\n  High-stress customers (>0.65)   : {len(stress_high)}")
print(f"  Of those, loan-eligible         : {stress_high['will_repay'].sum()}")
print(f"  Vulnerable customers flagged    : {len(vulnerable)}")
print(f"  Vulnerable + loan-eligible      : {vulnerable['will_repay'].sum()}")

print("\n  Loan Approval Rate by Experience Tier:")

for tier in ["New (<1yr)", "Mid (1-3yr)", "Senior (3-7yr)", "Veteran (7+yr)"]:
    subset = df[df["experience_tier"] == tier]

    if len(subset) > 0:
        rate = subset["will_repay"].mean() * 100
        avg_credit = subset["creditworthiness_score"].mean()

        print(
            f"    {tier:<18}  approval: {rate:5.1f}%   "
            f"avg creditworthiness: {avg_credit:.1f}"
        )

print("\n  Sample Loan Recommendations:")

sample = approved[[
    "customer_id",
    "district",
    "destination_country",
    "monthly_remit_npr",
    "creditworthiness_score",
    "reliability_index",
    "cash_gap_days",
    "repayment_probability",
    "recommended_loan_npr",
    "recommended_emi_months",
    "zero_interest_eligible"
]].head(8)

print(sample.to_string(index=False))

print("\n" + "═"*60)
print("  STEP 5: Generating Visualizations")
print("═"*60)

PURPLE = "#7F77DD"
TEAL = "#1D9E75"
AMBER = "#EF9F27"
CORAL = "#D85A30"
BLUE = "#378ADD"
GRAY = "#888780"
BG = "#F8F8F6"
LIGHT = "#EEEDFE"
RED = "#C0392B"

fig = plt.figure(figsize=(20, 18), facecolor=BG)

fig.suptitle(
    "AI-Powered Remittance Intelligence & Smart Loan System\n"
    "Nepal — Bank Internship Presentation  |  Enhanced Feature Engineering",
    fontsize=15,
    fontweight="bold",
    color="#2C2C2A",
    y=0.99
)

gs = gridspec.GridSpec(
    4,
    3,
    figure=fig,
    hspace=0.52,
    wspace=0.38
)

ax1 = fig.add_subplot(gs[0, 0])

country_avg = df.groupby(
    "destination_country"
)["monthly_remit_npr"].mean().sort_values()

colors_bar = [
    PURPLE if v == country_avg.max() else BLUE
    for v in country_avg.values
]

bars1 = ax1.barh(
    country_avg.index,
    country_avg.values,
    color=colors_bar,
    height=0.6,
    edgecolor="white",
    linewidth=0.5
)

ax1.set_title(
    "Avg Monthly Remittance by Country (NPR)",
    fontsize=9,
    fontweight="bold",
    color="#2C2C2A"
)

ax1.set_xlabel("NPR", fontsize=8, color=GRAY)
ax1.tick_params(labelsize=8)
ax1.set_facecolor(BG)
ax1.spines[["top","right","left"]].set_visible(False)

for bar, val in zip(bars1, country_avg.values):
    ax1.text(
        val + 300,
        bar.get_y() + bar.get_height()/2,
        f"{val/1000:.0f}k",
        va="center",
        fontsize=7,
        color="#444441"
    )

plt.savefig(
    os.path.join(OUTPUT_DIR, "remittance_dashboard.png"),
    dpi=150,
    bbox_inches="tight",
    facecolor=BG,
    edgecolor="none"
)

print("  ✓ Dashboard saved → remittance_dashboard.png")

print("\n" + "═"*60)
print("  FINAL SUMMARY REPORT")
print("═"*60)

print(f"""
  Dataset
  ───────────────────────────────────────────────────
  Total customers analysed       : {N}
  Total features per record      : {df.shape[1]}
  Output CSV                     : remittance_customers.csv

  Predictive Model
  ───────────────────────────────────────────────────
  Features used in model         : {len(FEATURES)}
  Accuracy                       : {acc*100:.1f}%
  Customers likely to repay      : {repayment_prob_label.sum()}
  Top predictor                  : {importances.index[0]}

  Enhanced Feature Engineering
  ───────────────────────────────────────────────────
  Remittance Momentum (mean)     : {remit_momentum.mean():.3f}
  Reliability Index (mean)       : {reliability_index.mean():.3f}
  Affordability Score (mean)     : {affordability.mean():.3f}
  Creditworthiness Score (mean)  : {creditworthiness_score.mean():.1f}

  Prescriptive Recommendations
  ───────────────────────────────────────────────────
  Loans recommended              : {(recommended_loan > 0).sum()}
  Avg loan amount (NPR)          : {approved["recommended_loan_npr"].mean():,.0f}
  Zero-interest eligible         : {approved["zero_interest_eligible"].sum()}

  Output Files
  ───────────────────────────────────────────────────
  ✓ remittance_customers.csv
  ✓ remittance_dashboard.png
  ✓ Saved in: {OUTPUT_DIR}
""")

print("═"*60)
print("  Done. Ready for your bank internship presentation!")
print("═"*60 + "\n")