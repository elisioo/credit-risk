import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
#  LOAD MODEL FROM NOTEBOOK-SAVED .PKL
# ─────────────────────────────────────────
def load_model():
    model_path = 'credit_risk_model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"'{model_path}' not found.\n"
            "      Please run all cells in 'Credit_Risk_Analysis.ipynb' first\n"
            "      to generate the saved model file."
        )
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# ─────────────────────────────────────────
#  INPUT HELPER
# ─────────────────────────────────────────
def get_input(prompt, min_val=None, max_val=None, is_float=True):
    while True:
        try:
            val = float(input(prompt)) if is_float else int(input(prompt))
            if min_val is not None and val < min_val:
                print(f"  ⚠  Value must be at least {min_val}. Try again.")
                continue
            if max_val is not None and val > max_val:
                print(f"  ⚠  Value must be at most {max_val}. Try again.")
                continue
            return val
        except ValueError:
            print("  ⚠  Please enter a valid number.")

# ─────────────────────────────────────────
#  COLLECT BORROWER INFO
# ─────────────────────────────────────────
def get_borrower_input():
    print("\n" + "─" * 50)
    print("  📋  ENTER BORROWER DETAILS")
    print("─" * 50)

    rev_util    = get_input("  Revolving Utilization Rate (0.0 - 1.0)  : ", 0.0, 1.0)
    age         = get_input("  Age                                      : ", 18, 100, is_float=False)
    late_30_59  = get_input("  Times Late 30-59 Days (last 2 yrs)      : ", 0, is_float=False)
    debt_ratio  = get_input("  Debt Ratio (monthly debt / income)       : ", 0.0)
    monthly_inc = get_input("  Monthly Income ($)                       : ", 0.0)
    open_credit = get_input("  Number of Open Credit Lines              : ", 0, is_float=False)
    late_90     = get_input("  Times Late 90+ Days (last 2 yrs)        : ", 0, is_float=False)
    real_estate = get_input("  Number of Real Estate Loans              : ", 0, is_float=False)
    late_60_89  = get_input("  Times Late 60-89 Days (last 2 yrs)      : ", 0, is_float=False)
    dependents  = get_input("  Number of Dependents                     : ", 0, is_float=False)

    return [[rev_util, age, late_30_59, debt_ratio, monthly_inc,
             open_credit, late_90, real_estate, late_60_89, dependents]]

# ─────────────────────────────────────────
#  PREDICT
# ─────────────────────────────────────────
def predict(model, data):
    cols = ['rev_util','age','late_30_59','debt_ratio','monthly_inc',
            'open_credit','late_90','real_estate','late_60_89','dependents']
    df_input = pd.DataFrame(data, columns=cols)
    pred  = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0]
    return pred, proba

# ─────────────────────────────────────────
#  DISPLAY RESULT
# ─────────────────────────────────────────
def display_result(pred, proba):
    no_default_pct = proba[0] * 100
    default_pct    = proba[1] * 100

    print("\n" + "═" * 50)
    print("  📊  CREDIT RISK PREDICTION RESULT")
    print("═" * 50)

    if default_pct < 30:
        risk_level = "🟢  LOW RISK"
        advice     = "Likely safe to approve the loan."
    elif default_pct < 60:
        risk_level = "🟡  MODERATE RISK"
        advice     = "Consider higher interest rate or reduced limit."
    else:
        risk_level = "🔴  HIGH RISK"
        advice     = "Recommend rejecting or requiring collateral."

    print(f"\n  Prediction     : {'DEFAULT' if pred == 1 else 'NO DEFAULT'}")
    print(f"  Risk Level     : {risk_level}")
    print(f"\n  Probability of No Default : {no_default_pct:.1f}%")
    print(f"  Probability of Default    : {default_pct:.1f}%")

    bar_len = 30
    filled  = int(default_pct / 100 * bar_len)
    bar     = "█" * filled + "░" * (bar_len - filled)
    print(f"\n  Default Risk   : [{bar}] {default_pct:.1f}%")
    print(f"\n  💡  Advice     : {advice}")
    print("═" * 50)

# ─────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────
def main():
    print("\n" + "═" * 50)
    print("  🏦  CREDIT RISK PREDICTION SYSTEM")
    print("  Powered by Gradient Boosting (AUC: 0.86)")
    print("═" * 50)
    print("  Loading model from credit_risk_model.pkl ...")

    try:
        model = load_model()
        print(f"  ✅  Model loaded! ({type(model).__name__})\n")
    except FileNotFoundError as e:
        print(f"\n  ❌  ERROR: {e}")
        return

    while True:
        data = get_borrower_input()
        pred, proba = predict(model, data)
        display_result(pred, proba)

        print("\n  Run another prediction?")
        again = input("  Enter 'y' to continue or any key to exit: ").strip().lower()
        if again != 'y':
            print("\n  👋  Goodbye!\n")
            break

if __name__ == "__main__":
    main()