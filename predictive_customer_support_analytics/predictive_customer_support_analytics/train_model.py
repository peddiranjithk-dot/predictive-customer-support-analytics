"""Train predictive models on support ticket data.

Model 1: classifier  -> probability a ticket will be escalated
Model 2: regressor   -> expected resolution time (hours)
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (classification_report, mean_absolute_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from generate_data import generate

CATEGORICAL = ["channel", "category", "priority", "customer_tier"]
NUMERIC = ["previous_tickets", "message_length", "sentiment_score",
           "first_response_minutes", "account_age_days"]
FEATURES = CATEGORICAL + NUMERIC


def build_pipeline(model):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
    return Pipeline([("prep", pre), ("model", model)])


def plot_importance(pipe, path):
    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = pipe.named_steps["model"].feature_importances_
    order = np.argsort(imp)[-10:]
    plt.figure(figsize=(8, 5))
    plt.barh(np.array(names)[order], imp[order])
    plt.title("Top features driving escalation")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    try:
        df = pd.read_csv("tickets.csv")
    except FileNotFoundError:
        df = generate()
        df.to_csv("tickets.csv", index=False)

    X = df[FEATURES]
    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X, df["escalated"], df["resolution_hours"], test_size=0.2,
        random_state=42, stratify=df["escalated"])

    # --- Escalation classifier ---
    clf = build_pipeline(RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                random_state=42, n_jobs=-1))
    clf.fit(X_train, yc_train)
    proba = clf.predict_proba(X_test)[:, 1]
    print("=== Escalation prediction ===")
    print(classification_report(yc_test, clf.predict(X_test)))
    print(f"ROC-AUC: {roc_auc_score(yc_test, proba):.3f}\n")

    # --- Resolution time regressor ---
    reg = build_pipeline(GradientBoostingRegressor(random_state=42))
    reg.fit(X_train, yr_train)
    pred = reg.predict(X_test)
    print("=== Resolution time prediction ===")
    print(f"MAE: {mean_absolute_error(yr_test, pred):.2f} hours")
    print(f"R2 : {r2_score(yr_test, pred):.3f}\n")

    joblib.dump(clf, "escalation_model.joblib")
    joblib.dump(reg, "resolution_model.joblib")
    plot_importance(clf, "feature_importance.png")
    print("Saved: escalation_model.joblib, resolution_model.joblib, feature_importance.png")


if __name__ == "__main__":
    main()
