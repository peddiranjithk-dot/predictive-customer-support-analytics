"""Score a new support ticket using the trained models."""
import joblib
import pandas as pd

clf = joblib.load("escalation_model.joblib")
reg = joblib.load("resolution_model.joblib")

new_ticket = pd.DataFrame([{
    "channel": "phone",
    "category": "technical",
    "priority": "urgent",
    "customer_tier": "premium",
    "previous_tickets": 5,
    "message_length": 900,
    "sentiment_score": -0.7,
    "first_response_minutes": 120,
    "account_age_days": 400,
}])

risk = clf.predict_proba(new_ticket)[0, 1]
hours = reg.predict(new_ticket)[0]

print(f"Escalation probability : {risk:.1%}")
print(f"Expected resolution    : {hours:.1f} hours")
print("Action                 :", "Route to senior agent now" if risk > 0.5 else "Standard queue")
