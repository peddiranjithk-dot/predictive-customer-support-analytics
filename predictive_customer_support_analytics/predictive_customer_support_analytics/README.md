# Predictive Customer Support Analytics

A simple base project that predicts, for each incoming support ticket:

1. **Escalation risk** - probability the ticket will need a senior agent (Random Forest classifier)
2. **Resolution time** - expected hours to resolve (Gradient Boosting regressor)

This lets a support team route risky tickets early and plan staffing.

## Project structure

| File | Purpose |
|------|---------|
| `generate_data.py` | Creates a synthetic ticket dataset (`tickets.csv`) |
| `train_model.py` | Trains and evaluates both models, saves them with joblib |
| `predict.py` | Scores a new ticket with the saved models |
| `requirements.txt` | Python dependencies |

## Setup and run

```bash
pip install -r requirements.txt
python generate_data.py
python train_model.py
python predict.py
```

`train_model.py` also writes `feature_importance.png` showing what drives escalation.

## Using your own data

Replace `tickets.csv` with real data containing these columns:
`channel, category, priority, customer_tier, previous_tickets, message_length,
sentiment_score, first_response_minutes, account_age_days, escalated, resolution_hours`.

## Ideas for extending

- Add text features (TF-IDF on ticket body) or a real sentiment model
- Add churn / CSAT prediction
- Serve predictions through a Flask or FastAPI endpoint
- Build a dashboard with Streamlit
