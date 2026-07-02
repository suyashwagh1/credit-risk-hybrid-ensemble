import streamlit as st
import joblib
import torch
import torch.nn as nn
import numpy as np

# ---- Load models ----
@st.cache_resource
def load_models():
    rf_model = joblib.load('rf_final_model.pkl')
    xgb_model = joblib.load('xgb_final_model.pkl')
    meta_model = joblib.load('meta_model.pkl')
    scaler_nn = joblib.load('scaler_nn.pkl')

    class CreditRiskNN(nn.Module):
        def __init__(self, input_dim):
            super(CreditRiskNN, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Dropout(0.3),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.network(x)

    nn_model = CreditRiskNN(input_dim=11)
    nn_model.load_state_dict(torch.load('nn_final_model.pth'))
    nn_model.eval()

    return rf_model, xgb_model, meta_model, scaler_nn, nn_model


rf_model, xgb_model, meta_model, scaler_nn, nn_model = load_models()

OPTIMAL_THRESHOLD = 0.33

# ---- App UI ----
st.title("Credit Risk Scoring")
st.write("Hybrid ensemble model (Random Forest + XGBoost + Neural Network) predicting loan default risk.")

st.header("Applicant Details")

col1, col2 = st.columns(2)

with col1:
    utilization = st.slider("Revolving Utilization of Unsecured Lines", 0.0, 2.0, 0.3, 0.01)
    age = st.number_input("Age", min_value=18, max_value=100, value=40)
    past_due_30_59 = st.number_input("Times 30-59 Days Past Due", min_value=0, max_value=20, value=0)
    debt_ratio = st.number_input("Debt Ratio", min_value=0.0, max_value=3000.0, value=0.3)
    monthly_income = st.number_input("Monthly Income ($)", min_value=0, max_value=200000, value=5000)

with col2:
    open_credit_lines = st.number_input("Number of Open Credit Lines/Loans", min_value=0, max_value=60, value=8)
    times_90_late = st.number_input("Times 90 Days Late", min_value=0, max_value=20, value=0)
    real_estate_loans = st.number_input("Number of Real Estate Loans/Lines", min_value=0, max_value=30, value=1)
    past_due_60_89 = st.number_input("Times 60-89 Days Past Due", min_value=0, max_value=20, value=0)
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=0)

# had_sentinel_value is not user-facing — always 0 for a new hypothetical applicant
had_sentinel = 0

st.header("Decision Threshold")
st.write(
    "The default cutoff below (0.33) was derived from a cost-sensitive analysis assuming a missed "
    "defaulter costs 5x a wrongly rejected applicant. Adjust it to see how the approve/flag decision "
    "changes under different risk tolerances."
)
threshold = st.slider(
    "Risk threshold for flagging an applicant",
    min_value=0.05, max_value=0.95, value=OPTIMAL_THRESHOLD, step=0.01
)

if st.button("Check Risk", type="primary"):
    features = np.array([[
        utilization, age, past_due_30_59, debt_ratio, monthly_income,
        open_credit_lines, times_90_late, real_estate_loans,
        past_due_60_89, dependents, had_sentinel
    ]])

    # RF and XGBoost predictions (raw features, no scaling needed)
    pred_rf = rf_model.predict_proba(features)[:, 1]
    pred_xgb = xgb_model.predict_proba(features)[:, 1]

    # NN prediction (needs scaling)
    features_scaled = scaler_nn.transform(features)
    features_tensor = torch.FloatTensor(features_scaled)
    with torch.no_grad():
        pred_nn = nn_model(features_tensor).numpy().flatten()

    # Meta-learner combines all three
    meta_input = np.column_stack([pred_rf, pred_xgb, pred_nn])
    final_prob = meta_model.predict_proba(meta_input)[:, 1][0]

    st.header("Result")

    st.metric("Predicted Default Probability", f"{final_prob:.1%}")

    if final_prob >= threshold:
        st.error(f"⚠️ HIGH RISK — flagged at threshold {threshold:.2f}")
    else:
        st.success(f"✅ LOW RISK — approved at threshold {threshold:.2f}")

    if threshold != OPTIMAL_THRESHOLD:
        st.caption(f"Note: using a custom threshold ({threshold:.2f}) instead of the cost-optimal default ({OPTIMAL_THRESHOLD}).")

    st.caption(f"Individual model predictions — RF: {pred_rf[0]:.1%}, XGBoost: {pred_xgb[0]:.1%}, NN: {pred_nn[0]:.1%}")