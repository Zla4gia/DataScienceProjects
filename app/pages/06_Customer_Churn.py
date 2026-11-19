import numpy as np, pandas as pd, joblib, streamlit as st
from common import header, footer, mdir, meta

header(6, "Customer Churn: Comparing Many Models", "Classification",
       "14 models were compared with 5-fold x 3 cross-validation on 7,043 telecom customers. Seven, including logistic regression, are statistically tied at AUC 0.845-0.849; the top 10% of customers by risk contain about 29% of churners, the top 30% about 67%.",
       "No hyper-parameter tuning; the campaign economics in the notebook are assumed numbers; associations, not causes.")
M = meta(6)


@st.cache_resource
def load():
    return joblib.load(mdir(6) / "churn_best.joblib"), joblib.load(mdir(6) / "churn_logreg.joblib")


best, lr = load()
cv, cdef, dflt = M["cat_values"], M["cat_default"], M["defaults"]
c1, c2, c3 = st.columns(3)
v = {"Contract": c1.selectbox("Contract", cv["Contract"]),
     "tenure": c1.slider("Tenure (months)", 0, 72, int(dflt["tenure"])),
     "MonthlyCharges": c1.slider("Monthly charges", 18.0, 119.0, float(dflt["MonthlyCharges"])),
     "InternetService": c2.selectbox("Internet service", cv["InternetService"], index=cv["InternetService"].index(cdef["InternetService"])),
     "PaymentMethod": c2.selectbox("Payment method", cv["PaymentMethod"]),
     "TechSupport": c2.selectbox("Tech support", cv["TechSupport"], index=cv["TechSupport"].index(cdef["TechSupport"])),
     "OnlineSecurity": c3.selectbox("Online security", cv["OnlineSecurity"], index=cv["OnlineSecurity"].index(cdef["OnlineSecurity"])),
     "PaperlessBilling": c3.selectbox("Paperless billing", cv["PaperlessBilling"], index=cv["PaperlessBilling"].index(cdef["PaperlessBilling"])),
     "SeniorCitizen": int(c3.checkbox("Senior citizen", False))}
row = {**cdef, **dflt, **v}
row["TotalCharges"] = row["tenure"] * row["MonthlyCharges"]
X = pd.DataFrame([row])
pb, pl = float(best.predict_proba(X)[0, 1]), float(lr.predict_proba(X)[0, 1])
m1, m2 = st.columns(2)
m1.metric(f"Churn probability ({M['best']})", f"{pb:.0%}", f"overall churn rate {M['churn_rate']:.0%}", delta_color="off")
m2.metric("Churn probability (logistic regression)", f"{pl:.0%}")
st.bar_chart(pd.DataFrame(M["cv"]).T["AUC"].sort_values(), height=300)
st.caption("Cross-validated AUC of the 14 models (the majority-class baseline is 0.50). Differences within the top group are smaller than fold-to-fold noise.")
footer(6, "06 Customer Churn Model Comparison.ipynb")
