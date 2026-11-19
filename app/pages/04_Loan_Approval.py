import numpy as np, pandas as pd, joblib, streamlit as st
from common import header, footer, mdir, meta, register_main


def add_feats(x):
    x = x.copy()
    x["TotalIncome"] = x.ApplicantIncome + x.CoapplicantIncome
    x["LogTotalIncome"] = np.log1p(x.TotalIncome)
    x["LogLoan"] = np.log1p(x.LoanAmount)
    x["EMI"] = x.LoanAmount * 1000 / x.Loan_Amount_Term
    x["EMI_to_income"] = x.EMI / x.TotalIncome.replace(0, np.nan)
    return x


register_main(add_feats=add_feats)
header(4, "Loan Approval Prediction", "Classification",
       "On 491 past decisions, every model reaches about 81% accuracy and the one-line rule 'approve if credit history = 1' reaches the same 81.3%; the models add ranking ability (AUC 0.70 -> 0.77). One 80/20 split of this size can swing accuracy from 68% to 90%.",
       "Toy example of past decisions at one lender, not a credit model. Gender and marital status are inputs of the trained model, and group approval rates differ more in the model than in the data: do not use for real decisions.")
M = meta(4)


@st.cache_resource
def load():
    return joblib.load(mdir(4) / "loan_logreg.joblib")


pipe = load()
cv = M["cat_values"]
c1, c2, c3 = st.columns(3)
gender = c1.selectbox("Gender", cv["Gender"])
married = c1.selectbox("Married", cv["Married"], index=1)
dep = c1.selectbox("Dependents", cv["Dependents"])
edu = c1.selectbox("Education", cv["Education"])
selfe = c2.selectbox("Self-employed", cv["Self_Employed"])
area = c2.selectbox("Property area", cv["Property_Area"], index=2)
ch = c2.selectbox("Credit history", ["1 (meets guidelines)", "0 (does not)", "unknown"])
inc = c3.number_input("Applicant income (monthly)", 150, 81000, 4000)
co = c3.number_input("Co-applicant income", 0, 41667, 1500)
amt = c3.number_input("Loan amount (thousands)", 17, 700, 130)
term = c3.selectbox("Term (months)", [360, 180, 480, 300, 240, 120, 84, 60, 36, 12], 0)
x = pd.DataFrame([{"Gender": gender, "Married": married, "Dependents": dep, "Education": edu, "Self_Employed": selfe, "ApplicantIncome": inc,
                   "CoapplicantIncome": co, "LoanAmount": amt, "Loan_Amount_Term": term,
                   "Credit_History": (np.nan if ch.startswith("unknown") else float(ch[0])), "Property_Area": area}])[M["features_raw"]]
x["Credit_History"] = x["Credit_History"].astype("object")
p = float(pipe.predict_proba(x)[0, 1])
th = M["threshold"]
st.metric("Estimated probability of approval", f"{p:.0%}", "approve" if p >= th else "refuse", delta_color="normal" if p >= th else "inverse")
st.progress(min(max(p, 0.0), 1.0))
st.caption(f"Decision threshold {th} (chosen with an assumed 3:1 cost of a bad approval vs a good refusal; the cost curve is flat from about 0.3 to 0.7). 69.9% of past applications were approved.")
st.write("Cross-validated results from the notebook:", {k: {m: v[m] for m in ("accuracy", "roc_auc")} for k, v in M["cv"].items()})
footer(4, "04 Loan Approval Prediction.ipynb")
