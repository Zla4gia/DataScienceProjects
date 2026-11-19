import numpy as np, pandas as pd, joblib, streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from common import header, footer, mdir, meta, register_main


class WoE(BaseEstimator, TransformerMixin):
    """Weight-of-evidence transformer (same class as in the notebook; used only to read the saved scorecard)."""

    def __init__(self, num_cols, cat_cols, n_bins=5, smooth=0.5):
        self.num_cols, self.cat_cols, self.n_bins, self.smooth = num_cols, cat_cols, n_bins, smooth

    def _group(self, s, c):
        if c in self.num_cols:
            return pd.cut(s, self.edges_[c], include_lowest=True).astype(str)
        return s.astype(str)

    def fit(self, X, y):
        raise NotImplementedError("fitting happens in the notebook")

    def transform(self, X):
        return pd.DataFrame({c: self._group(X[c], c).map(self.table_[c]).fillna(0.0).to_numpy() for c in self.num_cols + self.cat_cols}, index=X.index)


register_main(WoE=WoE)
header(5, "Credit Scoring and Risk Tiers", "Classification",
       "A transparent weight-of-evidence scorecard on 1,000 real German loans matches gradient boosting (cross-validated AUC 0.79, Gini 0.57, KS 0.46). Cutting the out-of-fold scores into 5 tiers gives bad rates from 63.5% (tier E) down to 5.0% (tier A).",
       "1994 German data, 1,000 loans, some tiny groups. Personal status (sex/marital), age and foreign-worker are inputs of the 20-variable card; in real lending some would be restricted. A demonstration, not a credit decision.")
M = meta(5)
S = joblib.load(mdir(5) / "scorecard.joblib")
pipe = S["pipeline"]
cv, dflt, cdef = M["cat_values"], M["defaults"], M["cat_default"]
c1, c2, c3 = st.columns(3)
v = {}
v["checking_status"] = c1.selectbox("Checking account", cv["checking_status"], index=cv["checking_status"].index(cdef["checking_status"]))
v["savings_status"] = c1.selectbox("Savings", cv["savings_status"], index=cv["savings_status"].index(cdef["savings_status"]))
v["credit_history"] = c1.selectbox("Credit history", cv["credit_history"], index=cv["credit_history"].index(cdef["credit_history"]))
v["purpose"] = c2.selectbox("Purpose", cv["purpose"], index=cv["purpose"].index(cdef["purpose"]))
v["duration"] = c2.slider("Duration (months)", 4, 72, int(dflt["duration"]))
v["credit_amount"] = c2.slider("Credit amount (DM)", 250, 18424, int(dflt["credit_amount"]))
v["age"] = c3.slider("Age", 19, 75, int(dflt["age"]))
v["employment"] = c3.selectbox("Employment length", cv["employment"], index=cv["employment"].index(cdef["employment"]))
v["housing"] = c3.selectbox("Housing", cv["housing"], index=cv["housing"].index(cdef["housing"]))
row = {**{k: float(x) for k, x in dflt.items()}, **cdef, **v}
X = pd.DataFrame([row])
p_bad = float(np.clip(pipe.predict_proba(X)[0, 1], 1e-4, 1 - 1e-4))
pts = S["offset"] + S["factor"] * np.log((1 - p_bad) / p_bad)
tier = ["E (highest risk)", "D", "C", "B", "A (lowest risk)"][int(np.digitize(pts, S["tier_edges"]))]
m1, m2, m3 = st.columns(3)
m1.metric("Score (points)", f"{pts:.0f}")
m2.metric("Risk tier", tier)
m3.metric("Estimated chance the loan goes bad", f"{p_bad:.0%}", f"sample average {M['bad_rate']:.0%}", delta_color="off")
st.dataframe(pd.DataFrame(M["tiers"]), hide_index=True, width="stretch")
st.caption("600 points = 30:1 good:bad odds and +30 points doubles the odds (conventions). The table shows out-of-fold bad rates by score quintile. Inputs not shown use the sample's typical values.")
st.write("Information value of the strongest variables:", dict(list(M["iv"].items())[:6]))
footer(5, "05 Credit Scoring and Risk Tiers.ipynb")
