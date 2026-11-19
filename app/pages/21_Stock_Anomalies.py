import numpy as np, pandas as pd, streamlit as st
from scipy import stats
from common import header, footer, mdir, meta

header(21, "Stock Market Anomaly Detection", "Time series",
       "Planted-anomaly test: a volatility-adjusted (EWMA) z-score finds 55% of planted 3-sigma jumps when 60 days are flagged, against 46% for a global z-score and 36% for an Isolation Forest. On 7 years of real data it flags 33 of 8,495 stock-days, and 76% of those fall on earnings days (about 3% of all days).",
       "A threshold from the normal distribution gives 4-6x more alarms than advertised (returns have heavy tails). 'Anomaly' depends on the reference: the March 2020 crash is flagged by a global z-score but not by the volatility-adjusted one. Not trading advice.")
M = meta(21)
st.subheader("Is today's move unusual?")
c1, c2, c3 = st.columns(3)
r = c1.number_input("Today's return (%)", -50.0, 50.0, -6.0, 0.5)
sd = c2.number_input("Typical daily move of this stock recently (EWMA volatility, %)", 0.2, 10.0, 1.6, 0.1)
th = c3.slider("Alarm threshold (|z|)", 2.0, 8.0, 4.5, 0.5)
z = abs(r / sd)
st.metric("z-score (return / recent volatility)", f"{z:.1f}", "ALARM" if z >= th else "normal", delta_color="inverse" if z >= th else "off")
st.write("A bell curve would call this event %.4f%% likely; the notebook measured that |z| > %.0f happened on %s of days." % (2 * stats.norm.sf(z) * 100, min(5, max(2, int(z))), {k: round(v, 3) for k, v in M["tail_table"].get(f"|z| > {min(5, max(2, int(z)))}", {}).items()}))
st.subheader("Flagged days in 2018-2024 (|z| > 4.5)")
fl = pd.read_csv(mdir(21) / "flagged_days.csv")
st.dataframe(fl, hide_index=True, width="stretch")
st.subheader("Detector comparison on planted jumps")
c1, c2 = st.columns(2)
c1.write("AUC by jump size (multiples of volatility)")
c1.dataframe(pd.DataFrame(M["auc_planted"]).T.round(3))
c2.write("Recall when 60 days are flagged (%)")
c2.dataframe(pd.DataFrame(M["recall_1pct"]).T.round(0))
footer(21, "21 Stock Market Anomaly Detection.ipynb")
