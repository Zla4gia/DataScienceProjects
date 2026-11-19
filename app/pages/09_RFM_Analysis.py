import numpy as np, pandas as pd, streamlit as st
from common import header, footer, mdir, meta

header(9, "RFM Analysis", "Clustering / segmentation",
       "94.6% of the 946 customers ordered once, so the reference's quartile scoring gives 94.6% of customers an F score of 1 and nobody 2 or 3. Rank-based quintile scoring plus rules gives 6 named segments, but k-means finds nothing beyond shuffled data and in a back-test the scores did not reliably predict who ordered again (28 repeat orders).",
       "61 days of what appears to be generated data. The segments are rules, not discoveries.")
M = meta(9)
r_edges, m_edges = M["r_edges_days"], M["m_edges"]
c1, c2, c3 = st.columns(3)
R = c1.slider("Days since last order", 1, 61, 20)
F = c2.selectbox("Number of orders", [1, 2, 3], 0)
Mn = c3.slider("Total spend", 12.0, 2400.0, 500.0)
r = 5 - int(np.digitize(R, r_edges))      # 5 = most recent
m = 1 + int(np.digitize(Mn, m_edges))
f = min(F, 3)
if f >= 2 and r >= 4: seg = "Champions (repeat, recent)"
elif f >= 2: seg = "Repeat, drifting"
elif r >= 4 and m >= 4: seg = "New big spenders"
elif r >= 4: seg = "New customers"
elif m >= 4: seg = "At risk big spenders (one order, not recent)"
else: seg = "Hibernating (one order, low spend, not recent)"
a, b, c = st.columns(3)
a.metric("Recency score (5 = most recent)", r)
b.metric("Spend score (5 = highest)", m)
c.metric("Segment", seg)
st.dataframe(pd.DataFrame(M["segments"]).round(2), hide_index=True, width="stretch")
st.caption(f"Back-test: {M['return_rate_after_cut']:.1%} of customers ordered again after the cut-off. {M['caveat']}")
footer(9, "09 RFM Analysis.ipynb")
