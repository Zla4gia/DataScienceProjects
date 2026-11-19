import numpy as np, pandas as pd, streamlit as st
from common import header, footer, meta
header(2, "Price Elasticity of Demand", "Regression", "Re-running the article's row-by-row ratio gives noise (only 8% of rows usable, 49% of ratios negative). A fixed-effects regression on the 10,000 distinct store-item-weeks gives an elasticity of +0.008 with a 95% interval of -0.035 to +0.051: no detectable price effect in this file.",
       "The data file looks constructed (each observation repeated about 10 times, money columns inconsistent). The curves below are *what the estimate implies*, not advice: an elasticity near zero makes revenue rise with any price, which is a warning about the data, not about customers.")
M = meta(2); b, (lo, hi) = M["elasticity"], M["ci"]
items = sorted(M["items"], key=lambda s: int(s.split("_")[1])); it = st.selectbox("Item", items, index=82 if len(items) > 82 else 0); I = M["items"][it]
chg = st.slider("Price change (%)", -15, 15, 5); st.caption(f"Observed weekly price moves for items are only a few percent; price range seen for this item: {I['price_min']:.0f} to {I['price_max']:.0f}.")
p0, q0 = I["price_mean"], I["qty_mean"]; pr = np.linspace(.85, 1.15, 31); rows = {}
for lab, bb in [("estimate", b), ("CI low", lo), ("CI high", hi)]: rows[lab] = q0 * pr ** bb
df = pd.DataFrame(rows, index=np.round(pr * p0, 1)); c1, c2 = st.columns(2)
c1.subheader("Predicted weekly quantity"); c1.line_chart(df); c2.subheader("Predicted revenue"); c2.line_chart(df.mul(df.index.to_numpy(), axis=0))
k = 1 + chg / 100; st.metric(f"Quantity if the price changes by {chg}%", f"{q0 * k ** b:,.0f}", f"{(k ** b - 1) * 100:+.2f}% (interval {((k ** lo) - 1) * 100:+.2f}% to {((k ** hi) - 1) * 100:+.2f}%)")
st.write("Elasticity estimate:", round(b, 3), "| 95% interval:", [round(lo, 3), round(hi, 3)], "| competitor-price coefficient:", round(M["cross_price"], 3), "(wrong sign for substitutes; not trusted)")
footer(2, "02 Price Elasticity of Demand.ipynb")
