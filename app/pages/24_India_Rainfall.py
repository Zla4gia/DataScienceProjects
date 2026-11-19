import numpy as np, pandas as pd, streamlit as st
from scipy import stats
import statsmodels.api as sm
from common import header, footer, mdir, meta

header(24, "Rainfall Trends in India", "Data visualization",
       "No significant trend in annual rainfall over 1901-2015 (-3.0 mm per decade, 95% interval -12 to +6, p = 0.52; Mann-Kendall p = 0.39). But 30-year slopes run from -49 to +71 mm per decade, so any short window can 'show' a trend. The monsoon is 75% of the rain; dry years are monsoon failures.",
       "One all-India series (regional trends can differ) and 115 values; the number of 'extreme' years depends on the definition.")
M = meta(24)
r = pd.read_csv(mdir(24) / "india_rainfall_clean.csv").set_index("YEAR")
y0, y1 = st.slider("Window of years", 1901, 2015, (1901, 2015))
col = st.selectbox("Series", ["ANNUAL", "Jun-Sep", "Mar-May", "Oct-Dec", "Jan-Feb"])
s = r.loc[y0:y1, col]
X = sm.add_constant(np.asarray(s.index, float) - y0)
o = sm.OLS(s.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3}) if len(s) > 12 else None
if o is not None:
    lo, hi = o.conf_int()[1]
    st.metric(f"Trend of {col}, {y0}-{y1}", f"{o.params[1] * 10:+.1f} mm per decade", f"95% interval {lo * 10:+.1f} to {hi * 10:+.1f} (p = {o.pvalues[1]:.2f})", delta_color="off")
    st.caption("Try short windows (30 years) and watch the sign flip; the whole century gives about -3 mm per decade with an interval around zero.")
else:
    st.warning("Choose a window of more than 12 years.")
st.line_chart(pd.DataFrame({col: s, "10-year mean": s.rolling(10, center=True).mean()}), height=300)
k = st.slider("Extreme-year threshold (standard deviations)", 0.5, 2.5, 1.5, 0.25)
mu, sd = r.ANNUAL.mean(), r.ANNUAL.std()
dry, wet = r.index[r.ANNUAL < mu - k * sd].tolist(), r.index[r.ANNUAL > mu + k * sd].tolist()
st.write(f"Beyond {k} sd of the mean ({mu:.0f} mm, sd {sd:.0f}): **{len(dry)} dry years** {dry} and **{len(wet)} wet years** {wet}. A normal distribution would give about {stats.norm.cdf(-k) * 115:.1f} on each side.")
st.write("Trend tests from the notebook:")
st.dataframe(pd.DataFrame(M["trend"]).T.round(3), width="stretch")
footer(24, "24 Rainfall Trends in India.ipynb")
