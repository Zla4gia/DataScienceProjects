import numpy as np, pandas as pd, streamlit as st
import statsmodels.api as sm
from common import header, footer, mdir, meta

header(23, "Carbon Emissions and Temperature", "Data visualization",
       "The CO2 file mixes two series (ppm and year-over-year %). CO2 rose from 316 to 421 ppm (+33%) and its yearly increase grew from 0.9 ppm/year in the 1960s to 2.4 in the 2010s. World temperature change rose 0.28 degrees C per decade (CI 0.24-0.32), with a change of slope near 1975. The reference's 0.96 correlation is mostly shared trend: after detrending it is 0.47, for year-to-year changes 0.31.",
       "The regression of temperature on CO2 is an association between two trending series, not a causal test or a climate model; the 'what-if' is a straight-line extrapolation with intervals that are probably too narrow.")
M = meta(23)
a = pd.read_csv(mdir(23) / "annual_co2_temperature.csv")
cr = pd.read_csv(mdir(23) / "country_warming_rates.csv")
y0, y1 = st.slider("Years used", int(a.year.min()), int(a.year.max()), (1961, int(a.year.max())))
s = a[(a.year >= y0) & (a.year <= y1)]
st.line_chart(s.set_index("year")[["co2"]], height=220)
st.line_chart(s.set_index("year")[["dT"]], height=220)
X = sm.add_constant(s.year.values - y0)
o = sm.OLS(s.dT.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
st.metric(f"World temperature trend {y0}-{y1}", f"{o.params[1] * 10:.2f} deg C per decade", f"95% interval {o.conf_int()[1][0] * 10:.2f} to {o.conf_int()[1][1] * 10:.2f}", delta_color="off")
st.subheader("What-if (association only)")
ppm = st.slider("Additional CO2 (ppm)", 0, 100, 30)
mm = sm.OLS(a.dT.values, sm.add_constant(a.co2.values)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
lo, hi = mm.conf_int()[1]
st.write(f"+{ppm} ppm goes with **{ppm * mm.params[1]:+.2f} degrees C** in this regression (interval {ppm * lo:.2f} to {ppm * hi:.2f}). It cannot tell the effect of CO2 from anything else that changed smoothly over 1961-2022.")
st.write("Correlation of CO2 and temperature:", M["correlations"])
st.subheader("Warming rate by country (degrees C per decade)")
q = st.text_input("Country", "Finland")
st.dataframe(cr[cr.country.str.contains(q, case=False)].round(3), hide_index=True, width="stretch")
st.bar_chart(cr.head(15).set_index("country").deg_C_per_decade, height=260)
footer(23, "23 Carbon Emissions Impact Analysis.ipynb")
