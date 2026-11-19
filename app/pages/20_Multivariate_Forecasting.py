import numpy as np, pandas as pd, joblib, streamlit as st
from common import header, footer, mdir, meta

header(20, "Multivariate Time Series Forecasting", "Time series",
       "Stock returns carry no usable multivariate signal (a VAR chosen by AIC has order 0; next-day R-squared -2% to -6% against predicting zero). On hourly transformer data the other six series help only with the right set-up: LightGBM predicting the *change* in oil temperature is 6-9% better at +6h/+12h and no better at +1h/+24h; linear models get worse.",
       "One test period (5,900 hours, Oct 2017 to Jun 2018), one seed. Differences of a few hundredths of a degree are within noise. The notebook explains why predicting levels fails (trees cannot extrapolate).")
M = meta(20)
D = joblib.load(mdir(20) / "ett_forecasters.joblib")
et = pd.read_csv(mdir(20) / "etth1_test_slice.csv.gz", parse_dates=["date"]).set_index("date")
cols, sc, feats = D["cols"], D["scaler"], D["features"]
Z = pd.DataFrame(sc.transform(et[cols]), index=et.index, columns=cols)
i = st.slider("Forecast origin (hours into the 8-month test period)", 30, len(et) - 25, 400)
t = et.index[i]
row = {}
for c in cols:
    for k in range(24):
        row[f"{c}_lag{k}"] = Z[c].iloc[i - k]
for nm, v, per in [("hour", t.hour, 24), ("dow", t.dayofweek, 7)]:
    row[nm + "_sin"], row[nm + "_cos"] = np.sin(2 * np.pi * v / per), np.cos(2 * np.pi * v / per)
x = pd.DataFrame([row])[feats]
cur = et.OT.iloc[i]
pred = float((D["lgbm_24h_multivariate"].predict(x)[0] + Z.OT.iloc[i]) * D["sd_ot"] + D["mu_ot"])
actual = et.OT.iloc[i + 24]
seasonal = et.OT.iloc[i]                      # last value (naive) for comparison
c1, c2, c3, c4 = st.columns(4)
c1.metric("Oil temperature now (deg C)", f"{cur:.1f}")
c2.metric("Forecast for +24 h (LightGBM, 7 series)", f"{pred:.1f}", f"{pred - cur:+.1f}")
c3.metric("Actual +24 h", f"{actual:.1f}")
c4.metric("Error of the model / of 'no change'", f"{abs(pred - actual):.2f} / {abs(cur - actual):.2f}")
st.line_chart(et.OT.iloc[max(0, i - 72):i + 25].rename("oil temperature"), height=300)
st.caption(f"Origin {t}. The line shows the 72 hours before the origin and the 24 hours after it (the part to be forecast).")
st.write("Test MAE of the oil temperature (deg C):")
st.dataframe(pd.DataFrame(M["test_mae"]).T.round(2), width="stretch")
st.write("Reduction of MAE from adding the 6 other series (%), by learner:")
st.dataframe(pd.DataFrame(M["gain_pct"]).T.round(1), width="stretch")
footer(20, "20 Multivariate Time Series Forecasting.ipynb")
