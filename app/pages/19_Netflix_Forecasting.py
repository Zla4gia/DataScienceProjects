import numpy as np, pandas as pd, joblib, streamlit as st
from common import header, footer, mdir, meta

header(19, "Netflix Subscriptions Forecasting", "Time series",
       "In a rolling back-test, a simple 'last value + recent average growth' is as good as ARIMA(1,1,1) (mean error about 6-7 million subscribers over 1-4 quarters; 'no growth' is 14.5). Error grows to 9-10 million four quarters ahead, and the nominal 80% intervals contain the truth only 43-47% of the time.",
       "42 quarters of one series. Treat the forecast as a scenario: the notebook's honest range is probably at least twice as wide as the printed band, and a slowdown or surge in growth is not in the model.")
M = meta(19)
F = joblib.load(mdir(19) / "netflix_forecasters.joblib")
s = F["series"]
h = st.slider("Quarters ahead", 1, 8, 4)
widen = st.slider("Widen the interval by (x) to reflect the poor back-test coverage", 1.0, 3.0, 2.0, 0.5)
fa = F["arima"].get_forecast(h)
mean = np.asarray(fa.predicted_mean)
ci = np.asarray(fa.conf_int(alpha=0.2))
half = (ci[:, 1] - ci[:, 0]) / 2 * widen
ets = np.asarray(F["ets"].forecast(h))
idx = pd.period_range(s.index[-1] + 1, periods=h, freq="Q")
hist = pd.Series(s.values, index=s.index.to_timestamp())
df = pd.DataFrame({"ARIMA(1,1,1)": mean, "interval low": mean - half, "interval high": mean + half, "ETS damped": ets}, index=idx.to_timestamp())
st.line_chart(pd.concat([hist.rename("actual"), df], axis=1), height=380)
st.dataframe(df.round(1).assign(period=idx.astype(str)).set_index("period"), width="stretch")
st.write("Back-test MAE (millions) by horizon:")
st.dataframe(pd.DataFrame(M["backtest_mae"]).T.round(2), width="stretch")
st.write("Share of true values inside the nominal 80% interval (%):", M["interval_coverage_pct"])
footer(19, "19 Netflix Subscriptions Forecasting.ipynb")
