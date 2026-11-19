import numpy as np, pandas as pd, joblib, streamlit as st
from common import header, footer, mdir, meta
header(3, "Bike Sharing Demand", "Regression", "Training on everything before Oct 2012 and testing on the last quarter, gradient boosting (Poisson loss) has RMSLE 0.43 and MAE 45 bikes/hour against 80 for 'average by hour and day type'. It under-predicts the growing demand by about 16 bikes/hour, mostly at commute peaks.",
       "The 80% band is calibrated on an earlier quarter and still covers only about 73% of test hours (raw: 52%). Two years of one city, no event data.")
M = meta(3)

@st.cache_resource
def load(): return joblib.load(mdir(3) / "bike_models.joblib")
B = load()
c1, c2, c3 = st.columns(3)
year = c1.selectbox("Year", [2011, 2012], index=1); month = c1.slider("Month", 1, 12, 6); weekday = c1.selectbox("Day of week (0 = Sunday)", list(range(7)), index=3)
workingday = c2.checkbox("Working day", True); holiday = c2.checkbox("Holiday", False); weather = c2.selectbox("Weather", M["weather"], index=0)
temp = c3.slider("Temperature (deg C)", 0, 41, 22); hum = c3.slider("Humidity", 0.0, 1.0, 0.6); wind = c3.slider("Wind speed", 0.0, 57.0, 12.0)
rows = pd.DataFrame([{"year": year, "hour": h, "weekday": weekday, "month": month, "holiday": int(holiday), "workingday": int(workingday), "weather": pd.Categorical([weather], categories=M["weather"])[0], "temp": temp, "feel_temp": temp, "humidity": hum, "windspeed": wind} for h in range(24)])[M["features"]]
rows["weather"] = pd.Categorical(rows["weather"].astype(str), categories=M["weather"])
mean = B["mean"].predict(rows); lo = np.clip(B["q10"].predict(rows) - B["conformal_Q"], 0, None); hi = B["q90"].predict(rows) + B["conformal_Q"]
st.line_chart(pd.DataFrame({"expected rentals": mean, "80% band low": lo, "80% band high": hi}, index=range(24)), height=320)
st.metric("Expected rentals over the day", f"{mean.sum():,.0f}", f"peak {mean.max():.0f} at {int(np.argmax(mean))}:00")
st.caption("Hourly forecast for the chosen day; the band widens the quantile-model band by the conformal amount (about 19.5 bikes).")
footer(3, "03 Bike Sharing Demand.ipynb")
