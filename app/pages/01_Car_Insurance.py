import numpy as np, pandas as pd, joblib, streamlit as st
from common import header, footer, mdir, meta
header(1, "Car Insurance Modelling", "Regression", "On 678,013 real French policies a gradient-boosted Poisson model beats a plain Poisson GLM (test deviance 0.578 vs 0.608; constant 0.627) and a GLM given the right shapes (0.598). The riskiest tenth of policies claims about 8.7x as often as the safest tenth. Age has a non-linear effect and brand-new cars show a spike.",
       "Claim *frequency* only (no claim cost). Associations in one portfolio, not causes. One random split.")
M = meta(1)

@st.cache_resource
def load():
    return joblib.load(mdir(1) / "freq_hgb.joblib"), joblib.load(mdir(1) / "freq_glm.joblib")
hgb, glm = load()
c1, c2, c3 = st.columns(3)
age = c1.slider("Driver age", 18, 90, 40); vage = c1.slider("Vehicle age (years)", 0, 20, 5); power = c1.slider("Vehicle power class", 4, 9, 6)
bm = c2.slider("Bonus-malus (100 = no history; lower = better record)", 50, 150, 60); dens = c2.number_input("Inhabitants per km2 where the car is kept", 1, 30000, 800); area = c2.selectbox("Area (A rural ... F urban centre)", M["categories"]["Area"], index=2)
gas = c3.selectbox("Fuel", M["categories"]["VehGas"]); brand = c3.selectbox("Vehicle brand", M["categories"]["VehBrand"], index=0); region = c3.selectbox("Region", M["categories"]["Region"], index=5); exposure = c3.slider("Years of cover", 0.1, 1.0, 1.0)
x = pd.DataFrame([{"DrivAge": age, "VehPower": power, "VehAge": vage, "BonusMalus": bm, "LogDensity": float(np.log(dens)), "Area": area, "VehBrand": brand, "VehGas": gas, "Region": region}])[M["features"]]
f_h, f_g = float(hgb.predict(x)[0]), float(glm.predict(x)[0])
m1, m2, m3 = st.columns(3)
m1.metric("Expected claims per year (boosting)", f"{f_h:.3f}", f"{f_h / M['overall_freq']:.1f}x the portfolio average ({M['overall_freq']:.3f})")
m2.metric("Poisson GLM says", f"{f_g:.3f}"); m3.metric(f"Expected claims over {exposure:.1f} years", f"{f_h * exposure:.3f}")
ages = np.arange(18, 91, 2); grid = pd.concat([x] * len(ages), ignore_index=True); grid["DrivAge"] = ages
st.line_chart(pd.DataFrame({"boosting": hgb.predict(grid), "GLM": glm.predict(grid)}, index=ages), height=260)
st.caption("Predicted claims per year for this profile as driver age varies (everything else as chosen above). The boosting model can bend where the GLM follows smooth splines.")
footer(1, "01 Car Insurance Modelling.ipynb")
