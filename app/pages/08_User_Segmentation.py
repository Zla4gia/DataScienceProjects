import numpy as np, pandas as pd, joblib, streamlit as st
from common import header, footer, mdir, meta

header(8, "User Profiling and Segmentation", "Clustering",
       "k-means will always return segments, but on this data they are not real: the columns are statistically independent (0 of 70 dependence tests survive correction), the best silhouette (0.052) equals that of shuffled data (0.051-0.054), bootstrap stability is poor (ARI 0.25) and no attribute predicts CTR or conversion.",
       "The file looks randomly generated, so this page demonstrates what NOT to trust: the 'segment' below is whichever cluster k-means happened to build.")
M = meta(8)


@st.cache_resource
def load():
    return joblib.load(mdir(8) / "segmenter.joblib")


S = load()
cv, rg, dflt = M["cat_values"], M["ranges"], M["defaults"]
c1, c2, c3 = st.columns(3)
v = {"age": c1.selectbox("Age band", cv["age"], 1), "gender": c1.selectbox("Gender", cv["gender"]), "location": c1.selectbox("Location", cv["location"]), "language": c1.selectbox("Language", cv["language"]),
     "education": c2.selectbox("Education", cv["education"]), "device": c2.selectbox("Device", cv["device"]), "income": c2.selectbox("Income", cv["income"], 2)}
for nm, col in zip(M["num"], [c2, c3, c3, c3, c3, c3, c3]):
    lo, hi = rg[nm]
    v[nm] = col.slider(nm, float(lo), float(hi), float(dflt[nm]))
chosen = st.multiselect("Interests", M["interest_cols"], default=M["interest_cols"][:2])
x = pd.DataFrame([v])[M["num"] + M["cat"]]
Z = np.c_[S["pre"].transform(x), np.array([[int(i in chosen) for i in S["interest_cols"]]])]
seg = int(S["kmeans"].predict(Z)[0])
st.metric("k-means segment (k = 5)", seg)
prof = pd.DataFrame(M["profile"]).set_index("segment")
st.dataframe(prof, width="stretch")
st.error(f"Warning: bootstrap stability of these segments is only ARI {M['bootstrap_ARI_mean']:.2f}. Their differences in CTR and conversion are what k-means produces from noise (best silhouette {M['real_best_silhouette']:.3f} vs {M['null_best_silhouette_mean']:.3f} on shuffled data).")
footer(8, "08 User Profiling and Segmentation.ipynb")
