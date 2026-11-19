import numpy as np, pandas as pd, joblib, streamlit as st, torch, torch.nn as nn
from scipy import stats
from common import header, footer, mdir, meta

header(30, "Synthetic Medical Records (GAN)", "Deep learning",
       "Four ways of making fake patient records from the Pima diabetes table are scored on fidelity, utility (train on synthetic, test on real) and privacy (distance to the nearest real patient, membership attack): copying real rows is perfect on the first two and fails privacy; the GAN and a Gaussian copula trade off between them. See the table below for the measured numbers.",
       "768 patients, 8 measurements. 'Synthetic' does not mean 'private': always check distances to real records. Do not use these rows for medical research or to represent real people.")
M = meta(30)
S = joblib.load(mdir(30) / "gan_support.joblib")
cols, INT, tr, qt = S["columns"], S["integer_columns"], S["train"], S["quantile_transformer"]


class G(nn.Module):
    def __init__(self, zd=32, o=9):
        super().__init__()
        self.n = nn.Sequential(nn.Linear(zd, 128), nn.LeakyReLU(.2), nn.Linear(128, 128), nn.LeakyReLU(.2), nn.Linear(128, o))

    def forward(self, z):
        return self.n(z)


@st.cache_resource
def load():
    g = G()
    g.load_state_dict(torch.load(mdir(30) / "gan_generator.pt", map_location="cpu"))
    return g.eval()


g = load()
n = st.slider("Records to generate", 50, 1000, 300, 50)
seed = st.number_input("Seed", 0, 9999, 1)
if st.button("Generate with the GAN"):
    z = torch.randn(n, 32, generator=torch.Generator().manual_seed(int(seed)))
    with torch.no_grad():
        arr = g(z).numpy()
    d = pd.DataFrame(qt.inverse_transform(arr), columns=cols)
    for c in INT:
        d[c] = d[c].round().clip(lower=0)
    d["outcome"] = (d.outcome > 0.5).astype(int)
    st.dataframe(d.head(20), width="stretch")
    ks = {c: round(stats.ks_2samp(tr[c], d[c]).statistic, 3) for c in cols[:-1]}
    st.write("Kolmogorov-Smirnov distance to the real training columns (0 = identical):", ks)
    sc = tr[cols[:-1]].std().replace(0, 1)
    A, B = (d[cols[:-1]] / sc).to_numpy(), (tr[cols[:-1]] / sc).to_numpy()
    dist = np.sqrt(((A[:, None, :] - B[None]) ** 2).sum(-1)).min(1)
    st.metric("Median distance to the nearest real training patient", f"{np.median(dist):.2f}", f"{(dist < 0.05).mean():.1%} are near-copies (< 0.05)", delta_color="off")
    st.download_button("Download CSV", d.to_csv(index=False), "synthetic_records.csv")
st.write("Scores from the notebook (hold-out = 231 real patients; 'resample' copies real rows):")
st.dataframe(pd.DataFrame(M["scores"]).T.round(3), width="stretch")
st.write("Reference utility (train on real, test on real hold-out, AUC):", M["trtr"])
footer(30, "30 Synthetic Medical Records GAN.ipynb")
