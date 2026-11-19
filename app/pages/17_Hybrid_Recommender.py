import numpy as np, pandas as pd, joblib, streamlit as st
from scipy import sparse
from common import header, footer, mdir, meta

header(17, "Hybrid Recommendation System (MovieLens 100K)", "Recommendation systems",
       "With a time-based hold-out, the blend of item-kNN, SVD and genre content scores NDCG@10 0.189 vs 0.175 (SVD) and 0.174 (item-kNN): a small but consistent gain (paired bootstrap). Popularity wins for brand-new users, and a model that predicts ratings best (RMSE 0.976) ranks poorly (NDCG@10 0.052).",
       "100K ratings from 943 users on 1,682 movies; genres are the only content. Tell the app a few movies you like and judge the list yourself.")
M = meta(17)


@st.cache_resource
def load():
    D = joblib.load(mdir(17) / "hybrid_recommender.joblib")
    X = D["P"].T.tocsr().astype(float)
    nrm = np.sqrt(np.asarray(X.multiply(X).sum(1))).ravel() + 1e-9
    Sim = (X @ X.T).toarray() / np.outer(nrm, nrm)
    np.fill_diagonal(Sim, 0)
    return D, Sim


D, Sim = load()
titles, w, CONT, svd, P = D["titles"], D["weights"], D["content"], D["svd"], D["P"]
liked = st.multiselect("Movies you like", titles, default=[t for t in ["Star Wars (1977)", "Raiders of the Lost Ark (1981)", "Empire Strikes Back, The (1980)"] if t in titles])
if liked:
    ids = [titles.index(t) for t in liked]
    p = np.zeros(len(titles))
    p[ids] = 1
    prof = p @ CONT
    prof /= np.linalg.norm(prof) + 1e-9
    c = prof @ CONT.T
    kn = p @ Sim
    sv = svd.inverse_transform(svd.transform(sparse.csr_matrix(p[None, :])))[0]
    pv = np.asarray(P.sum(0)).ravel().astype(float)
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    s = w[0] * z(pv) + w[1] * z(c) + w[2] * z(kn) + w[3] * z(sv)
    s[ids] = -np.inf
    top = np.argsort(-s)[:10]
    st.subheader("Recommended")
    st.dataframe(pd.DataFrame({"movie": [titles[i] for i in top], "blend score": np.round(s[top], 2), "item-kNN": np.round(z(kn)[top], 2), "SVD": np.round(z(sv)[top], 2), "genres": np.round(z(c)[top], 2)}), hide_index=True, width="stretch")
st.write("Blend weights tuned on a validation slice:", M["weights"])
st.write("Test results:")
st.dataframe(pd.DataFrame(M["test"]).T.round(4), width="stretch")
st.write("Cold items / new users (NDCG@10):")
st.dataframe(pd.DataFrame(M["cold"]).T.round(4), width="stretch")
footer(17, "17 Hybrid Recommendation System.ipynb")
