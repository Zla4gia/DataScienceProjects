import numpy as np, pandas as pd, streamlit as st
from common import header, footer, mdir, meta

header(16, "Music Recommendation System", "Recommendation systems",
       "From 10 audio numbers alone, the best representation (a learned embedding) puts a song of the same genre in the top 10 for 16.6% of queries (standardised features 15.4%, raw numbers 5.3%, random songs 1.2%). Scaling the features mattered more than any model. It works for distinctive styles (classical) and poorly for mainstream pop/rock.",
       "No listener data. Genre agreement is a proxy for taste, not proof that you would enjoy a song.")
M = meta(16)


@st.cache_resource
def load():
    S = pd.read_csv(mdir(16) / "songs.csv.gz")
    return S, np.load(mdir(16) / "embeddings.npy").astype(np.float32)


S, E = load()
q = st.text_input("Search a song or artist", "Clair de Lune")
hits = S[S.track_name.str.contains(q, case=False, regex=False) | S.artists.str.contains(q, case=False, regex=False)].sort_values("popularity", ascending=False).head(30)
if len(hits) == 0:
    st.warning("No match; try another spelling.")
else:
    pick = st.selectbox("Pick the song", hits.index, format_func=lambda i: f"{S.track_name[i]} - {S.artists[i]} [{S.genre[i]}]")
    s = E @ E[pick]
    s[pick] = -9
    top = np.argsort(-s)[:10]
    st.subheader("Songs with the most similar audio profile")
    st.dataframe(S.loc[top, ["track_name", "artists", "genre", "popularity"]].assign(similarity=np.round(s[top], 3)).reset_index(drop=True), width="stretch")
    st.caption(f"Seed genres: {S.genres[pick].replace('|', ', ')}. Neighbours in other genres are common.")
st.write("Notebook evaluation (held-out songs, 10 recommendations each):")
st.dataframe(pd.DataFrame(M["evaluation"]).T.round(3), width="stretch")
footer(16, "16 Music Recommendation System.ipynb")
