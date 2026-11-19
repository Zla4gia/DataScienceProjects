import joblib, pandas as pd, streamlit as st
from common import header, footer, mdir, meta

header(13, "Text Classification Pipeline (20 Newsgroups)", "Natural language processing",
       "TF-IDF + linear models, sentence embeddings + logistic regression and a fine-tuned DistilBERT are compared on 20 Newsgroups. The notebook also shows how much the headline accuracy depends on the split: random splits and leftover e-mail headers inflate the score.",
       "Newsgroup posts are old, informal and topic-based. A high score here says little about other kinds of text. Predictions on very short or off-topic text are unreliable.")
M = meta(13)


@st.cache_resource
def load_tfidf():
    return joblib.load(mdir(13) / "tfidf_logreg.joblib")


@st.cache_resource
def load_embed():
    from sentence_transformers import SentenceTransformer
    return joblib.load(mdir(13) / "minilm_logreg.joblib"), SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


kind = st.radio("Model", ["TF-IDF + logistic regression", "MiniLM embeddings + logistic regression"], horizontal=True)
txt = st.text_area("Text", "The new graphics card renders polygons much faster than the last generation, but the driver crashes under Windows.", height=140)
if st.button("Classify") and txt.strip():
    if kind.startswith("TF-IDF"):
        D = load_tfidf()
        p = D["clf"].predict_proba(D["vectorizer"].transform([txt]))[0]
    else:
        D, enc = load_embed()
        p = D["clf"].predict_proba(enc.encode([txt], normalize_embeddings=True))[0]
    top = pd.Series(p, index=D["names"]).sort_values(ascending=False).head(5)
    st.bar_chart(top)
    st.caption(f"Top class: {top.index[0]} ({top.iloc[0]:.0%}). With 20 classes, a flat distribution means the model is unsure.")
st.write("Test accuracy of each model (official train/test split):")
st.dataframe(pd.DataFrame(M["test"]).T.round(3), width="stretch")
st.write("Accuracy (%) of the same task under different splitting set-ups (leakage experiment):")
st.dataframe(pd.Series(M["setup"], name="accuracy %").to_frame(), width="stretch")
footer(13, "13 Text Classification Pipeline.ipynb")
