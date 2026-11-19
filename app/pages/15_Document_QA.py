import re, numpy as np, pandas as pd, joblib, streamlit as st
from common import header, footer, mdir, meta

header(15, "Document Analysis: GitHub Terms of Service", "NLP",
       "Retrieval with sentence embeddings found the right passage in the top 3 for 14 of 14 verified questions; combining the reader's confidence with the retrieval weight gave the right passage 93% of the time (F1 0.60). A reference-style summary of the first 512 tokens covers 0 of 14 key facts (per-section summaries: 8).",
       "The reader (DistilBERT-SQuAD) cannot say 'I don't know': it answers every question, including ones the document does not cover. Not legal advice; check the answer against the cited passage.")
M = meta(15)
D = joblib.load(mdir(15) / "gh_tos_index.joblib")
P = pd.DataFrame(D["passages"])
PE = np.load(mdir(15) / "gh_tos_passage_embeddings.npy").astype(np.float32)


@st.cache_resource
def models():
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    tk = AutoTokenizer.from_pretrained("distilbert/distilbert-base-cased-distilled-squad")
    qa = AutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-cased-distilled-squad").eval()
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"), tk, qa


def read(tk, qam, q, ctx, max_ans=40):
    import torch
    enc = tk(q, ctx, return_tensors="pt", truncation="only_second", max_length=384, return_offsets_mapping=True)
    offs = enc.pop("offset_mapping")[0].numpy()
    seq = np.array([x if x is not None else -1 for x in enc.sequence_ids(0)])
    with torch.no_grad():
        o = qam(**enc)
    m = torch.from_numpy(seq == 1)
    ps = torch.softmax(o.start_logits[0].masked_fill(~m, -1e9), 0).numpy()
    pe = torch.softmax(o.end_logits[0].masked_fill(~m, -1e9), 0).numpy()
    sc = np.tril(np.triu(np.outer(ps, pe)), max_ans)
    i, j = np.unravel_index(sc.argmax(), sc.shape)
    return ctx[offs[i][0]:offs[j][1]], float(sc[i, j])


st.write("Sample questions:", [g[0] for g in M["gold"][:4]], "| unanswerable:", M["unanswerable"][:2])
q = st.text_input("Ask a question about the GitHub Terms of Service", M["gold"][0][0])
if q:
    emb, tk, qam = models()
    qe = emb.encode([q], normalize_embeddings=True)[0]
    s_e = PE @ qe
    s_t = (D["tfidf"].transform(P.text) @ D["tfidf"].transform([q]).T).toarray().ravel()
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    h = z(s_t) + z(s_e)
    top = np.argsort(-h)[:3]
    w = np.exp(h[top] - h[top].max())
    w /= w.sum()
    cands = []
    for i, wi in zip(top, w):
        a, sc = read(tk, qam, q, P.text[i])
        cands.append((sc * wi, a, sc, int(i)))
    best = max(cands)
    st.success(f"**{best[1]}**")
    st.caption(f"Section: {P.section[best[3]]} | reader confidence {best[2]:.2f} | similarity of the best passage to the question {float(s_e.max()):.2f}")
    st.write("Source passage:", P.text[best[3]])
    st.warning("The reader always returns a span. If the question is not about this document the answer is meaningless: check the similarity above and the passage.")
st.write("Retrieval (hit %):", M["retrieval_hit_pct"])
st.write("QA on 14 verified questions:", M["qa"], "| coverage of key facts by summary type:", M["coverage"])
footer(15, "15 Document Analysis.ipynb")
