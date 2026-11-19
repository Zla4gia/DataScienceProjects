import re, numpy as np, pandas as pd, streamlit as st
from common import header, footer, mdir, meta

header(26, "Multi-Agent Research Assistant with a Verifier", "AI agents",
       "Planner, researcher, writer and verifier cooperate on 12 Wikipedia topics without any paid API. The verifier (a small language-inference model) flags planted errors but also false-alarms on about two thirds of genuine sentences at threshold 0.5; the notebook measures how often an abstractive writer produces sentences its evidence does not support (an independent judge finds under 3% for every writer) and whether a verifier guard reduces that (it did not).",
       "Small models and a fixed question template: this shows the *architecture and its checking*, not the quality of an LLM-based assistant (no LLM was run). Always read the cited passage; the verifier can miss errors and flag correct sentences.")
M = meta(26)


@st.cache_resource
def base():
    from sentence_transformers import SentenceTransformer
    P = pd.read_csv(mdir(26) / "wikipedia_passages.csv.gz")
    return P, np.load(mdir(26) / "passage_embeddings.npy").astype(np.float32), SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def nli():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    t = AutoTokenizer.from_pretrained("cross-encoder/nli-MiniLM2-L6-H768")
    return t, AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-MiniLM2-L6-H768").eval()


def entail(premise, hyp):
    import torch
    t, m = nli()
    b = t(premise, hyp, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        p = torch.softmax(m(**b).logits, -1)[0]
    ie = [i for i, l in m.config.id2label.items() if l.lower().startswith("entail")][0]
    return float(p[ie])


SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"])')
sents = lambda t: [s.strip() for s in SPLIT.split(t) if len(s.split()) >= 5]
ASPECTS = ["What is {t} and how is it defined?", "What is the history of {t}?", "How does {t} work?", "What are the main uses and applications of {t}?", "What are the risks, criticisms or limitations of {t}?"]
P, PE, emb = base()
topic = st.selectbox("Topic (the knowledge base holds these 12 Wikipedia articles)", M["topics"])
thr = st.slider("Verifier threshold (probability that the evidence supports the sentence)", 0.05, 0.95, 0.5, 0.05)
abstractive = st.checkbox("Use the abstractive writer (DistilBART, downloads ~0.7 GB the first time, slow on CPU)", False)
if st.button("Write a report"):
    idx = np.where(P.topic.values == topic)[0]
    tk = bart = None
    if abstractive:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tk = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-6-6")
        bart = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-6-6").eval()
    for a in ASPECTS:
        q = a.format(t=topic)
        qe = emb.encode([q], normalize_embeddings=True)[0]
        top = idx[np.argsort(-(PE[idx] @ qe))[:3]]
        cand = [(s, int(pid)) for pid in top for s in sents(P.text[pid])]
        se = emb.encode([c[0] for c in cand], normalize_embeddings=True)
        order = np.argsort(-(se @ qe))[:4]
        ev = [cand[i] for i in order]
        premise = " ".join(s for s, _ in ev)
        if abstractive:
            b = tk(premise, truncation=True, max_length=400, return_tensors="pt")
            with torch.no_grad():
                g = bart.generate(**b, num_beams=4, max_length=70, min_length=25, no_repeat_ngram_size=3, early_stopping=True)
            out = [(s.strip(), sorted({p for _, p in ev})) for s in SPLIT.split(tk.decode(g[0], skip_special_tokens=True)) if len(s.split()) >= 4]
        else:
            out = [(s, [p]) for s, p in ev[:2]]
        st.markdown(f"#### {q}")
        for s, pids in out:
            p = entail(premise, s)
            ok = p >= thr
            st.markdown(f"{'✅' if ok else '⚠️'} {s}  \n<sub>passages {pids} · verifier support {p:.2f}{'' if ok else ' · NOT SUPPORTED at this threshold'}</sub>", unsafe_allow_html=True)
        with st.expander("Evidence used"):
            for s, pid in ev:
                st.write(f"[{pid}] ({P.section[pid]}) {s}")
st.write("Verifier test on planted errors (share flagged %):")
st.dataframe(pd.DataFrame(M["verifier_planted"]).T.round(2), width="stretch")
st.write("Unsupported sentences by writer, judged by an independent model (%):")
st.dataframe(pd.DataFrame(M["system"]).T.round(1), width="stretch")
footer(26, "26 Multi-Agent Research Assistant.ipynb")
