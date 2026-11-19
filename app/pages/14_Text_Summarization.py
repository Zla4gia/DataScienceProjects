import re, numpy as np, pandas as pd, joblib, streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from common import header, footer, mdir, meta

header(14, "Text Summarization", "NLP",
       "On 120 CNN/DailyMail articles: DistilBART (fine-tuned for this data) ROUGE-1 43.0 > Lead-3 39.1 = my trained extractive model 39.3 > t5-small 36.5 (the reference's model). A learned sentence picker did not beat 'take the first three sentences'.",
       "ROUGE measures word overlap with a human summary, not truth: the generators can write sentences not supported by the article (the notebook found numbers absent from the source in 1-2% of summaries). News articles only.")
M = meta(14)
SPLIT = re.compile(r'(?<=[.!?])["”\']?\s+(?=["“\']?[A-Z0-9])')
sents = lambda t: [s.strip() for s in SPLIT.split(t) if len(s.strip()) > 1]


@st.cache_resource
def load():
    return joblib.load(mdir(14) / "extractive_gbm.joblib")["model"]


def feats(art):
    ss = sents(art)[:40]
    if not ss:
        return ss, np.zeros((0, 7))
    if len(ss) > 1:
        v = TfidfVectorizer(stop_words="english").fit(ss)
        Mx = v.transform(ss)
        sim = np.asarray(Mx @ np.asarray(Mx.mean(0)).T).ravel()
    else:
        sim = np.zeros(len(ss))
    X = np.c_[np.arange(len(ss)), np.arange(len(ss)) / len(ss), [len(s.split()) for s in ss], sim, [bool(re.search(r"\d", s)) for s in ss], [("\"" in s or "“" in s) for s in ss], np.full(len(ss), len(ss))]
    return ss, X


ex = json_ex = None
default = "Andrew Getty, one of the heirs to billions of oil money, appears to have died of natural causes, a Los Angeles Police Department spokesman said. The coroner's preliminary assessment is there was no foul play involved in the death of Getty, grandson of oil tycoon J. Paul Getty, said Detective Meghan Aguilar. Andrew Getty, 47, had several health issues, Aguilar said, adding that an autopsy will be conducted. There is no criminal investigation underway, he said. Some medication had also been recovered from Getty's home, though investigators don't know whether Getty was taking it. Getty was found dead on Tuesday. His death was reported to police by a friend."
txt = st.text_area("Paste a news article (English)", default, height=220)
ss, X = feats(txt)
if len(ss) < 4:
    st.warning("Paste at least four sentences.")
else:
    gbm = load()
    p = gbm.predict_proba(X)[:, 1]
    top = sorted(np.argsort(-p)[:3])
    c1, c2 = st.columns(2)
    c1.subheader("Lead-3 (first three sentences)")
    c1.write(" ".join(ss[:3]))
    c2.subheader("Trained extractive model (3 sentences)")
    c2.write(" ".join(ss[i] for i in top))
    st.caption("Both only copy sentences, so they cannot invent facts. The notebook found they score about the same (ROUGE-1 39.1 vs 39.3).")
    if st.button("Also generate an abstractive summary with DistilBART (downloads ~0.7 GB the first time; slow on CPU)"):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tk = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-6-6")
        m = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-6-6").eval()
        b = tk(txt, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            g = m.generate(**b, num_beams=4, max_length=80, min_length=25, no_repeat_ngram_size=3, early_stopping=True)
        out = tk.decode(g[0], skip_special_tokens=True)
        st.subheader("DistilBART")
        st.write(out)
        nums = lambda s: set(re.findall(r"\d[\d,\.]*\d|\d", s))
        new = nums(out) - nums(txt)
        st.write("Numbers in the summary that do not appear in the article:", sorted(new) if new else "none (this crude check can miss other kinds of errors)")
st.subheader("ROUGE on 120 test articles (95% bootstrap intervals)")
st.dataframe(pd.DataFrame(M["rouge"]).T, width="stretch")
st.dataframe(pd.DataFrame(M["style"]).T, width="stretch")
footer(14, "14 Text Summarization.ipynb")
