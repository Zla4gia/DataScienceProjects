import re, json, numpy as np, pandas as pd, streamlit as st, torch, torch.nn as nn, torch.nn.functional as F
from common import header, footer, mdir, meta

header(28, "Predictive Keyboard (LSTM)", "Deep learning",
       "A two-layer word-level LSTM trained on WikiText-2 against a word-frequency baseline and an interpolated trigram model: see the notebook for perplexity, top-1/3/5 accuracy and the keystroke-saving simulation (share of keystrokes saved by a 3-suggestion bar).",
       "Small model trained on encyclopedia text, 8,000-word vocabulary: it suggests Wikipedia-style words, not your style. Rare words map to <unk> and are never suggested.")
M = meta(28)


class LM(nn.Module):
    def __init__(self, V, d=200, layers=2, drop=0.3):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.rnn = nn.LSTM(d, d, layers, dropout=drop, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.out = nn.Linear(d, V)
        self.out.weight = self.emb.weight

    def forward(self, x, h=None):
        e = self.drop(self.emb(x))
        o, h = self.rnn(e, h)
        return self.out(self.drop(o)), h


@st.cache_resource
def load():
    vocab = json.load(open(mdir(28) / "vocab.json", encoding="utf8"))["vocab"]
    m = LM(len(vocab))
    m.load_state_dict(torch.load(mdir(28) / "keyboard_lstm.pt", map_location="cpu"))
    return m.eval(), vocab


model, vocab = load()
w2i = {w: i for i, w in enumerate(vocab)}
isalpha = np.array([w.isalpha() for w in vocab])
text = st.text_input("Start typing (the keyboard suggests the next word)", "the capital of the united")
typed = st.text_input("Letters of the next word typed so far (optional)", "")
toks = re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[.,;:!?()\"-]", text.lower())
if toks:
    with torch.no_grad():
        lo, _ = model(torch.tensor([w2i.get(w, 0) for w in toks]).view(1, -1))
    p = F.softmax(lo[0, -1], -1).numpy()
    p[0] = 0
    mask = isalpha & np.array([w.startswith(typed.lower()) for w in vocab])
    cand = np.where(mask)[0]
    top = cand[np.argsort(-p[cand])[:5]]
    c = st.columns(5)
    for col, i in zip(c, top):
        col.metric(vocab[i], f"{p[i] * 100:.1f}%")
    st.caption("Probability of each suggestion (before filtering by the typed letters, over the whole 8,000-word vocabulary).")
st.write("Test results (WikiText-2 test text):")
st.dataframe(pd.DataFrame(M["test"]).T.round(2), width="stretch")
st.write("Keystrokes saved with a 3-suggestion bar (%):", M["keystrokes_saved_pct"])
footer(28, "28 Predictive Keyboard LSTM.ipynb")
