import json, numpy as np, pandas as pd, streamlit as st, torch, torch.nn as nn, torch.nn.functional as F
from common import header, footer, mdir, meta

header(29, "A Mini GPT from Scratch (Shakespeare)", "Deep learning",
       "A 0.8-million-parameter decoder-only transformer written from its parts and trained for 3,000 steps on 1.1 million characters. The notebook compares its validation loss with n-gram baselines, measures how much context it uses, and scores generated text for real-word rate, repetition and copying of the training text at different temperatures.",
       "A tiny model trained for minutes: the text looks like Shakespeare in rhythm and layout but is largely meaningless. Low temperature gives dull repetition, high temperature gives non-words.")
M = meta(29)
CFG = json.load(open(mdir(29) / "config.json", encoding="utf8"))
chars, D, H, L, CTX = CFG["chars"], CFG["D"], CFG["H"], CFG["L"], CFG["CTX"]
V, stoi = len(chars), {c: i for i, c in enumerate(chars)}
DROP = 0.1


class Head(nn.Module):
    def __init__(self, hs):
        super().__init__()
        self.k = nn.Linear(D, hs, bias=False); self.q = nn.Linear(D, hs, bias=False); self.v = nn.Linear(D, hs, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(CTX, CTX))); self.drop = nn.Dropout(DROP)

    def forward(self, x):
        T = x.shape[1]
        k, q, v = self.k(x), self.q(x), self.v(x)
        w = (q @ k.transpose(-2, -1)) * k.shape[-1] ** -0.5
        w = w.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        w = self.drop(F.softmax(w, -1))
        self.last = w.detach()
        return w @ v


class MHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList(Head(D // H) for _ in range(H)); self.proj = nn.Linear(D, D); self.drop = nn.Dropout(DROP)

    def forward(self, x):
        return self.drop(self.proj(torch.cat([h(x) for h in self.heads], -1)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MHA(); self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D), nn.Dropout(DROP)); self.l1, self.l2 = nn.LayerNorm(D), nn.LayerNorm(D)

    def forward(self, x):
        x = x + self.sa(self.l1(x))
        return x + self.ff(self.l2(x))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D); self.pos = nn.Embedding(CTX, D); self.blocks = nn.Sequential(*[Block() for _ in range(L)]); self.ln = nn.LayerNorm(D); self.out = nn.Linear(D, V)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        return self.out(self.ln(self.blocks(x)))


@st.cache_resource
def load():
    m = GPT()
    m.load_state_dict(torch.load(mdir(29) / "mini_gpt.pt", map_location="cpu"))
    return m.eval()


model = load()
prompt = st.text_area("Prompt", "ROMEO:\n")
c1, c2, c3 = st.columns(3)
temp = c1.slider("Temperature", 0.3, 1.5, 0.8, 0.1)
n = c2.slider("Characters to generate", 50, 600, 300, 50)
seed = c3.number_input("Random seed", 0, 9999, 1)
if st.button("Generate"):
    torch.manual_seed(int(seed))
    idx = torch.tensor([[stoi.get(c, 0) for c in prompt]])
    with torch.no_grad():
        for _ in range(n):
            lo = model(idx[:, -CTX:])[:, -1] / temp
            idx = torch.cat([idx, torch.multinomial(F.softmax(lo, -1), 1)], 1)
    st.code("".join(chars[i] for i in idx[0].tolist()), language=None)
st.write("Validation loss (nats per character) of simple models vs the transformer:")
st.dataframe(pd.Series({**M["baselines"], "mini GPT (3,000 steps)": M["val_loss"]}, name="val loss").round(3).to_frame(), width="stretch")
st.write("Generated-text measures by temperature:")
st.dataframe(pd.DataFrame(M["generation"]).T.round(1), width="stretch")
footer(29, "29 Mini GPT from Scratch.ipynb")
