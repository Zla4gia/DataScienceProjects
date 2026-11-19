import numpy as np, pandas as pd, joblib, streamlit as st
from common import header, footer, mdir, meta

header(18, "Market Basket Analysis", "Recommendation systems",
       "On 9,835 grocery baskets the association rules are real (1,325 rules with lift >= 1.5, against 0.4 in shuffled baskets; top rules keep about 88% of their lift on new baskets). But for completing a basket they are no better than 'suggest popular items' (26-27% vs 25.8% hit rate@5). On the reference's 250-basket file the rules are noise.",
       "Rank rules by lift, not confidence (confidence just rediscovers whole milk). A rule says items co-occur, not that one causes the other.")
M = meta(18)


@st.cache_resource
def load():
    return joblib.load(mdir(18) / "basket_cooccurrence.joblib"), pd.read_csv(mdir(18) / "groceries_rules.csv")


C, R = load()
items = C["items"]
cnt, co = np.asarray(C["counts"], float), np.asarray(C["cooc"], float)
basket = st.multiselect("Items in the basket", items, default=["yogurt", "tropical fruit"])
if basket:
    idx = [items.index(i) for i in basket]
    conf = co / np.maximum(cnt[:, None], 1)
    pj = cnt / C["n_baskets"]
    lift = conf / np.maximum(pj[None, :], 1e-9)
    lift[co < C["min_cooc"]] = 0
    s_lift = (conf[idx] * lift[idx]).sum(0)
    s_pop = cnt.copy()
    s_conf = conf[idx].max(0)
    for s in (s_lift, s_pop, s_conf):
        s[idx] = -1
    cols = st.columns(3)
    for col, (nm, s) in zip(cols, [("Popularity", s_pop), ("Max confidence", s_conf), ("Lift-weighted", s_lift)]):
        col.subheader(nm)
        col.write([items[i] for i in np.argsort(-s)[:5]])
    st.caption("In the notebook's hold-out test the three lists find a hidden item 25.8%, 26.4% and 26.0% of the time: statistically indistinguishable.")
q = st.text_input("Filter rules containing an item", "ham")
sub = R[R.antecedents.str.contains(q, case=False) | R.consequents.str.contains(q, case=False)].head(25) if q else R.head(25)
st.dataframe(sub.round(3), hide_index=True, width="stretch")
st.write("Hold-out basket-completion hit rate @5 (%):", M["hit_rate"])
footer(18, "18 Market Basket Analysis.ipynb")
