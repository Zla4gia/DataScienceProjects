import numpy as np, pandas as pd, streamlit as st
from common import header, footer, mdir, meta

header(22, "Netflix Content Strategy Analysis", "Data visualization",
       "Viewing is extremely concentrated (top 1% of rows = 24% of hours, top 10% = 68%, Gini 0.80; mean 6x the median). 67% of rows have no release date, every repeated title is a pair with different hours, and the release-month pattern cannot be separated from the unseen reporting window: the data do not support a 'release strategy'.",
       "Rows are title-seasons from stacked reports; 'hours per row' mixes films with whole seasons. Associations only.")
M = meta(22)


@st.cache_data
def load():
    d = pd.read_csv(mdir(22) / "netflix_clean.csv.gz", parse_dates=["release"])
    return d


d = load()
c1, c2, c3 = st.columns(3)
typ = c1.multiselect("Type", sorted(d.type.unique()), default=sorted(d.type.unique()))
lang = c2.multiselect("Language", sorted(d.language.unique()), default=sorted(d.language.unique()))
gl = c3.multiselect("Available globally", sorted(d["global"].unique()), default=sorted(d["global"].unique()))
f = d[d.type.isin(typ) & d.language.isin(lang) & d["global"].isin(gl)]
a, b, c = st.columns(3)
a.metric("Rows", f"{len(f):,}")
b.metric("Median hours (millions)", f"{f.hours_m.median():.2f}")
c.metric("Mean hours (millions)", f"{f.hours_m.mean():.2f}")
srt = f.hours_m.sort_values(ascending=False).to_numpy()
if len(srt) > 10:
    cum = srt.cumsum() / srt.sum()
    st.line_chart(pd.DataFrame({"share of hours": cum}, index=np.arange(1, len(cum) + 1) / len(cum) * 100), height=240)
    st.caption(f"Concentration curve for the filtered rows: the top 10% of rows account for {cum[int(.1 * len(cum)) - 1]:.0%} of their hours.")
st.dataframe(f.sort_values("hours_m", ascending=False).head(25)[["Title", "type", "language", "global", "release", "hours_m"]], hide_index=True, width="stretch")
st.write("Medians with bootstrap intervals from the notebook:")
st.dataframe(pd.DataFrame(M["groups"]), hide_index=True, width="stretch")
st.write("Release-month tests (2023 releases):", M["month_test"])
footer(22, "22 Netflix Content Strategy Analysis.ipynb")
