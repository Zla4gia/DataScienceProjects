import json
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Skill enhancement: 30 data-science projects", layout="wide")
ROOT = Path(__file__).resolve().parent
try:
    P = json.load(open(ROOT / "projects.json", encoding="utf8"))
except FileNotFoundError:
    P = json.load(open(ROOT.parent / "_projects.json", encoding="utf8"))

st.title("Skill enhancement: 30 data-science projects")
st.markdown(
    "Thirty practice projects (three per topic) built from the project lists on *Python in Plain English* / *amanxai.com*. "
    "For each one I solved the problem with my own code, **checked it against the article's method**, and recorded where the two differ. "
    "Each page of this app lets you try the saved model, with the main finding and the caveats next to it. "
    "Pick a project in the sidebar."
)
st.warning("These are learning projects. Several of the data sets turned out to be synthetic or very small; each notebook says so. Nothing here is advice for lending, medicine, trading or any real decision.")
rows = [{"#": p[0], "Topic": p[1], "Project": p[2], "Reference article": p[3] or "(not on the lists)", "Source": p[4]} for p in P]
st.dataframe(rows, width="stretch", hide_index=True)
