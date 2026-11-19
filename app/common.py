"""Shared helpers for the Skill-enhancement Streamlit app.

Models live in  <repo>/NN/model/   (local layout)  or  <space>/models/NN/  (Hugging Face Space layout).
"""
import json, sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
REPO_URL = "https://github.com/YOUR-GITHUB-USERNAME/skill-enhancement"      # replaced when the repository is published


def mdir(n: int) -> Path:
    for p in (ROOT / f"{n:02d}" / "model", ROOT / "models" / f"{n:02d}", APP_DIR / "models" / f"{n:02d}"):
        if p.exists():
            return p
    raise FileNotFoundError(f"model folder for project {n:02d} not found")


def meta(n: int) -> dict:
    return json.load(open(mdir(n) / "meta.json", encoding="utf8"))


def register_main(**objs):
    """Pickles written inside notebooks refer to functions/classes in __main__; make them importable."""
    import __main__
    for k, v in objs.items():
        setattr(__main__, k, v)


def header(n: int, title: str, topic: str, finding: str, caveat: str):
    st.set_page_config(page_title=f"{n:02d} {title}", page_icon=None, layout="wide")
    st.title(f"{n:02d} · {title}")
    st.caption(f"Topic: {topic}")
    st.markdown(f"**What the notebook found.** {finding}")
    st.info(f"**Read with care.** {caveat}")


def footer(n: int, notebook: str):
    st.divider()
    st.caption(f"Full analysis, charts and comparison with the reference article: `{n:02d}/{notebook}` in the repository ({REPO_URL}).")
