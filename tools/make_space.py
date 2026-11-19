"""Build the Hugging Face Space folder (hf_space/) from the repository.

    python tools/make_space.py            -> creates ./hf_space  (app + only the small model files the app needs)
    python tools/make_space.py --check    -> also prints the folder size and the largest files

Layout produced (what Hugging Face expects for a Streamlit Space):
    hf_space/README.md           (YAML header: sdk streamlit, app_file app.py)
    hf_space/app.py, common.py, projects.json, pages/*.py
    hf_space/requirements.txt
    hf_space/models/NN/...       (copy of NN/model without files larger than the limit)
"""
import shutil, sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "hf_space"
LIMIT = 40 * 1024 * 1024          # skip individual files above 40 MB (Spaces handle large files with Git LFS, but small is faster)
SKIP_DIRS = {"distilbert_20ng"}

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir()
app = ROOT / "app"
for f in ["app.py", "common.py", "projects.json"]:
    shutil.copy(app / f, OUT / f)
shutil.copytree(app / "pages", OUT / "pages", ignore=shutil.ignore_patterns("__pycache__"))
skipped = []
for n in range(1, 31):
    folder = next(iter(sorted(ROOT.glob(f"{n:02d} *"))), ROOT / f"{n:02d}")
    src = folder / "model"
    if not src.exists():
        continue
    for p in src.rglob("*"):
        if p.is_dir() or any(d in p.parts for d in SKIP_DIRS):
            continue
        if p.stat().st_size > LIMIT:
            skipped.append((str(p.relative_to(ROOT)), round(p.stat().st_size / 1e6, 1)))
            continue
        dst = OUT / "models" / f"{n:02d}" / p.relative_to(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, dst)
(OUT / "requirements.txt").write_text("""--extra-index-url https://download.pytorch.org/whl/cpu
streamlit>=1.40
pandas>=3.0
numpy>=2.0
scipy>=1.14
statsmodels>=0.14
scikit-learn==1.9.0
joblib>=1.4
lightgbm>=4.5
xgboost>=3.0
torch>=2.5
torchvision>=0.20
transformers>=5.0
sentence-transformers>=3.0
scikit-image>=0.24
pillow>=10
pyarrow>=15
streamlit-drawable-canvas>=0.9
gymnasium>=1.0
pygame>=2.5
""", encoding="utf8")
(OUT / "README.md").write_text("""---
title: Skill enhancement - 30 data science projects
emoji: "📊"
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.58.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Try the saved models of 30 practice projects, with their caveats
---

# Skill enhancement: 30 data-science projects

Interactive pages for 30 practice projects (regression, classification, clustering, computer vision, NLP,
recommendation, time series, data visualisation, AI agents, deep learning). Each page shows the main finding
of its notebook next to the caveats, and lets you try the saved model.
Source code, notebooks and the comparison with each reference article: see the GitHub repository linked in the app.
""", encoding="utf8")
total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
print("built", OUT, "| files:", sum(1 for p in OUT.rglob("*") if p.is_file()), "| size: %.1f MB" % (total / 1e6))
if skipped:
    print("skipped (too large):", skipped)
if "--check" in sys.argv:
    big = sorted(((p.stat().st_size, p) for p in OUT.rglob("*") if p.is_file()), reverse=True)[:8]
    for s, p in big:
        print("%8.1f MB  %s" % (s / 1e6, p.relative_to(OUT)))
