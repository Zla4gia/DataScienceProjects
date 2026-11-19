"""Build the two free Hugging Face artefacts (no compute Space needed):

  hf_models/   a MODEL repository: every project's saved models (model/ folder) + a model card (README.md)
  hf_static/   a STATIC Space: one HTML landing page (project table, findings, links) + Space README

Usage:  python tools/make_hf_files.py          (build only)
Then:   python tools/upload_hf.py <hf-username> (after `hf auth login`)
"""
import html, json, re, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/Zla4gia/DataScienceProjects"
OUT_M, OUT_S = ROOT / "hf_models", ROOT / "hf_static"
for d in (OUT_M, OUT_S):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir()

projects = json.load(open(ROOT / "_projects.json", encoding="utf8"))
# one-line findings from the top-level README table
findings = {}
for line in (ROOT / "README.md").read_text(encoding="utf8").splitlines():
    m = re.match(r"\|\s*(\d\d)\s*\|\s*([^|]+?)\s*\|\s*\[([^\]]+)\]\([^)]*\)\s*\|\s*(.+?)\s*\|\s*$", line)
    if m:
        findings[int(m.group(1))] = m.group(4)

rows, total = [], 0
for n, topic, name, url, src in projects:
    folder = next(iter(sorted(ROOT.glob(f"{n:02d} *"))), None)
    if folder is None or not (folder / "model").exists():
        continue
    slug = re.sub(r"[^a-z0-9]+", "-", folder.name.split(" - ")[0][3:].lower()).strip("-")
    dst = OUT_M / f"{n:02d}-{slug}"
    shutil.copytree(folder / "model", dst, ignore=shutil.ignore_patterns("__pycache__"))
    files = [p for p in dst.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    total += size
    rows.append((n, topic, name, f"{n:02d}-{slug}", len(files), size, findings.get(n, ""), folder.name))

card = f"""---
library_name: sklearn
tags:
- tabular
- nlp
- computer-vision
- time-series
- recommendation
- reinforcement-learning
- educational
---

# Skill enhancement: saved models for 30 data-science projects

Trained models, fitted pipelines and small artefacts from 30 practice projects (three per topic: regression, classification, clustering, computer vision, NLP, recommendation systems, time series, data visualization, AI agents, deep learning). Each project solves a problem from the *Python in Plain English* / *amanxai.com* project lists, compares the result with the reference article, and records the limits.

**These are learning projects, not products.** Several data sets are small, old or synthetic. Nothing here is advice for lending, medicine, trading or decisions about real people.

* Code, executed notebooks and the Streamlit app: {GITHUB} (currently private)
* Each folder below matches a project folder in that repository (`<NN> <project> - <topic>`).

## Safety note on file formats
`*.joblib` files are Python pickles made with scikit-learn 1.9.0. **Do not load pickles from sources you do not trust**: loading one can run arbitrary code. `*.pt` files are PyTorch weights saved with `torch.save`, which is also pickle-based (load them with `torch.load(..., weights_only=True)` where the file holds only tensors). `*.safetensors`, `*.json`, `*.csv` and `*.npy` are plain data. Several pickles refer to helper functions defined in the project's notebook, so load them inside the matching notebook or app page.

## Contents
Total size: {total/1e6:.0f} MB.

| Folder | Topic | Project | Files | Size | What the notebook found |
|---|---|---|---|---|---|
""" + "\n".join(f"| `{f}` | {t} | {nm} | {c} | {s/1e6:.1f} MB | {fi} |" for n, t, nm, f, c, s, fi, _ in rows) + """

## Loading an example
```python
from huggingface_hub import snapshot_download
import joblib
path = snapshot_download("<your-username>/skill-enhancement-models", allow_patterns="01-*/*")
glm = joblib.load(path + "/01-car-insurance-modelling/freq_glm.joblib")   # only if you trust this repository
```

## Data and licences
Each notebook states where its data come from; the data belong to their providers. A few folders contain small derived files (for example cleaned tables, a passage index with Wikipedia text under CC BY-SA 4.0). Check the source licence before reuse.
"""
(OUT_M / "README.md").write_text(card, encoding="utf8")

# ---- static landing page -------------------------------------------------------------------
tr = "\n".join(
    f"<tr><td>{n:02d}</td><td>{html.escape(t)}</td><td>{html.escape(nm)}</td><td>{html.escape(fi)}</td></tr>"
    for n, t, nm, f, c, s, fi, fo in rows)
page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Skill enhancement: 30 data-science projects</title>
<style>
:root {{ --bg:#ffffff; --fg:#1b2430; --muted:#5b6675; --line:#d9dee5; --accent:#1f4e79; --head:#f1f4f8; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#12161c; --fg:#e6e9ee; --muted:#9aa5b4; --line:#2a313b; --accent:#7fb0e3; --head:#1a2029; }} }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:16px/1.55 system-ui, sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ font-size:1.7rem; margin:0 0 8px; }} p, li {{ color:var(--fg); }} .muted {{ color:var(--muted); }}
a {{ color:var(--accent); }}
.note {{ border:1px solid var(--line); border-left:4px solid var(--accent); padding:10px 14px; background:var(--head); margin:16px 0; }}
.wrap {{ overflow-x:auto; }} table {{ border-collapse:collapse; width:100%; min-width:720px; }}
th, td {{ border:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }} th {{ background:var(--head); }}
</style></head><body><main>
<h1>Skill enhancement: 30 data-science projects</h1>
<p class="muted">Three projects per topic: regression, classification, clustering, computer vision, NLP, recommendation systems, time series, data visualization, AI agents and deep learning. Each one solves a problem from a reference tutorial, compares the result with it, and records the limits.</p>
<div class="note"><strong>Learning projects, not products.</strong> Several data sets are small, old or synthetic and each notebook says so. Nothing here is advice for lending, medicine, trading or decisions about real people. The interactive Streamlit app is in the code repository; this page is a static summary.</div>
<p>Code and notebooks: <a href="{GITHUB}">{GITHUB}</a> (currently private). Saved models: the model repository <code>skill-enhancement-models</code> on this account.</p>
<div class="wrap"><table>
<thead><tr><th>#</th><th>Topic</th><th>Project</th><th>What the notebook found</th></tr></thead>
<tbody>
{tr}
</tbody></table></div>
</main></body></html>
"""
(OUT_S / "index.html").write_text(page, encoding="utf8")
(OUT_S / "README.md").write_text("""---
title: Skill enhancement - 30 data-science projects
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
---

Static summary page for 30 data-science practice projects (see index.html). The code, notebooks and Streamlit app are in the linked GitHub repository.
""", encoding="utf8")

print(f"model repo: {len(rows)} project folders, {total/1e6:.0f} MB -> {OUT_M}")
print(f"static space -> {OUT_S}")
for n, t, nm, f, c, s, fi, fo in rows:
    if s > 40e6:
        print("  large:", f, f"{s/1e6:.0f} MB")
missing = [n for n in range(1, 31) if n not in {r[0] for r in rows}]
print("projects without a model folder:", missing)
