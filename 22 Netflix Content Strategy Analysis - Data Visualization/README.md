# 22 - Netflix Content Strategy Analysis
**Topic:** Data Visualization  |  **Where the project came from:** page list  |  **Reference:** [reference article](https://amanxai.com/2024/09/30/netflix-content-strategy-analysis-with-python/)

**Topic:** Data Visualization  |  **Reference:** "Netflix Content Strategy Analysis with Python" (amanxai.com, Sep 2024)

**Question.** What kind of content gets watched on Netflix, and does *when* a title is released matter?

**Reference method:** load a table of 24,812 titles with viewing hours, group by content type, language and release month/season, draw bar and pie charts, and read strategy lessons off them.
**What I check before drawing conclusions:** (1) the **data quality** (67% of titles have no release date; some titles appear twice; hours are written in the Indian digit-grouping style); (2) how **concentrated** viewing is (a few titles take a large share, so averages mislead); (3) release timing **with the age confound removed** (a title released in January has had more time to be watched than one released in November); (4) language and availability comparisons with intervals.

---

## 8. Summary: my approach vs the reference
| | Reference | This notebook |
|---|---|---|
| Data audit | none | 67% of rows have no release date (not at random), every repeated title is a pair with different hours (92%), hours stored as Indian-style text |
| Averages | mean hours by group | medians with bootstrap intervals (mean is 6x the median; top 10% of rows = 68% of hours) |
| Release timing | hours by month/season | restricted to 2023 releases, exposure examined: raw and per-day views both misleading; month explains about 4% of variance |
| Seasons | later seasons draw more | rise is real in the table but is survivorship |
| Conclusions | "strategy lessons" | what the table supports (concentration, shows vs movies, Korean titles) vs what it does not (timing strategy, effect of global release) |

**What I learned.** (1) Audit missing values: *which* rows are missing changes the meaning of a chart. (2) With a heavy-tailed target, report medians and concentration. (3) Time-since-release is a confounder in any "release month" chart. (4) A chart is evidence for a claim only if the comparison is fair; say what the data cannot answer.

**Limits.** No reporting period, no runtime, no genre, no budget, no marketing; two stacked rows per repeated title; 33% date coverage. **Next step:** get the reporting windows and runtimes (hours per runtime hour), then compare launches by days-in-window.

---

## Files
* `22 Netflix Content Strategy Analysis.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `meta.json`, `netflix_clean.csv.gz`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "22 Netflix Content Strategy Analysis.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
