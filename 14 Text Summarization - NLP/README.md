# 14 - Text Summarization
**Topic:** NLP  |  **Where the project came from:** page list  |  **Reference:** [reference article](https://amanxai.com/2024/10/07/text-summarization-model-using-llms/)

**Topic:** NLP  |  **Reference:** "Text Summarization Model using LLMs" (amanxai.com, Oct 2024)

**Question.** Can a model shorten a news article into a few sentences that keep the main points, and how would we know?

**Reference method:** load `t5-small`, summarise **one** hand-picked text, look at the output. No score, no comparison.
**What I do:** evaluate on **120 real CNN/DailyMail test articles** with their human-written highlights, using ROUGE (word-overlap with the reference summary), against four systems: **Lead-3** (just the first three sentences), a **supervised extractive model I train myself** (gradient boosting that picks sentences), **t5-small** (the reference's model) and **DistilBART-CNN** (a model fine-tuned for this exact data). I add two checks ROUGE cannot give: how much of each summary is **copied vs newly written**, and whether any **number in a summary is missing from the article** (a simple test for invented facts). Bootstrap confidence intervals show which differences are real.

---

## 8. Summary: my approach vs the reference
| | Reference | This notebook |
|---|---|---|
| Evaluation | one hand-picked text, read by eye | 120 CNN/DailyMail test articles, ROUGE-1/2/L with bootstrap intervals |
| Baselines | none | Lead-3 and a supervised extractive model I trained (110,732 sentences) |
| Models | t5-small | t5-small, DistilBART-CNN-6-6 |
| Result | a summary | DistilBART 43.0 > Lead-3 39.1 = my extractor 39.3 > t5-small 36.5 (ROUGE-1) |
| Extra checks | none | copying vs writing (10-11% new word pairs for neural models), unsupported-number alarm (0-1.7%) |

**What I learned.** (1) Without a baseline a summary score means nothing: the reference's t5-small does *worse* than copying the first three sentences. (2) A model trained for the task (DistilBART on CNN/DM) is clearly better than a general model. (3) A learned sentence picker did not beat a position rule: check that complexity buys something. (4) ROUGE measures overlap, not truth; add checks such as the number alarm.

**Limits.** 120 articles (intervals about +/-2 points); only the first 512 tokens are read by the neural models; ROUGE-Lsum splitting uses my own sentence splitter; no human judgement of fluency or accuracy; the number check is crude and the 3 flagged cases were not verified. **Next step:** a larger sample, human ratings on a subset, and a faithfulness metric based on entailment.

---

## Files
* `14 Text Summarization.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `examples.json`, `extractive_gbm.joblib`, `meta.json`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "14 Text Summarization.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
