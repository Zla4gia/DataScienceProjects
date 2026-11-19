# 13 - Text Classification Pipeline with Hugging Face Transformers
**Topic:** NLP  |  **Where the project came from:** page list  |  **Reference:** [reference article](https://amanxai.com/2025/07/22/text-classification-pipeline-with-hugging-face-transformers/)

**Topic:** NLP  |  **Reference:** "Building a Text Classification Pipeline with Hugging Face Transformers" (amanxai.com, Jul 2025)

**Question.** Which of 20 newsgroup topics does a post belong to, and how much does a transformer really add over a classic bag-of-words model?

**Reference method:** the 20 Newsgroups data (headers, footers and quotes removed), a **random 80/20 split** of all 18,846 posts, DistilBERT fine-tuned with the Hugging Face `Trainer`.
**What I add:** (1) the **same data three ways** to measure how much the evaluation setup (random vs the official date-based split; headers removed or not) changes the score; (2) strong classic baselines (TF-IDF + linear models) and sentence embeddings, so the transformer must *earn* its cost; (3) a DistilBERT fine-tune that fits on a CPU (shorter inputs, a subset of the training data), compared fairly against a classic model trained on the **same** subset; (4) which topics are confused and why.

---

## 10. Summary: my approach vs the reference
| | Reference | This notebook |
|---|---|---|
| Split | random 80/20 over all 18,846 posts | official date-based split (7,532 later posts as the test set) |
| Evaluation set-up | one | four set-ups: 70.3% (honest) to 93.3% (random split with headers kept) for the same model |
| Models | DistilBERT fine-tuned (full data, 512 tokens) | TF-IDF LR 70.3%, TF-IDF SVM 70.6%, MiniLM + LR 65.4%, **DistilBERT on a CPU budget 66.7%** (6,000 posts, 128 tokens) |
| Fair comparison | none | TF-IDF trained on the same 6,000 posts: 67.6%; majority class 5.3% |
| Errors | not examined | overlapping "misc"/religion/politics groups; quirk words learned as topic evidence |

**What I learned.** (1) The way a text set is split and cleaned changes the headline score (70% to 93% for one model) by more than the choice of model does; always state the set-up. (2) A fine-tuned transformer is not automatically better than TF-IDF + a linear model: with 6,000 examples, 128 tokens and 2 CPU hours it was not. (3) Always include the simple baseline trained on the *same* data. (4) Look at the largest weights of a linear model: they reveal shortcuts.

**Limits.** My DistilBERT run is a reduced budget (about 53% of the training posts, 128 tokens, 2 epochs, one seed, no tuning), so it says nothing about the full-data, 512-token transformer used by the reference. No confidence intervals were computed for the model differences. 20 Newsgroups is old, informal English from 1993; the date-based split is stricter than a random one, but the same authors and threads can still appear in both parts. **Next step:** run the full-data/512-token fine-tune on a GPU, add a paired test between the best TF-IDF and the transformer, and check with a split by author or thread.

---

## Files
* `13 Text Classification Pipeline.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `distilbert_20ng`, `meta.json`, `minilm_logreg.joblib`, `tfidf_logreg.joblib`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "13 Text Classification Pipeline.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
