# 15 - Document Analysis
**Topic:** NLP  |  **Where the project came from:** page list  |  **Reference:** [reference article](https://amanxai.com/2024/10/21/document-analysis-using-llms-with-python/)

**Topic:** NLP  |  **Reference:** "Document Analysis using LLMs with Python" (amanxai.com, Oct 2024)

**Question.** Given a long document (here the **GitHub Terms of Service**, about 8,000 words), can a program answer specific questions about it, point to where the answer is, and say "I can't find that"? And what does a quick summary miss?

**Reference method:** extract the text of a long PDF (Google's Terms of Service), summarise it with `t5-small` (which reads only the first 512 tokens), and look at the output. No questions, no scores.
**What I do:** (1) split the document into sections and passages; (2) measure how much of the document a "first 512 tokens" summary can possibly cover; (3) build **retrieval + extractive question answering** (a search step picks passages, a SQuAD-trained DistilBERT reader extracts the answer) and **score it** on 14 questions whose gold answers I verified word-for-word in the text, plus 4 questions the document does not answer; (4) analyse the document itself: readability by section, and who the obligations fall on ("you must..." vs "we may...").

---

## 10. Summary: my approach vs the reference
| | Reference | This notebook |
|---|---|---|
| Document | Google Terms of Service (PDF) | GitHub Terms of Service (public markdown, about 7,500 words) |
| Summary | t5-small on the first 512 tokens | same, plus per-section summaries: coverage of 14 key facts **0 vs 8** |
| Questions | not covered | retrieval + extractive QA scored on 14 verified questions + 4 unanswerable |
| Retrieval | n/a | MiniLM finds the right passage in the top 3 for 14 of 14 questions |
| Answers | n/a | best selection strategy: right passage 93%, F1 0.60, about 12 of 14 substantively correct (my reading) |
| Failure mode found | not tested | reader returns an answer to all unanswerable questions; confidence cannot reject them |

**What I learned.** (1) A "summary" built from the first 512 tokens is a summary of the preamble: always compute how much of the document the model sees. (2) Retrieval quality was good; the weak point was how to choose between candidate answers, and a simple combination of retrieval and reader scores helped. (3) Evaluate with questions whose answers you have verified in the source: the code asserts that each gold answer is present word-for-word. (4) Extractive readers do not abstain; design for that.

**Limits.** One document; 14 + 4 questions written by me (so the gold set is small and shares my view of the text); exact-match and F1 are rough proxies; the "answers are right" judgement is mine alone; no legal advice is implied. **Next step:** a reader that supports "no answer", a larger question set from several people, and a comparison with an LLM that must quote its source passage.

---

## Files
* `15 Document Analysis.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `gh_tos_index.joblib`, `gh_tos_passage_embeddings.npy`, `meta.json`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "15 Document Analysis.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
