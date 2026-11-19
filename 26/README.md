# 26 - Multi-Agent Research Assistant with a Verifier
**Topic:** AI Agents  |  **Where the project came from:** page list  |  **Reference:** [reference article](https://amanxai.com/2025/09/16/building-a-multi-agent-system-using-gemini-api/)

**Topic:** AI Agents  |  **Reference:** "Building a Multi-Agent System using Gemini API" (amanxai.com, Sep 2025)

**Question.** If several specialised agents (a planner, a researcher, a writer) cooperate to produce a short report on a topic, how do we know the report is *supported by its sources*?

**Reference method:** three Gemini-powered agents in sequence: **Planner** (splits a topic into sub-topics), **Researcher** (writes about each), **Synthesizer** (merges into a report). The agents' output is judged by reading it.
**What I do:** build the same pipeline but with **grounding**: the researcher retrieves passages from real Wikipedia articles and the writer must cite them; I add a fourth agent, the **Verifier**, that checks each sentence against its cited evidence with a language-inference model. Then I **measure**: (1) can the verifier catch planted errors (changed numbers, negations, swapped names, off-topic sentences)? (2) how often does an *abstractive* writer produce sentences the evidence does not support, as judged by a second, independent model? (3) does the verifier-guarded pipeline reduce that rate, and at what cost?
**LLM back-end.** The agents run without any paid API: the planner uses fixed question templates, the researcher uses sentence-embedding retrieval, the writers are an extractive picker and **DistilBART** (a small summarisation model). A `ClaudeAgent` class in section 9 shows where an LLM would replace the planner and writer, but **it was not run** (no API key was available in this session).

---

## 10. Summary: my approach vs the reference
| | Reference (Gemini agents) | This notebook |
|---|---|---|
| Agents | Planner, Researcher, Synthesizer (all LLM) | Planner (templates), Researcher (retrieval from 807 Wikipedia passages), Writer (extractive or DistilBART), **Verifier** |
| Evidence | model's own knowledge | cited Wikipedia passages |
| Quality check | read the output | verifier tested on planted errors; independent judge; unsupported-sentence rate with bootstrap interval |
| Findings | a report | verifier catches 86-100% of planted errors **but false-alarms on 64% of genuine sentences**; judge: 0% false alarms, 83-100% detection; unsupported rate 0.7-2.5% for all three writers with overlapping intervals; the guard replaced 56.6% of sentences for no measurable gain |

**What I learned.** (1) Test a checker on **correct** inputs as well as on damaged ones: the verifier looked excellent on errors and was unusable on good text. (2) The checker needs its own checker: I validated the judge on the same planted errors before trusting it. (3) A small summarisation model stays close to its sources, so its unsupported rate is already low; the guard's cost (half of the sentences rewritten) outweighs its benefit here. (4) Faithful sentences can still lose context (pronouns, cut quotations).

**Limits.** No LLM was run (`ClaudeAgent` in section 9 is illustrative); 12 articles and fixed question templates; planted errors are crude and few for numbers (2) and names (7); the judge has its own error floor of a few percent; retrieval returns similar sentences for different questions. **Next steps:** run a real LLM writer with the same verifier and judge, tune the verifier threshold on genuine sentences (or use the DeBERTa model as verifier), and re-rank retrieval with a cross-encoder.

---

## Files
* `26 Multi-Agent Research Assistant.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `example_report.json`, `meta.json`, `passage_embeddings.npy`, `wikipedia_passages.csv.gz`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "26 Multi-Agent Research Assistant.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
