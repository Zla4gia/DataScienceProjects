# 27 - End to End Chatbot
**Topic:** AI Agents  |  **Where the project came from:** page list  |  **Reference:** [reference article](https://amanxai.com/2023/03/27/end-to-end-chatbot-using-python/)

**Topic:** AI Agents  |  **Reference:** "End to End Chatbot using Python" (amanxai.com, Mar 2023)

**Question.** Can a chatbot recognise what a user wants (their *intent*), answer, and say "I did not understand" when the request is outside what it can do?

**Reference method:** a small hand-written intents file (a few intents with example phrases), a TF-IDF + logistic-regression classifier, a canned response per intent, and a Streamlit chat page.
**What I add:** I use the public **CLINC150** benchmark: **150 intents in 10 domains** (banking, travel, work, kitchen...) **plus "out-of-scope" questions** the bot must *not* act on. I compare TF-IDF with sentence embeddings, test three ways of handling out-of-scope inputs, look at which intents are confused, test **typos**, measure speed, and wrap the best model into a working dialogue manager with real handlers (jokes, coin flips, dice, date/time, calculator...), which I test on scripted conversations.

---

## 10. Summary: my approach vs the reference
| | Reference | This notebook |
|---|---|---|
| Data | a tiny hand-written intents file | CLINC150: 150 intents x 100 questions + out-of-scope (4,500 + 1,000 test) |
| Model | TF-IDF + logistic regression | TF-IDF 91.4%, embeddings + 5-NN 89.9%, **embeddings + logistic regression 95.3%** |
| Unknown questions | not handled | three strategies: never refuse (0% refused), threshold 0.79 (91.7% refused, in-scope accuracy 86.0%), trained oos class (60.6% refused, 96.0% precision) |
| Robustness | not tested | typos: 95.3% -> 84.7% (4% of letters) -> 52.5% (15%) |
| Chatbot | canned responses | handlers for 12 intents, honest fallbacks, restricted calculator, 25/25 scripted messages |

**What I learned.** (1) A chatbot is mostly an *out-of-scope* problem: accuracy on known intents (95%) says little about what happens on the 18% of questions it should not act on. (2) A confidence threshold and a trained reject class are two points on a trade-off; choose by the cost of a wrong action. (3) Pretrained sentence encoders beat TF-IDF on paraphrases but not on garbled text. (4) Test dialogues I wrote myself are a smoke test, not an evaluation.

**Limits.** English only; single-turn (no dialogue memory or slot filling); handlers for 12 of 150 intents; the typo noise is synthetic; the CPU timings were taken on a busy machine. **Next step:** multi-turn state (follow-up questions), spelling correction, and a calibrated reject option tuned to a stated cost.

---

## Files
* `27 End to End Chatbot.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `chatbot_intents.joblib`, `meta.json`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "27 End to End Chatbot.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
