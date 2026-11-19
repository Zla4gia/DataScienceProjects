# 28 - Predictive Keyboard Model (PyTorch LSTM)
**Topic:** Deep Learning  |  **Where the project came from:** page list  |  **Reference:** [reference article](https://amanxai.com/2025/05/13/building-a-predictive-keyboard-model-with-pytorch/)

**Topic:** Deep Learning  |  **Reference:** "Building a Predictive Keyboard Model with PyTorch" (amanxai.com, May 2025)

**Question.** Can a neural network guess the next word well enough to save a typist keystrokes, and does a deep model beat a simple counting model?

**Reference method:** a word-level LSTM in PyTorch trained on a small text, shown with a few example completions.
**What I add:** (1) a proper data set (**WikiText-2**, about 2 million words) with a held-out **test set**; (2) the standard language-model score, **perplexity**, and the keyboard scores that matter: **top-1/3/5 accuracy** of the next word; (3) two baselines: **word frequency** and a **trigram counting model**; (4) a **typing simulation** that measures the share of keystrokes saved by a bar of 3 suggestions when the user has typed 0, 1, 2, ... letters of the word; (5) an error analysis (which words are predictable, which are not).

---

## 9. Summary: my approach vs the reference
| | Reference | This notebook |
|---|---|---|
| Data | a small text | WikiText-2 (2.0 M training tokens), validation and test sets |
| Model | one word-level LSTM | LSTM (2.25 M parameters, 2 x 200 units, tied weights) vs **trigram counts** vs **word frequency** |
| Metric | example completions | perplexity 83 vs 132 vs 352; top-3 accuracy 42.0 / 37.8 / 23.3%; **keystrokes saved 46.6 / 45.6 / 35.1%** |
| Analysis | none | accuracy by word frequency, example suggestions, vocabulary coverage |

**What I learned.** (1) A counting baseline is a strong competitor on the product metric: the LSTM wins on perplexity and accuracy but only by 1 point on keystrokes saved. (2) The score depends on the vocabulary: with 8,000 words 12% of tokens cannot be suggested at all. (3) Language models learn the style of their data (Wikipedia), not "English". (4) My LSTM was still improving when training stopped.

**Limits.** Six epochs on a shared CPU (validation perplexity still falling); one run, no seeds; the keystroke simulation uses 150 windows and a three-suggestion bar, with no personalisation; perplexity is not comparable with the literature. **Next steps:** train to convergence with learning-rate decay, a Transformer language model, a larger vocabulary or sub-word units for rare words, and per-user adaptation.

---

## Files
* `28 Predictive Keyboard LSTM.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `keyboard_lstm.pt`, `meta.json`, `vocab.json`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "28 Predictive Keyboard LSTM.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
