# 29 - Generative AI Model from Scratch (mini GPT)
**Topic:** Deep Learning  |  **Where the project came from:** page list  |  **Reference:** [reference article](https://amanxai.com/2024/08/05/generative-ai-model-from-scratch-with-python/)

**Topic:** Deep Learning  |  **Reference:** "Generative AI Model from Scratch with Python" (amanxai.com, Aug 2024)

**Question.** What does it take to build a tiny GPT-style model (a *decoder-only transformer*) from nothing, how well does it learn, and how can we tell whether its text is good?

**Reference method:** a small generative model built and run, with a text sample shown.
**What I do:** write the transformer **from the individual pieces** (token and position embeddings, masked multi-head self-attention, feed-forward blocks, residual connections, layer norm), train it on the 1.1-million-character Tiny Shakespeare text, and then **measure** it: (1) validation loss against simple baselines (uniform, letter frequencies, and counting models that look back 1, 2 or 4 characters); (2) how much of the context the model actually uses; (3) generated text under different sampling temperatures, scored with objective measures (share of generated words that occur in the training text, repetition, diversity, and **verbatim copying** of the training text).

---

## 10. Summary: my approach vs the reference
| | Reference | This notebook |
|---|---|---|
| Model | a small generative model built and run | a decoder-only transformer written from its parts (0.82 M parameters, 4 layers, 4 heads, context 128) |
| Evaluation | a text sample shown | validation loss against counting baselines: **1.546** vs 1.757 (4-character n-gram), 4.174 (uniform) |
| Context | not examined | loss flattens after about 8 characters of context (1.577 at 8, 1.523 at 128) |
| Generated text | one sample | four temperatures, objective measures: valid words 97.9% (T=0.5) to 76.7% (T=1.3), no verbatim 40-character copying |
| Inside the model | not shown | attention distance grows with depth (about 2 characters in layer 0, up to about 12 in layer 3) |

**What I learned.** (1) A language model should be judged first by held-out loss against simple baselines; here the transformer's gain over a 4-character counter is real but moderate (about 12%). (2) Most of the useful context is the last few characters; a longer window helped little for a model this small. (3) Sample quality measures at this scale (valid words, repetition, copying) are blunt: several did not separate the settings, and at T=1.0 the counting model matched the transformer on valid words. (4) Temperature is a trade between fluency and variety, not a quality knob. (5) Reading generated text by eye is easy to over-trust: it *sounds* like Shakespeare while meaning almost nothing.

**Limits.** Characters, not words (so the vocabulary is tiny and spelling is part of the task); one training run, one seed, no tuning of size or learning rate; the validation set is the last 10% of one play collection, so style is not independent of the training part; the measures use 20 samples per setting with no intervals; the model is far too small to say anything about large models. **Next step:** repeat with several seeds, add dropout or early stopping (the train-validation gap is growing), and compare a word-piece vocabulary.

---

## Files
* `29 Mini GPT from Scratch.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `config.json`, `meta.json`, `mini_gpt.pt`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "29 Mini GPT from Scratch.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
