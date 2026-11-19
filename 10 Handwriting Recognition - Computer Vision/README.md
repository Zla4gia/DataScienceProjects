# 10 - Handwriting Recognition
**Topic:** Computer Vision  |  **Where the project came from:** author's computer-vision list (page has no CV section)  |  **Reference:** [reference article](https://amanxai.com/2020/10/10/handwriting-recognition-with-python/)

**Topic:** Computer Vision  |  **Reference:** "Handwriting Recognition with Python" (amanxai.com, Oct 2020)

**Question.** Can a model read handwritten digits, and does it still work when the writing is not as tidy as the benchmark?

**Reference method:** k-nearest-neighbours on 32x32 black-and-white images flattened into 1,024-long vectors (a small teaching data set that is not publicly downloadable).
**What I do:** use the standard **MNIST** data (60,000 + 10,000 images), reproduce the reference idea (k-NN on raw pixels), add a logistic regression and a small **CNN** (PyTorch, CPU), study the **mistakes**, check **confidence calibration**, and then test **robustness**: rotated, shifted, noisy, thicker/thinner and inverted digits, with and without training on augmented data. A model that scores 99% on MNIST but fails on slightly different writing is not a handwriting reader.

---

## 10. Summary: my approach vs the reference
| | Reference | This notebook |
|---|---|---|
| Data | 32x32 binary text images (thousands) | MNIST: 55,000 train / 5,000 validation / 10,000 test |
| Model | k-NN on flattened pixels | k-NN (same idea), logistic regression, CNN, CNN + augmentation |
| Test accuracy | not reported as one comparable number | k-NN 96.99%, logistic 92.49%, **CNN 99.09%**, CNN+aug 98.96% |
| Confidence | none | CNN calibrated (ECE 0.0025); 93% coverage at 99.95% accuracy |
| Robustness | not tested | rotation, shift, scale, noise, stroke width, inversion: big drops outside the training conditions |

**What I learned.** (1) The k-NN idea works (97%) but a CNN makes about 3 times fewer errors with one fast epoch loop. (2) A near-perfect benchmark score says little about messy input: shift by 4 pixels cost the plain CNN 30 points. (3) Augmentation fixes what it simulates and can hurt elsewhere. (4) A calibrated confidence enables a "refer to a human" rule.

**Limits.** MNIST digits are centred, size-normalised and written by US census workers and students; photos, other scripts or other pens will differ. Only 5 epochs, one seed per model (so tiny differences like 99.09 vs 98.96 are not meaningful). **Next step:** train with milder, mixed augmentations plus noise and stroke jitter, and evaluate on digits drawn by real users.

---

## Files
* `10 Handwriting Recognition.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `meta.json`, `mnist_cnn_aug.pt`, `mnist_cnn_plain.pt`, `mnist_logreg.joblib`, `test_samples.npy`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "10 Handwriting Recognition.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
