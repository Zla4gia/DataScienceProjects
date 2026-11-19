# 11 - Sign Language Classification
**Topic:** Computer Vision  |  **Where the project came from:** author's computer-vision list  |  **Reference:** [reference article](https://amanxai.com/2020/12/05/sign-language-classification-with-python/)

**Topic:** Computer Vision  |  **Reference:** "Sign Language Classification with Python" (amanxai.com, Dec 2020)

**Question.** Which of the 26 letters of the American Sign Language alphabet is a person signing in a photo?

**Reference method:** Kaggle "Sign Language MNIST" (28x28 grey-scale pixels, a separate test file), a small Keras network and image augmentation, one accuracy report. (That Kaggle file needs an account; it is not on the public Hugging Face hub in a loadable form, so I use a different public data set below.)
**My data:** 10,873 real **colour photos** of hands (26 classes, about 420 each) with very different backgrounds and sizes, which is a harder, more realistic problem than 28x28 grey pixels.
**What I focus on:** (1) a pretrained network (ResNet-18) used as a **feature extractor** plus a simple classifier, against a CNN trained from scratch; (2) **data leakage**: exactly ~420 images per class hints that the set contains augmented copies of the same original photos, which would put near-identical pictures in both training and test and inflate the score. I test for it and measure how much; (3) which letters get confused; (4) robustness to flips, rotation, brightness and blur.

---

## 10. Summary: my approach vs the reference
| | Reference | This notebook |
|---|---|---|
| Data | Sign-MNIST (28 x 28 grey, separate signers in the test file) | 10,873 real colour photos, 26 letters, varied backgrounds |
| Model | small Keras network + augmentation | ResNet-18 features + logistic regression / k-NN, and a small CNN from scratch |
| Score | one accuracy | random split 75.3% vs **group split 58.3%** (range 45-75% by grouping rule) |
| Leakage | not examined | near-copies found: 34% of images have a neighbour with similarity >= 0.85; the gap is about 17 points |
| Errors | classification report | per-letter F1, confusions (fist letters, G/H, P/Q) |
| Robustness | not examined | mirror harmless; rotation, blur, lighting and cropping drop accuracy to 20-45% |

**What I learned.** (1) A classifier evaluated with a random split on photo data can report a score 17 points too high when near-identical shots exist. (2) Always look at the nearest neighbours of your images before you trust a number. (3) Pretrained features and a simple classifier beat a small CNN from scratch on a few thousand varied photos. (4) Check the embedding space before using it: raw positive features make every pair look similar.

**Limits.** The grouping rule is my own and the true "same photo" boundary is unknown (hence the range); one fold for the model comparison; the CNN was still improving; no hand detection or cropping; J and Z are static here. **Next step:** hand cropping with a detector (MediaPipe) before classification and a person-level split if signer IDs were available.

---

## Files
* `11 Sign Language Classification.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `asl_logreg_resnet18.joblib`, `asl_small_cnn.pt`, `meta.json`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "11 Sign Language Classification.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
