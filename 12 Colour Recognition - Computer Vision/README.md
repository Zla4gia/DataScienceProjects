# 12 - Flower Recognition
**Topic:** Computer Vision  |  **Where the project came from:** author's computer-vision list  |  **Reference:** [reference article](https://amanxai.com/2020/11/24/flower-recognition-with-python/)

**Topic:** Computer Vision  |  **Reference:** "Colour Recognition with Python" (amanxai.com, Sep 2020)

**Question.** Given a pixel (or the dominant colours of a picture), what would a person call it, and how should "closest colour" be measured?

**Reference method:** a CSV of ~1,000 colour names with RGB values; for a clicked pixel, loop over every row and return the name with the smallest **sum of absolute RGB differences** (an OpenCV double-click program).
**Why this deserves a closer look:** RGB distance is not how people see colour. I test four distance measures (RGB L1 as in the reference, RGB Euclidean, CIELAB, CIEDE2000) against two independent checks that do not depend on how I measure distance, then build an image pipeline (dominant colours by k-means in CIELAB) and stress-test it with noise, compression and lighting changes.

---

## 11. Summary: my approach vs the reference
| | Reference | This notebook |
|---|---|---|
| Naming | nearest name by sum of absolute RGB differences, Python loop | four distance measures, vectorised / k-d tree |
| Data | CSV of about 1,000 colour names (origin undocumented) | xkcd colour survey (949 names, shipped with matplotlib) |
| Validation | by eye on one picture | leave-one-out family agreement (86-89%), independent CSS test (71-76%), classifier (86-88% CV) |
| Images | clicked pixels | dominant colours by k-means in CIELAB, with name and family |
| Robustness | not tested | noise and compression harmless; lighting changes cut family accuracy to 67-83% |
| Speed | about 138 ms per pixel | 0.09 s for 40,000 pixels |

**What I learned.** (1) A perceptual colour space is the right default but, measured honestly on 570 and 78 colours, the gain over RGB is within the uncertainty (2-4 points). (2) The hard part of colour naming is lighting, not distance. (3) A "ground truth" built from name text (last word) is itself noisy; imbalance (teal: 12 names) decides some of the errors. (4) Vectorisation matters more than the choice of metric for speed.

**Limits.** English names from one survey; no skin-tone or textile-specific calibration; no white-balance step; flat-patch stress tests only. **Next step:** add a grey-world white-balance step and re-test the brightness conditions.

---

## Files
* `12 Colour Recognition.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `colour_family_knn.joblib`, `meta.json`, `xkcd_colours.csv`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "12 Colour Recognition.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
