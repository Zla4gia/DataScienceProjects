# 08 - User Profiling & Segmentation
**Topic:** Clustering  |  **Where the project came from:** page list  |  **Reference:** [reference article](https://amanxai.com/2024/02/26/user-profiling-and-segmentation-using-python/)

**Topic:** Clustering  |  **Reference:** "User Profiling and Segmentation using Python" (amanxai.com, Feb 2024)

**Question.** Can 1,000 social-media users be grouped into meaningful segments for ad targeting?

**Reference method:** pick some columns, scale/one-hot them, run k-means with a chosen k, describe each cluster.
**The trap that method hides:** k-means *always* returns k groups, whether or not the data contain groups. So before profiling anything I ask the question the reference skips: **is there cluster structure at all, or am I describing noise?** I test it three ways (dependence between columns, a clusterability statistic, and a comparison with data that has the same columns but no relationships), then segment anyway and check whether the segments differ on outcomes.

---

## 8. Summary: my approach vs the reference
| | Reference | This notebook |
|---|---|---|
| Method | k-means on chosen columns, describe clusters | first test whether structure exists, then cluster, then test the clusters |
| Dependence between columns | not checked | largest correlation 0.08; 0 of 70 tests survive correction |
| Clusterability | not checked | Hopkins 0.50; best silhouette 0.052 = shuffled-data range |
| Stability | not checked | bootstrap ARI 0.25 (unstable) |
| Segment value | "personalised campaigns" | segments' effect sizes are inside the null range; no rule predicts CTR/conversion |

**What I learned.** (1) k-means always returns k clusters: a result is not evidence of structure. (2) Shuffle each column independently to create a *null data set*, and see whether your score beats it. (3) Check stability by resampling before naming segments. (4) Test any proposed segment on an outcome it was *not* built from, with multiple-testing correction.

**Limits.** The file appears to be randomly generated (inferred from the flat distributions and zero dependence), so the negative result is about this file; real user data usually has structure. The tests used are standard but not exhaustive (for example, non-linear dependence between numeric columns was not examined beyond what k-means itself would find). **Next step:** repeat the same pipeline on real behavioural data, where I would expect the null comparison to show a clear gap.

---

## Files
* `08 User Profiling and Segmentation.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `meta.json`, `segmenter.joblib`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "08 User Profiling and Segmentation.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
