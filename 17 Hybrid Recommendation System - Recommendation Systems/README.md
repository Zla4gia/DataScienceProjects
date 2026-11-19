# 17 - Hybrid Recommendation System
**Topic:** Recommendation Systems  |  **Where the project came from:** page list  |  **Reference:** [reference article](https://amanxai.com/2023/06/05/hybrid-recommendation-system-using-python/)

**Topic:** Recommendation Systems  |  **Reference:** "Hybrid Recommendation System using Python" (amanxai.com, Jun 2023)

**Question.** Which movies should we suggest to a user, and does mixing collaborative filtering (people who liked what you liked) with content (genres) beat each alone?

**Reference method:** a fashion-products table; content-based filtering with TF-IDF text similarity, collaborative filtering with `surprise` SVD to predict ratings, and a weighted blend; no hold-out evaluation. (The `surprise` package does not install on this machine's Python version, and the fashion file's ratings cannot be validated, so I use the standard **MovieLens 100K** data: 100,000 real ratings by 943 users on 1,682 films.)
**What I do:** a proper **time-based hold-out** (every user's last 20% of ratings is hidden), **ranking metrics** (Recall@10, NDCG@10, hit rate), five recommenders (popularity, content, item-kNN, SVD, hybrid), the **blend weights tuned on a separate validation slice**, plus coverage, novelty, diversity, **cold-start items** and **new users**, and a check that a lower rating-prediction error (RMSE) is not the same as a better top-10 list.

---

## 10. Summary: my approach vs the reference
| | Reference | This notebook |
|---|---|---|
| Data | fashion products table (ratings not verifiable) | MovieLens 100K: 100,000 real ratings, 943 users, 1,682 movies |
| Models | content (TF-IDF text) + `surprise` SVD, blended | popularity, content, item-kNN, SVD, hybrid with weights tuned on a validation slice |
| Evaluation | none | time-based hold-out, NDCG@10 / recall@10 / hit rate, paired bootstrap, coverage, novelty, diversity |
| Result | a blend | hybrid 0.189 > SVD 0.175 = item-kNN 0.174 > popularity 0.104 (NDCG@10; hybrid gain clear but small) |
| Stress tests | none | cold movies (hybrid best), new users (popularity best), rating RMSE vs ranking quality (RMSE-optimal model ranks poorly) |

**What I learned.** (1) Evaluate with a time-based split and ranking metrics, not rating error. (2) A blend of two strong models helped a little, with a clear but small interval; weights must be tuned away from the test set. (3) The best method depends on the situation: popularity wins for new users, collaborative methods for cold items. (4) Coverage, novelty and diversity move in different directions from accuracy.

**Limits.** One 20% time cut; explicit ratings turned into binary "liked" signals; genres only as content; 907 evaluated users (intervals shown are for user sampling only); no online test. **Next step:** richer content (plot text embeddings), implicit-feedback ALS, and a rule that switches to popularity for users with fewer than about 10 ratings.

---

## Files
* `17 Hybrid Recommendation System.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `hybrid_recommender.joblib`, `meta.json`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "17 Hybrid Recommendation System.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
