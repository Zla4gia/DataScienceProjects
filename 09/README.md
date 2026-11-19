# 09 - RFM Analysis
**Topic:** Clustering  |  **Where the project came from:** page list  |  **Reference:** [reference article](https://amanxai.com/2025/08/26/a-guide-to-rfm-analysis-using-python/)

**Topic:** Clustering / customer segmentation  |  **Reference:** "A Guide to RFM Analysis using Python" (amanxai.com, Aug 2025)

**Question.** Which customers are the most valuable and which are drifting away, judged from when, how often and how much they buy?

**Reference method:** compute R, F, M per customer, score each 1-4 by quartiles, concatenate the scores into segments.
**What I check that it does not:** (1) whether the data support quartile scoring at all (here 95% of customers bought once, so "frequency quartiles" break), (2) a rank-based scoring that handles ties, (3) whether k-means finds anything the rules do not (with a shuffled-data null), and (4) the most important test: **do the RFM scores predict what customers do next?**

---

## 9. Summary: my approach vs the reference
| | Reference | This notebook |
|---|---|---|
| Scoring | quartile cut-offs on R, F, M | rank-based quintiles for R and M, business bins for F (cut-offs inspected first) |
| Data check | not mentioned | 94.6% of customers ordered once; F quartiles all equal 1 -> reference scoring gives 94.6% score 1 and nobody 2-3 |
| Segments | RFM score codes | 6 named rule segments with customers, revenue share and action |
| Discovery | none | k-means vs shuffled null: not better than noise |
| Validation | none | back-test: 5.5% return; no reliable predictive value (28 events) |

**What I learned.** (1) Print the percentile cut-offs before using quantile scores: discrete variables break them. (2) RFM segments are rules, useful as a communication tool, but their value must be shown by a forward-in-time test. (3) Revenue concentration should be measured, not assumed; here it is far below 80/20. (4) A nice-looking silhouette needs a null comparison.

**Limits.** 61 days is too short to measure recency or loyalty properly; 28 repeat events give little statistical power; the file looks generated; monetary value is revenue, not margin. **Next step:** a year or more of real transactions, a customer-lifetime-value model (BG/NBD + Gamma-Gamma), and the same back-test.

---

## Files
* `09 RFM Analysis.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `meta.json`, `rfm_kmeans.joblib`, `rfm_table.csv`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "09 RFM Analysis.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
