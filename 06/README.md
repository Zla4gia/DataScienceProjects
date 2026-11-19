# 06 - Customer Churn: Comparing Many Models
**Topic:** Classification  |  **Where the project came from:** page list method (Compare Multiple ML Models is a regression article; I use its method on a classification problem)  |  **Reference:** [reference article](https://amanxai.com/2024/02/19/compare-multiple-machine-learning-models/)

**Topic:** Classification  |  **Reference method:** "Compare Multiple Machine Learning Models" (amanxai.com, Feb 2024)

**About the source.** That article trains several *regression* models on a 414-row house-price file and ranks them by one test-set score. The list has no other model-comparison project, so I keep its **method** (train a diverse set of models, compare them on metrics *and* cost, pick one) and apply it to a classification problem that is new to me: **which telecom customers will churn?**

**Why comparing models is harder than it looks:** one split gives one number, and small differences are noise. I use repeated cross-validation, show the spread, test whether differences are real (corrected paired t-test), and weigh accuracy against training time.

---

## 7. Summary
| | Reference article | This notebook |
|---|---|---|
| Models | a few regressors on 414 rows | 14 models (incl. baseline) on 7,043 customers |
| Evaluation | one test split | 5-fold x 3 repeats, mean +/- sd, PR-AUC as well as AUC |
| Comparison | rank by one score | corrected paired t-test: 7 of 13 models tied with the best |
| Cost | mentioned | fit time measured; logistic regression 0.05 s |
| Outcome | "pick the best" | pick the simplest model in the top group; translate ranking into campaign gain |

**What I learned.** (1) Differences smaller than the fold-to-fold spread are not rankings. (2) Include a trivial baseline and a logistic regression in every comparison; they are often in the top group. (3) A model comparison is only a starting point: tuning, calibration and the business value of the ranking decide what to deploy.

**Limits.** No hyper-parameter tuning; one telecom data set; the campaign economics are assumed; churn here is a single snapshot label without a time dimension. **Next step:** tune the top models in a nested CV and add survival analysis (time to churn).

---

## Files
* `06 Customer Churn Model Comparison.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `churn_best.joblib`, `churn_logreg.joblib`, `meta.json`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "06 Customer Churn Model Comparison.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
