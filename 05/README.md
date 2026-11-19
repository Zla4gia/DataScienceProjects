# 05 - Credit Scoring and Risk Tiers
**Topic:** Classification  |  **Where the project came from:** page list (replaces Hybrid ML Model, which is stock forecasting, not classification)  |  **Reference:** [reference article](https://amanxai.com/2023/07/03/credit-scoring-and-segmentation-using-python/)

**Topic:** Classification  |  **Reference:** "Credit Scoring and Segmentation using Python" (amanxai.com, Jul 2023)

**Question.** Can we turn an applicant's profile into a single *credit score* that ranks people by the chance they will not repay, and then cut the score into risk tiers a lender can act on?

**Reference method:** a 1,000-row synthetic table with **no repayment outcome**; the score is a hand-weighted formula over a few columns, then k-means groups the scores. Nothing in it can tell whether the score predicts defaults.
**What I do differently:** use the real German Credit data (1,000 loans, **with** a good/bad outcome), build a classic **scorecard** (weight-of-evidence + logistic regression, points scaled like a bureau score), validate it with the industry metrics (AUC, Gini, KS), compare with a more flexible model, and check that the tiers actually separate bad rates.

---

## 8. Summary: my approach vs the reference
| | Reference | This notebook |
|---|---|---|
| Data | 1,000 synthetic rows, no outcome | German Credit: 1,000 real loans, 30% bad |
| Score | hand-weighted formula | WoE + logistic regression, scaled to points (600 = 30:1 odds, +30 points = double the odds) |
| Segments | k-means on the score | risk tiers by score quintile |
| Validation | none possible (no outcome) | repeated CV: AUC 0.79, Gini 0.57, KS 0.47; bad rate 63.5% -> 5.0% across tiers |
| Interpretability | hidden in the formula | points per group, information value per variable |

**What I learned.** (1) A score without outcome data can be formatted nicely but cannot be validated; with outcomes, tier monotonicity and KS/Gini are the checks. (2) A transparent scorecard matched gradient boosting here. (3) WoE tables must be learned inside each fold, or the validation is flattered. (4) Always read the points: counter-intuitive groups are a warning about data quirks or small samples.

**Limits.** 1,000 loans from 1994 Germany; some categories have very few cases; WoE bins are simple quantile bins; the tier cut-offs are illustrative. **Next step:** monotonic binning, reject-inference for applicants who were never approved, and removal of protected attributes with a re-validation.

---

## Files
* `05 Credit Scoring and Risk Tiers.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `credit_gbm.joblib`, `meta.json`, `scorecard.joblib`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "05 Credit Scoring and Risk Tiers.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
