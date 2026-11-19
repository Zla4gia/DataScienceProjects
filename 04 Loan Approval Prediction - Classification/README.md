# 04 - Loan Approval Prediction
**Topic:** Classification  |  **Where the project came from:** page list  |  **Reference:** [reference article](https://amanxai.com/2023/05/15/loan-approval-prediction-using-python/)

**Topic:** Classification  |  **Reference:** "Loan Approval Prediction using Python" (amanxai.com, May 2023)

**Question.** From an applicant's profile, will the loan be approved? (Historical decisions of one housing-finance company.)

**Reference method:** fill missing values, one random 80/20 split, `StandardScaler` + random forest, report one accuracy.
**What I do differently:** data sets this small (491 labelled rows) make a single split's accuracy very noisy, so I (1) put preparation inside a pipeline so it cannot leak, (2) score with repeated cross-validation and show the spread, (3) compare against two trivial baselines, (4) check calibration and the cost of a wrong approval, and (5) look at how decisions differ between applicant groups.

**Data note.** The reference uses 614 rows; the public copy I could download has labels for **491** of them (the other 123 are an unlabelled test file). Numbers are therefore not directly comparable with the article.

---

## 8. Summary: my approach vs the reference
| | Reference | This notebook |
|---|---|---|
| Evaluation | one random 80/20 split | 5-fold CV x 10 repeats; plus 200 single-split seeds to show the instability |
| Baselines | none | approve-everyone (69.9%); credit-history rule (81.3%) |
| Result | random forest accuracy (one number) | **all models about 81%; the one-line rule matches them**; models add AUC (0.70 -> 0.77) |
| Decision | accuracy | cost-aware threshold (assumed 3:1), calibration, confusion matrix |
| Responsible use | not covered | group approval rates, model vs history |

**What I learned.** (1) Always benchmark against the simplest rule an analyst would write: here it equals the ML models in accuracy. (2) On a few hundred rows a single split can mislead by +/-7 points. (3) Picking a threshold needs a stated cost assumption, and the result is often flat. (4) Check group differences before deploying anything that decides about people.

**Limits.** 491 labelled rows (reference 614); no repayment outcomes, only past approvals; the 3:1 cost is assumed; one lender's decisions. **Next step:** repayment data to model default risk instead of approval, and a proper fairness audit with legal input.

---

## Files
* `04 Loan Approval Prediction.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `loan_logreg.joblib`, `loan_rf.joblib`, `meta.json`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "04 Loan Approval Prediction.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
