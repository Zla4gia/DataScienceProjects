# 03 - Bike Sharing Demand
**Topic:** Regression  |  **Where the project came from:** NOT on either list (well-known count-regression problem; replaces Concrete Strength, which repeats Homework-Day5)  |  **Reference:** none (not on either list)

**Topic:** Regression  |  **Source:** UCI Bike Sharing (Washington D.C., 2011-2012), loaded from OpenML
**Reference solution:** none (not on either list; it replaces Concrete Strength, which repeated Homework-Day5). I compare against a **ladder of simpler models** instead.

**Question.** How many bikes will be rented in a given hour, from the hour of day, day type and weather forecast? A bike-share operator needs this to place and rebalance bikes.

**What makes it interesting:** the target is a *count* with strong daily rhythm, a growth trend between the two years, and weather effects. And because the real use is forecasting the *future*, I split **by time** (train on the past, test on the final three months) instead of randomly.

---

## 9. Summary
| Step | Result |
|---|---|
| Honest evaluation | time-ordered split: train to Sep 2012, test Oct-Dec 2012 |
| Floor | hour x working-day average: RMSLE 0.60, MAE 80 |
| Best model | gradient boosting (Poisson loss): RMSLE 0.43, MAE 45 bikes/hour |
| Main weakness | under-predicts the growing demand (+16 bikes/hour bias), worst at commute peaks |
| Uncertainty | 80% band covers about 73% of test hours after calibration (raw: 52%) |

**What I learned.** (1) Plot the target by hour and day type before modelling: the whole problem is in those two curves. (2) A forward-in-time split exposes a growth bias a random split would hide. (3) Permutation importance can mislead when a feature barely varies in the evaluation set: confirm with an ablation. (4) Prediction intervals need an explicit coverage check on future data.

**Limits.** Two years of one city; no events/holiday detail beyond a flag; no lag features (yesterday's rentals) because a planner forecasting ahead would not have them. **Next step:** add an explicit trend term or refit weekly, and a separate casual/registered model.

---

## Files
* `03 Bike Sharing Demand.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `bike_models.joblib`, `meta.json`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "03 Bike Sharing Demand.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
