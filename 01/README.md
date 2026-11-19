# 01 - Car Insurance Modelling
**Topic:** Regression  |  **Where the project came from:** page list  |  **Reference:** [reference article](https://amanxai.com/2024/01/08/car-insurance-modelling-using-python/)

**Topic:** Regression  |  **Reference:** "Car Insurance Modelling using Python" (amanxai.com, Jan 2024)

**Goal.** Predict how many claims a policy will produce per year of cover, so a price can be set per risk.
**Why this is not an ordinary regression.** The target is a *count* (0, 1, 2, ...) and policies are observed for different lengths of time (`Exposure`, in years). A plain least-squares model on the count ignores both facts.

**What the reference does:** a Poisson GLM (statsmodels) on a small 1,000-row file with 3 predictors (it looks synthetic; I could not rerun it because its data file is no longer online), no exposure, no hold-out score.
**What this notebook does differently:** real data (678,013 French motor policies, `freMTPL2freq`), exposure handled properly, four models (constant, two Poisson GLMs, boosting) scored on a hold-out set with the Poisson deviance, and a check of whether the ranking is useful (lift by decile).

---

## 10. Summary and what I would do next
* The real portfolio shows the same direction as the reference for driver age (younger drivers claim more) but also a **non-linear age effect, a new-car spike, a strong bonus-malus effect and an urban effect** that the reference's small file does not show.
* **Model ranking (test deviance, 135,603 policies):** constant 0.6274, plain GLM 0.6078 (3.1% better), richer GLM 0.5976 (4.8%), boosting 0.5783 (7.8%). About a third of the plain GLM's gap to boosting came from the GLM's weak specification; the boosting advantage over the richer GLM is 0.0193 (95% interval 0.0170 to 0.0214, paired bootstrap on one split). The GLM is easier to explain to a regulator; choose by the use case.
* The reference could not be rerun on its data (the file is no longer online); I compared with the coefficients it printed. The reference has no exposure, hold-out score or over-dispersion check.
* **Limits:** one random split; capping rules (claims at 4, exposure at 1 year) are assumptions; no claim *severity* model (a price needs frequency x average claim cost); features are policy attributes only. **Age and region are regulated or sensitive rating factors in many jurisdictions**; a real pricing model needs a legal and fairness review. Next step: a Gamma severity model on `freMTPL2sev` and a combined pure-premium estimate.

---

## Files
* `01 Car Insurance Modelling.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `freq_glm.joblib`, `freq_glm_rich.joblib`, `freq_hgb.joblib`, `meta.json`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "01 Car Insurance Modelling.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
