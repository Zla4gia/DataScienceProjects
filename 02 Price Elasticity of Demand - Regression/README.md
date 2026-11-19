# 02 - Price Elasticity of Demand Analysis
**Topic:** Regression  |  **Where the project came from:** page list  |  **Reference:** [reference article](https://amanxai.com/2024/08/19/price-elasticity-of-demand-analysis-with-python/)

**Topic:** Regression  |  **Reference:** "Price Elasticity of Demand Analysis with Python" (amanxai.com, Aug 2024)

**Question.** If a retailer changes the price of an item by 1%, by about how much does the quantity sold change? That ratio is the *price elasticity of demand* (PED).

**Reference method:** compute `pct_change()` of price and of quantity row by row and divide. **My method:** first check how the file is built, then estimate one elasticity with a log-log regression that compares each store-item *with itself* over time, and test whether a model that uses price predicts next week's sales better than one that ignores it.

---

## 9. Summary: my approach vs the reference
| | Reference | This notebook |
|---|---|---|
| Unit of analysis | 100,000 raw rows (about 10 copies each) | 10,000 distinct store-item-weeks |
| Estimate | mean of row-wise %dQ/%dP ratios (about -0.35 reported; -0.46 / -0.33 when I re-ran it) | log-log regression with store-item and week effects: **+0.008 (95% CI -0.035 to +0.051)** |
| Uncertainty | none | clustered standard errors, subgroup intervals |
| Validation | none | forward-in-time hold-out week, with a no-price ablation |
| Conclusion | "demand is slightly inelastic" | no detectable price effect in this file; revenue what-if is not decision-grade |

**What I learned.**
1. Check how a file is built before computing anything: the biggest result here (the duplicated rows and inconsistent money columns) came from `groupby`/`nunique`, not from a model.
2. A ratio of two noisy percentage changes is a poor estimator; a regression with a stated interval is much more honest.
3. "Not significant" is a finding: it says the data cannot support a pricing decision, not that elasticity is exactly zero.

**Limits.** Five weeks per item; competitor price is an average of inconsistent copies; no cost data; a synthetic-looking file. **Next step:** a dataset with real promotions or price experiments, where the price changes are large and not chosen by the sellers.

---

## Files
* `02 Price Elasticity of Demand.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `demand_hgb.joblib`, `elasticity_model.joblib`, `meta.json`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "02 Price Elasticity of Demand.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
