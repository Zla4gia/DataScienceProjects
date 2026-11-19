# 20 - Multivariate Time Series Forecasting
**Topic:** Time Series  |  **Where the project came from:** page list  |  **Reference:** [reference article](https://amanxai.com/2024/03/11/multivariate-time-series-forecasting-using-python/)

**Topic:** Time Series  |  **Reference:** "Multivariate Time Series Forecasting using Python" (amanxai.com, Mar 2024)

**Question.** If we forecast one series, does using the *other* series measured alongside it make the forecast better?

**Reference method:** four stock prices over about 3 months, differenced to make them stationary, a **Vector AutoRegression (VAR)** fitted and used to forecast; no comparison with anything simpler.
**What I do:** two experiments. **(A) The reference's own setting** (stocks) tested against a "no change" forecast, on its file and on 7 years of daily prices. **(B) A data set where multivariate information should matter:** hourly power-transformer data (ETTh1: 6 electrical load measurements and the oil temperature, 17,420 hours). I forecast the oil temperature 1, 6, 12 and 24 hours ahead with a seasonal-naive baseline, a VAR, and ridge regression / gradient boosting each fitted twice: **using only the oil temperature's own past (univariate)** and **using all 7 series (multivariate)**. The difference between the two fits is the value of the other series.

---

## 8. Summary: my approach vs the reference
| | Reference | This notebook |
|---|---|---|
| Data | 62 days of 4 stock prices | the same file, **plus** 7 years of stock returns **and** 17,420 hours of 7 power-transformer series (ETTh1) |
| Model | VAR on differenced prices | VAR, ridge and gradient boosting, univariate vs multivariate, level vs change target |
| Baseline | none | no-change, last value, same hour yesterday |
| Stocks | forecast shown | VAR order by AIC = 0; no better than "no change" (RMSE 2.97 vs 3.02 for AAPL); next-day returns R2 -2% to -6% vs zero, sign right 48-50% of days |
| Transformer | n/a | other series help only with the right setup: LightGBM predicting change, +6h and +12h (6-9% lower MAE); linear models and level targets get worse |
| Best result | n/a | LightGBM, 7 series, change target: mean MAE 1.10 vs 1.20 for "last value" (about 8% better) |

**What I learned.** (1) "Multivariate" is not automatically better; test the univariate version of the *same* learner. (2) Where the data have no predictable structure (daily stock returns) a VAR learns nothing (order 0). (3) For series that drift, predict the change, not the level, or tree models fail. (4) Always compare with last-value and seasonal-naive forecasts.

**Limits.** One test period; one seed; no hyper-parameter search for the linear models; ETTh1's training year is not like its test months; no intervals for the small differences. **Next step:** several rolling test periods, a tuned and regularised linear model, and a neural sequence model for completeness.

---

## Files
* `20 Multivariate Time Series Forecasting.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `ett_forecasters.joblib`, `etth1_test_slice.csv.gz`, `meta.json`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "20 Multivariate Time Series Forecasting.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
