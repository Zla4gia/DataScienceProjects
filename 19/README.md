# 19 - Netflix Subscriptions Forecasting
**Topic:** Time Series  |  **Where the project came from:** page list  |  **Reference:** [reference article](https://amanxai.com/2023/08/07/netflix-subscriptions-forecasting-using-python/)

**Topic:** Time Series  |  **Reference:** "Netflix Subscriptions Forecasting using Python" (amanxai.com, Aug 2023)

**Question.** How many paying subscribers will Netflix have in the next few quarters?

**Reference method:** read the ACF/PACF plots of the differenced series, pick ARIMA(1,1,1), fit on all 42 quarters and forecast the next 10 quarters. No test of the forecast.
**What I do:** compare ARIMA with simpler baselines (last value, last value plus average growth, linear trend) and with exponential smoothing, using **rolling-origin back-testing** (fit on the past, forecast the next 1-4 quarters, move forward, repeat); look at errors **before and after the 2022 slowdown**; check whether the **prediction intervals** contain the truth as often as they claim; then forecast 8 quarters with the best method and say what the forecast can and cannot tell.

---

## 8. Summary: my approach vs the reference
| | Reference | This notebook |
|---|---|---|
| Model choice | ACF/PACF by eye -> ARIMA(1,1,1) | ARIMA(1,1,1), ARIMA with AIC order, ETS, drift variants, linear trend, naive |
| Evaluation | none | rolling-origin back-test, 81 forecasts per model, horizon 1-4, MASE, before/after 2022 |
| Result | a 10-quarter forecast | top five methods within 0.6 M (MAE about 6-6.7 M); naive 14.5 M; error of 9-10 M at 4 quarters |
| Uncertainty | none shown | 80% intervals contain the truth only 43-47% of the time |
| Forecast | ARIMA | 8 quarters, ARIMA +5.0 M/quarter (279 M by 2025Q3) vs ETS 269.5 M, labelled as a scenario |

**What I learned.** (1) A forecast is only as good as its back-test: the reference's model is no better than "last value plus recent growth". (2) Plot the ACF with its confidence band in mind: a borderline lag-1 spike cannot fix an order. (3) Check interval coverage: nominal 80% intervals covered 43%. (4) Series with regime changes (a pandemic surge, a slowdown) punish methods that extrapolate a fitted trend.

**Limits.** 42 points; a single series; no covariates (price changes, content releases, password-sharing policy); the back-test period contains one regime change; interval checks are based on a few dozen forecasts. **Next step:** add external regressors and a model with a changing trend (a local-linear-trend state-space model), and evaluate on a later data release.

---

## Files
* `19 Netflix Subscriptions Forecasting.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `meta.json`, `netflix_forecasters.joblib`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "19 Netflix Subscriptions Forecasting.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
