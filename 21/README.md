# 21 - Stock Market Anomaly Detection
**Topic:** Time Series  |  **Where the project came from:** page list  |  **Reference:** [reference article](https://amanxai.com/2024/04/08/stock-market-anomaly-detection-using-python/)

**Topic:** Time Series  |  **Reference:** "Stock Market Anomaly Detection using Python" (amanxai.com, Apr 2024)

**Question.** How do we flag unusual days in stock prices and trading volume, and how do we know a detector works when nobody has labelled the anomalies?

**Reference method:** download one year of prices for five stocks, compute the **z-score of the adjusted close price and of the volume over the whole period** and call every point with |z| > 2 an anomaly.
**What I do:** (1) show what that rule actually flags; (2) build detectors on the right quantity (**daily returns**, scaled by recent volatility) and compare them with a rolling z-score, a robust (median/MAD) version, an EWMA-volatility z-score and an Isolation Forest; (3) **plant synthetic jumps and volume bursts of known size** in the real series and measure which detector finds them at a fixed false-alarm budget (the only honest "ground truth" available); (4) look at what the best detector flags in 7 years of real data: market-wide days versus single-stock days.

---

## 10. Summary: my approach vs the reference
| | Reference | This notebook |
|---|---|---|
| Quantity analysed | z-score of the *price level* and volume over the whole year | daily *returns*, scaled by recent volatility; volume vs its recent history |
| What the rule flags | 0-5% of days, long stretches that follow trends | isolated surprises: 33 stock-days of 8,495 |
| Validation | none | planted jumps (3-8x volatility) and volume bursts; recall@60: EWMA 55% for 3x jumps vs global z 46%, Isolation Forest 36% |
| Threshold | |z| > 2 from a bell curve | empirical: |z| > 3 happens 4-6x more often than a bell curve says; alarm budget 0.39% |
| Real-data check | none | 76% of flags fall on earnings days (3% of days); March 2020 is not flagged by the adaptive detector, but is by the global rule |

**What I learned.** (1) Choose the quantity before the rule: z-scores of price levels measure trends, not anomalies. (2) With no labels, plant anomalies of known size to compare detectors, and state the recall at a fixed number of alarms. (3) A simple volatility-adjusted score beat the Isolation Forest here. (4) "Anomaly" depends on the reference regime: against the long-run average, March 2020 is the anomaly; against the previous weeks it is not.

**Limits.** Five large US stocks, 7 years; planted anomalies are one kind (a persistent jump; a volume multiplier); earnings dates unverified; a threshold of 4.5 is a choice; no intraday data. **Next step:** a GARCH model for the volatility, a test on a stock with fewer news events, and a labelled set of known events.

---

## Files
* `21 Stock Market Anomaly Detection.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `anomaly_detectors.joblib`, `flagged_days.csv`, `meta.json`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "21 Stock Market Anomaly Detection.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
