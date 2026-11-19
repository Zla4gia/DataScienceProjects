# 24 - Rainfall Trends in India Analysis
**Topic:** Data Visualization  |  **Where the project came from:** page list  |  **Reference:** [reference article](https://amanxai.com/2024/11/18/rainfall-trends-in-india-analysis-with-python/)

**Topic:** Data Visualization  |  **Reference:** "Rainfall Trends in India Analysis with Python" (amanxai.com, Nov 2024)

**Question.** Has India's rainfall changed over the 20th century, how much does it vary, and how often do dry and wet extremes occur?

**Reference method:** plot annual rainfall and a 10-year moving average, read "no apparent long-term trend" off the picture, flag years beyond +/-1.5 standard deviations as drought / extreme-rain years, and apply an Isolation Forest.
**What I add:** (1) audit the file's arithmetic; (2) **test** the trend (ordinary regression with robust errors, and the non-parametric **Mann-Kendall test with Sen's slope**) and show how much the answer **depends on the period** chosen; (3) test for **persistence** and **multi-decade swings** (the reason a picture can look like a trend); (4) show that the Isolation Forest on one column is only a threshold in disguise; (5) count extremes by period, with the uncertainty that few events bring.

---

## 9. Summary: my approach vs the reference
| | Reference | This notebook |
|---|---|---|
| Trend | "no apparent trend" from a picture | tested: -3.0 mm/decade (95% interval -12.1 to +6.1), Mann-Kendall p = 0.39; 30-year windows range -49 to +71 mm/decade |
| Seasons | bar chart of means | monsoon = 75% of rain; no season has a robust trend (one nominal decline in Jan-Feb, p = 0.03/0.08, among 5 tests) |
| Persistence | not examined | lag-2 autocorrelation 0.31; no significant mean shift between thirds (p = 0.23) |
| Extremes | 1.5 sd rule: 5 dry, 7 wet | same, plus 1 sd and percentile rules; counts match a normal distribution (no excess of extremes) |
| Isolation Forest | listed as anomaly detection | equivalent to taking the most extreme values of one column |
| Drivers of dry years | not examined | June-October deficit, 84% in the monsoon months (JJAS) |

**What I learned.** (1) "No trend visible" and "no trend" are different claims: the interval shows what a century of data can and cannot exclude. (2) Slopes over short windows are unstable for variable series with mild persistence. (3) Extreme-year counts change with the definition; compare them with what the distribution would predict. (4) A multivariate anomaly algorithm on one variable is a threshold.

**Limits.** One all-India series (regional trends could differ); area-weighted annual totals hide the distribution within the year (heavy-rain days, dry spells); no temperature, ENSO or station information; only 115 values per series. **Next step:** regional and daily data, extreme-rain indices and a link with sea-surface-temperature indices.

---

## Files
* `24 Rainfall Trends in India.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `india_rainfall_clean.csv`, `meta.json`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "24 Rainfall Trends in India.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
