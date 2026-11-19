# 23 - Analyzing the Impact of Carbon Emissions
**Topic:** Data Visualization  |  **Where the project came from:** page list  |  **Reference:** [reference article](https://amanxai.com/2024/11/25/carbon-emissions-impact-analysis-with-python/)

**Topic:** Data Visualization  |  **Reference:** "Carbon Emissions Impact Analysis with Python" (amanxai.com, Nov 2024)

**Question.** How has atmospheric CO2 changed since 1958, how has surface temperature changed since 1961, and how closely are they linked?

**Reference method:** merge a monthly CO2 table with annual country temperatures, draw a correlation heat map (reported r = 0.96), fit trend lines, and simulate "what-if" scenarios with a regression.
**What I check first:** (1) the CO2 file contains **two different series stacked in one column**; (2) a correlation between two series that both trend upward is almost guaranteed to be high and says little by itself, so I compare correlations of the **levels, of the detrended series and of the year-to-year changes**; (3) the **speed** of change (acceleration of CO2, a break in the warming rate); (4) a **held-out test** of the "what-if" regression.

---

## 8. Summary: my approach vs the reference
| | Reference | This notebook |
|---|---|---|
| Data | merged two files | found two series stacked in the CO2 file (ppm and year-over-year %); separated them and checked the relationship (r = 1.000) |
| CO2 | trend line | trend, seasonal swing (7 ppm), acceleration (0.86 -> 2.4 ppm/yr; quadratic fit 1.7 ppm error vs 12.2 for a line on 2013-23) |
| Temperature | trend lines | World +0.28 degrees C/decade (CI 0.24-0.32), a change in slope near 1975, 99% of 185 countries warming |
| Link CO2-temperature | correlation 0.96 (levels) | 0.94 for levels, 0.47 after detrending, 0.31 for yearly changes; slope 1.7 degrees C per 100 ppm |
| What-if | regression scenarios | held-out test: CO2 alone RMSE 0.18 vs time trend 0.33; both together 0.63 (unstable) |

**What I learned.** (1) Check what is *in* a column: two series shared one. (2) A correlation between two trending series says little; compare detrended and differenced versions. (3) The acceleration of CO2 is visible and a model without it fails within a decade. (4) A what-if tool built on two collinear trending series needs a stated caveat.

**Limits.** Annual world means only; no baseline given for temperature; per-country slopes have no intervals and differing missing years; a regression on two trending series cannot establish cause (the physical mechanism is not tested here); the break year is at the edge of the search grid. **Next step:** add emissions data (not just concentration), ocean and volcanic indices, and a physically-based energy-balance model.

---

## Files
* `23 Carbon Emissions Impact Analysis.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `annual_co2_temperature.csv`, `country_warming_rates.csv`, `meta.json`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "23 Carbon Emissions Impact Analysis.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
