# 07 - Geospatial Clustering (earthquake hotspots)
**Topic:** Clustering  |  **Where the project came from:** page list (reference uses Kaggle delivery data; I use public USGS earthquake data)  |  **Reference:** [reference article](https://amanxai.com/2025/04/22/geospatial-clustering-with-python/)

**Topic:** Clustering  |  **Reference:** "Geospatial Clustering with Python" (amanxai.com, Apr 2025)

**Question.** Where do strong earthquakes (magnitude 5+) concentrate, and which clustering method finds those zones honestly?

**Reference method:** k-means on food-delivery coordinates (Kaggle data, not downloadable here) to create delivery zones, plus a `geodesic` distance calculation.
**What I do differently:** use real, public geographic points (USGS, 17,000 M5+ earthquakes, 2015-2024) whose shape is *not* round, so the choice of algorithm matters. I compare **k-means**, **DBSCAN** and **HDBSCAN** on the sphere (not on raw lat/lon degrees), check them with a quantitative score, test stability, and turn the best one into a lookup ("which hotspot is this point in, and how far is it?").

---

## 8. Summary: my approach vs the reference
| | Reference | This notebook |
|---|---|---|
| Data | delivery coordinates (Kaggle) | 16,941 USGS earthquakes M5+, 2015-2024 |
| Geometry | lat/lon with k-means; geodesic distance for one feature | unit-sphere vectors, so distances and the date line are handled correctly |
| Algorithms | k-means | k-means, DBSCAN, HDBSCAN with settings sensitivity |
| Checks | visual map | silhouette, region purity, subsample stability (ARI), noise share |
| Output | zones | hotspot table + a point-to-hotspot lookup with distance |

**What I learned.** (1) k-means is right when you need a *partition* (delivery zones, where every address needs an owner) and wrong when the real structure is thin belts and noise. (2) DBSCAN's two parameters change the answer a lot; always show a sensitivity grid. (3) Scores like silhouette favour some algorithms by construction; use several checks. (4) "Not in a cluster" must be read as "not dense in this data", never as "safe".

**Limits.** Ten years only, magnitude 5+ only; place labels are crude ground truth; no time dimension (aftershock sequences are clustered in time as well); the 100 km / 25 setting is a judgement. **Next step:** space-time clustering (ST-DBSCAN) to separate aftershock swarms from long-lived zones.

---

## Files
* `07 Geospatial Clustering - Earthquake Hotspots.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `events_sample.csv.gz`, `hotspots.csv`, `meta.json`, `quake_clusters.joblib`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "07 Geospatial Clustering - Earthquake Hotspots.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
