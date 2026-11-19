import numpy as np, pandas as pd, joblib, streamlit as st
from sklearn.neighbors import NearestNeighbors
from common import header, footer, mdir, meta

header(7, "Geospatial Clustering: Earthquake Hotspots", "Clustering",
       "On 16,941 earthquakes of magnitude 5+ (2015-2024), k-means must assign every point and cuts plate boundaries into segments, DBSCAN finds the dense subduction zones but labels 34% of events as noise, and HDBSCAN finds the richest structure but is the least stable (subsample ARI 0.66 vs 0.94 for DBSCAN).",
       "'Not in a hotspot' means 'not dense in M5+ events in this decade', never 'safe': Los Angeles gets no hotspot only because large quakes there are rare in this period. This is not a seismic-hazard map.")
M = meta(7)
R = 6371.0


@st.cache_resource
def load():
    d = joblib.load(mdir(7) / "quake_clusters.joblib")
    return d, NearestNeighbors(n_neighbors=1).fit(d["core_vectors"]), pd.read_csv(mdir(7) / "hotspots.csv"), pd.read_csv(mdir(7) / "events_sample.csv.gz")


D, nn, hot, ev = load()
c1, c2 = st.columns(2)
lat = c1.number_input("Latitude", -90.0, 90.0, 35.7)
lon = c2.number_input("Longitude", -180.0, 180.0, 139.7)
la, lo = np.radians(lat), np.radians(lon)
v = np.array([[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])
d, j = nn.kneighbors(v)
km = 2 * R * np.arcsin(np.clip(d[0, 0] / 2, 0, 1))
label = int(D["core_labels"][j[0, 0]])
if km <= D["eps_km"]:
    st.success(f"Inside hotspot cluster {label}: {km:.0f} km from its nearest dense core point.")
    row = hot[hot.cluster == label]
    if len(row):
        st.dataframe(row, hide_index=True, width="stretch")
else:
    st.warning(f"No dense hotspot within {D['eps_km']:.0f} km (nearest dense core point {km:,.0f} km away).")
pts = ev.rename(columns={"latitude": "lat", "longitude": "lon"})
show = pts[pts.cluster >= 0].sample(min(4000, int((pts.cluster >= 0).sum())), random_state=0)
st.map(pd.concat([show[["lat", "lon"]], pd.DataFrame({"lat": [lat], "lon": [lon]})]), size=12000)
st.caption("Sample of events inside DBSCAN hotspots (100 km, 25 events) plus your point. The table lists events, share of M6+, median depth and centre of the hotspot.")
st.dataframe(pd.DataFrame(M["comparison"]).T.round(3), width="stretch")
footer(7, "07 Geospatial Clustering - Earthquake Hotspots.ipynb")
