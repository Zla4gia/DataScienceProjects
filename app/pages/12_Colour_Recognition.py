import numpy as np, pandas as pd, joblib, streamlit as st
from PIL import Image
from skimage import color as skc
from sklearn.cluster import KMeans
from common import header, footer, mdir, meta

header(12, "Colour Recognition", "Computer Vision",
       "Naming a colour by its nearest entry in a table of 949 survey names: perceptual distances (CIELAB, CIEDE2000) are only a few points better than RGB distance (leave-one-out family agreement 88-89% vs 86%, uncertainty +/-3.3). Lighting matters more than the distance: half brightness cuts family accuracy to 67%. The reference's pixel loop needs about 138 ms per pixel; a vectorised lookup is thousands of times faster.",
       "Names come from an English-language survey and the 'family' (red, green, ...) is the last word of the name, so some answers are debatable (dark reds are often called 'brown').")
M = meta(12)


@st.cache_resource
def load():
    t = pd.read_csv(mdir(12) / "xkcd_colours.csv")
    return t, skc.rgb2lab(t[["R", "G", "B"]].to_numpy(float)[None] / 255)[0], joblib.load(mdir(12) / "colour_family_knn.joblib")["family_knn"]


T, LAB, fam = load()


def name_of(rgb, metric):
    v = np.array(rgb, float)
    if metric == "RGB L1 (the reference)":
        d = np.abs(T[["R", "G", "B"]].to_numpy(float) - v).sum(1)
    elif metric == "RGB Euclidean":
        d = np.sqrt(((T[["R", "G", "B"]].to_numpy(float) - v) ** 2).sum(1))
    else:
        lab = skc.rgb2lab(v[None, None] / 255)[0, 0]
        d = skc.deltaE_ciede2000(np.repeat(lab[None], len(LAB), 0), LAB) if metric == "CIEDE2000" else np.sqrt(((LAB - lab) ** 2).sum(1))
    return T["name"][int(d.argmin())]


tab1, tab2 = st.tabs(["Pick a colour", "Palette of an image"])
with tab1:
    hexc = st.color_picker("Colour", "#3b8f6e")
    rgb = tuple(int(hexc[i:i + 2], 16) for i in (1, 3, 5))
    st.write("RGB", rgb)
    names = {m: name_of(rgb, m) for m in ["RGB L1 (the reference)", "RGB Euclidean", "CIELAB dE76", "CIEDE2000"]}
    st.dataframe(pd.DataFrame({"distance": list(names), "nearest name": list(names.values())}), hide_index=True, width="stretch")
    lab = skc.rgb2lab(np.array(rgb)[None, None] / 255.0)[0, 0]
    st.metric("Colour family (k-NN on CIELAB)", str(fam.predict(lab[None])[0]))
with tab2:
    up = st.file_uploader("Image", type=["png", "jpg", "jpeg"])
    k = st.slider("Number of colours", 3, 8, 5)
    if up:
        im = Image.open(up).convert("RGB")
        im.thumbnail((200, 200))
        a = np.asarray(im).reshape(-1, 3) / 255.0
        if len(a) > 20000:
            a = a[np.random.RandomState(0).choice(len(a), 20000, replace=False)]
        L = skc.rgb2lab(a[None])[0]
        km = KMeans(k, n_init=4, random_state=0).fit(L)
        share = np.bincount(km.labels_, minlength=k) / len(L)
        rows = []
        for c, s in sorted(zip(km.cluster_centers_, share), key=lambda t: -t[1]):
            rgb8 = np.clip(skc.lab2rgb(c[None, None])[0, 0] * 255, 0, 255).round().astype(int)
            d = skc.deltaE_ciede2000(np.repeat(c[None], len(LAB), 0), LAB)
            rows.append({"colour": "#%02x%02x%02x" % tuple(rgb8), "share %": round(s * 100, 1), "nearest name": T["name"][int(d.argmin())], "family": str(fam.predict(c[None])[0])})
        st.image(Image.open(up), width=300)
        df = pd.DataFrame(rows)
        st.dataframe(df.style.apply(lambda col: [f"background-color: {v}; color: transparent" if col.name == "colour" else "" for v in col], axis=0), hide_index=True, width="stretch")
st.write("Notebook results (leave-one-out family agreement %):", M["loo_family_%"], "| lighting stress test:", M["stress"])
footer(12, "12 Colour Recognition.ipynb")
