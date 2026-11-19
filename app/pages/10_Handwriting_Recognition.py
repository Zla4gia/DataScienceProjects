import numpy as np, joblib, streamlit as st, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from common import header, footer, mdir, meta

header(10, "Handwriting Recognition", "Computer Vision",
       "On MNIST a small CNN reaches 99.09% test accuracy (91 errors in 10,000) against 96.99% for k-NN on raw pixels (the reference's idea) and 92.49% for logistic regression. It is well calibrated (ECE 0.0025). But a 4-pixel shift costs the plain CNN 30 points and white-on-black vs black-on-white matters: inverted images score 30%.",
       "MNIST digits are centred and size-normalised. Draw a digit big, in the middle, white on black (or upload a clear dark-on-light photo and tick 'invert'); the app centres it like MNIST does.")
M = meta(10)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(1, 32, 3, padding=1), nn.Conv2d(32, 64, 3, padding=1)
        self.d1, self.f1, self.f2 = nn.Dropout(.25), nn.Linear(64 * 7 * 7, 128), nn.Linear(128, 10)
        self.d2 = nn.Dropout(.5)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.f2(self.d2(F.relu(self.f1(self.d1(x.flatten(1))))))


@st.cache_resource
def load():
    out = {}
    for k, f in [("CNN", "mnist_cnn_plain.pt"), ("CNN + augmentation", "mnist_cnn_aug.pt")]:
        m = CNN()
        m.load_state_dict(torch.load(mdir(10) / f, map_location="cpu"))
        out[k] = m.eval()
    return out, np.load(mdir(10) / "test_samples.npy")


models, samples = load()


def prep(img, invert=False):
    a = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    if invert:
        a = 1 - a
    if a.max() <= 0.05:
        return None
    ys, xs = np.where(a > 0.2 * a.max())
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape
    s = 20.0 / max(h, w)
    im = Image.fromarray((a * 255).astype(np.uint8)).resize((max(1, round(w * s)), max(1, round(h * s))))
    c = np.zeros((28, 28), np.float32)
    b = np.asarray(im, dtype=np.float32) / 255.0
    c[(28 - b.shape[0]) // 2:(28 - b.shape[0]) // 2 + b.shape[0], (28 - b.shape[1]) // 2:(28 - b.shape[1]) // 2 + b.shape[1]] = b
    yy, xx = np.indices(c.shape)
    cy, cx = (c * yy).sum() / c.sum(), (c * xx).sum() / c.sum()
    return np.roll(np.roll(c, int(round(14 - cy)), 0), int(round(14 - cx)), 1)


def predict(a):
    x = torch.tensor(a).view(1, 1, 28, 28)
    with torch.no_grad():
        return {k: F.softmax(m(x), 1)[0].numpy() for k, m in models.items()}


tab1, tab2, tab3 = st.tabs(["Draw", "Upload", "Test-set samples"])
arr = None
with tab1:
    try:
        from streamlit_drawable_canvas import st_canvas
        cv = st_canvas(stroke_width=18, stroke_color="#FFFFFF", background_color="#000000", height=280, width=280, drawing_mode="freedraw", key="cv")
        if cv.image_data is not None:
            arr = prep(Image.fromarray(cv.image_data.astype("uint8")).convert("L"))
    except Exception as e:
        st.info("Drawing canvas not available here (" + str(e)[:60] + "); use Upload or the samples.")
with tab2:
    up = st.file_uploader("Image of one digit", type=["png", "jpg", "jpeg"])
    inv = st.checkbox("Invert (use for dark ink on a light background)", True)
    if up:
        arr = prep(Image.open(up), invert=inv)
with tab3:
    i = st.slider("Sample number", 0, len(samples) - 1, 0)
    if st.button("Use this sample"):
        arr = samples[i, :784].reshape(28, 28)
    st.image(samples[i, :784].reshape(28, 28), width=120, caption=f"true label {int(samples[i, 784])}", clamp=True)
if arr is not None:
    st.image(arr, width=140, caption="what the model sees (28 x 28)", clamp=True)
    for k, p in predict(arr).items():
        top = np.argsort(-p)[:3]
        st.write(f"**{k}:** " + ", ".join(f"{t} ({p[t]:.0%})" for t in top))
st.write("Notebook results:", M["test_accuracy"])
footer(10, "10 Handwriting Recognition.ipynb")
