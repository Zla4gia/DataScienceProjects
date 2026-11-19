import numpy as np, joblib, streamlit as st
from PIL import Image
from common import header, footer, mdir, meta

header(11, "Sign Language (ASL alphabet) Classification", "Computer Vision",
       "On 10,873 real photos, a logistic regression on ResNet-18 features scores 75% with a standard random split but only 58% when near-copies of the same shot are kept together (the true figure for a new signer is probably 45-58%); k-NN reached 65% on one fold. Rotation, blur and lighting changes cut accuracy to 20-45%.",
       "Expect many mistakes: fist-like letters (A, E, M, N, S, T) and G/H, P/Q are confused, J and Z are motion letters shown here as still frames, and the model sees the whole photo (no hand detection).")
M = meta(11)


@st.cache_resource
def load():
    import torch, torch.nn as nn
    from torchvision import models, transforms
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    net.eval()
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return net, tf, joblib.load(mdir(11) / "asl_logreg_resnet18.joblib")


up = st.file_uploader("Photo of a hand signing one letter", type=["png", "jpg", "jpeg"])
st.write("Notebook results:", {"evaluation": M["leakage"], "models (group split, fold 0)": M["models_group_split"]})
if up:
    import torch
    net, tf, S = load()
    img = Image.open(up).convert("RGB")
    st.image(img, width=240)
    with torch.no_grad():
        f = net(tf(img).unsqueeze(0)).numpy()
    p = S["clf"].predict_proba(f)[0]
    top = np.argsort(-p)[:5]
    st.subheader("Predicted letters")
    st.bar_chart({S["letters"][i]: float(p[i]) for i in top})
    st.caption("Top-5 probabilities. A low top probability means the model is unsure; treat it as a guess.")
st.write("Robustness test (accuracy % on held-out photos after each change):", M["robustness"])
footer(11, "11 Sign Language Classification.ipynb")
