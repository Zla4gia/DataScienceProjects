import numpy as np, pandas as pd, streamlit as st, torch, torch.nn as nn
from common import header, footer, mdir, meta

header(25, "AI Agent that Masters a Game (CartPole)", "AI agents / reinforcement learning",
       "Four agents on the same footing: random (mean about 22), a hand-written rule, tabular Q-learning and the reference's DQN. See the learning curves, the stability and the robustness table in the notebook: results vary a lot between seeds, and agents trained on the normal game can fail when gravity, pole length or push strength change.",
       "Single game, simulated physics. The agent below is the best DQN seed; run it in modified worlds to see how brittle a learned policy is.")
M = meta(25)


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.n = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, x):
        return self.n(x)


@st.cache_resource
def load():
    net = QNet()
    net.load_state_dict(torch.load(mdir(25) / "cartpole_dqn.pt", map_location="cpu"))
    return net.eval(), np.load(mdir(25) / "cartpole_qtable.npy")


net, Q = load()
rule_k = M["rule_k"]
BINS = [np.array(b) for b in M["bins"]]
disc = lambda o: tuple(int(np.digitize(o[i], BINS[i])) for i in range(4))
policies = {"DQN": lambda o: int(net(torch.as_tensor(o, dtype=torch.float32)).argmax()),
            "Q-learning (table)": lambda o: int(np.argmax(Q[disc(o)])),
            f"hand-written rule (angle + {rule_k} x angular velocity)": lambda o: int(o[2] + rule_k * o[3] > 0),
            "random": lambda o: np.random.randint(2)}
c1, c2, c3, c4 = st.columns(4)
pol = c1.selectbox("Agent", list(policies))
grav = c2.slider("Gravity (normal 9.8)", 5.0, 20.0, 9.8)
length = c3.slider("Half pole length (normal 0.5)", 0.2, 1.2, 0.5)
force = c4.slider("Push strength (normal 10)", 4.0, 14.0, 10.0)
n = st.slider("Episodes", 5, 50, 20)
if st.button("Run"):
    import gymnasium as gym
    res = []
    for ep in range(n):
        e = gym.make("CartPole-v1")
        u = e.unwrapped
        u.gravity, u.length, u.force_mag = grav, length, force
        u.polemass_length = u.masspole * u.length
        o, _ = e.reset(seed=ep)
        tot = 0
        while True:
            o, r, te, tr, _ = e.step(policies[pol](o))
            tot += r
            if te or tr:
                break
        res.append(tot)
    res = np.array(res)
    st.metric("Mean reward (max 500)", f"{res.mean():.0f}", f"{(res >= 500).mean():.0%} of episodes reach 500", delta_color="off")
    st.bar_chart(pd.Series(res, name="reward per episode"))
st.subheader("Recorded episodes (normal physics)")
a, b = st.columns(2)
a.image(str(mdir(25) / "cartpole_random.gif"), caption="random policy")
b.image(str(mdir(25) / "cartpole_dqn.gif"), caption="trained DQN")
st.write("Notebook results (final 100-episode scores by seed):")
st.dataframe(pd.DataFrame(M["final"]), hide_index=True, width="stretch")
st.write("Robustness (mean reward in changed worlds):")
st.dataframe(pd.DataFrame(M["robustness"]).T, width="stretch")
footer(25, "25 AI Agent to Master a Game.ipynb")
