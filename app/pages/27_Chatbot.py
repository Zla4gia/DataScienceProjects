import ast, operator, random, re, datetime, numpy as np, pandas as pd, joblib, streamlit as st
from common import header, footer, mdir, meta

header(27, "End-to-End Chatbot (intents + 'I don't know')", "AI agents",
       "Trained on CLINC150 (150 intents + out-of-scope questions): see the notebook for in-scope accuracy of TF-IDF vs sentence embeddings, three ways to handle out-of-scope questions (the confidence threshold and a trained 'oos' class trade accuracy against refusals), typo robustness and speed. The bot below has real handlers for a dozen intents.",
       "Intent recognition is not understanding: for 100+ intents the bot only says what it recognised. It handles jokes, coin, dice, date, time, arithmetic and a few chat intents; anything else gets an honest 'I cannot do that yet'.")
M = meta(27)
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def safe_eval(node):
    if isinstance(node, ast.Expression):
        return safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](safe_eval(node.left), safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](safe_eval(node.operand))
    raise ValueError("unsupported")


WORDS = {"plus": "+", "minus": "-", "times": "*", "multiplied by": "*", "divided by": "/", "over": "/", "to the power of": "**", "x": "*"}


def calc(msg):
    s = msg.lower()
    for w, o in WORDS.items():
        s = s.replace(w, o)
    expr = re.sub(r"[^0-9\.\+\-\*/\(\)\s]", " ", s).strip()
    try:
        return "That is %s." % round(safe_eval(ast.parse(re.sub(r"\s+", "", expr), mode="eval")), 6)
    except Exception:
        return "I could not read that calculation."


JOKES = ["Why did the scarecrow win an award? He was outstanding in his field.", "I told my computer I needed a break, and it said it would go to sleep.", "Why do programmers prefer dark mode? Because light attracts bugs."]
HANDLERS = {"greeting": lambda m: "Hello! How can I help you?", "goodbye": lambda m: "Goodbye! Have a nice day.", "thank_you": lambda m: "You are welcome!", "tell_joke": lambda m: random.choice(JOKES),
            "flip_coin": lambda m: "It's " + random.choice(["heads", "tails"]) + ".", "roll_dice": lambda m: "You rolled a %d." % random.randint(1, 6), "date": lambda m: "Today is " + datetime.date.today().strftime("%A, %d %B %Y") + ".",
            "time": lambda m: "It is " + datetime.datetime.now().strftime("%H:%M") + ".", "calculator": calc, "who_made_you": lambda m: "I was built as a learning project with scikit-learn and a sentence encoder.",
            "what_can_i_ask_you": lambda m: "Try: tell me a joke, flip a coin, roll a dice, what time is it, what is 12 times 7.", "are_you_a_bot": lambda m: "Yes, I am a program, not a person."}


@st.cache_resource
def load():
    from sentence_transformers import SentenceTransformer
    return joblib.load(mdir(27) / "chatbot_intents.joblib"), SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


D, enc = load()
names, th, oos, clf = D["intent_names"], D["threshold"], D["oos_index"], D["classifier"]


def reply(msg):
    p = clf.predict_proba(enc.encode([msg], normalize_embeddings=True))[0]
    k = int(clf.classes_[p.argmax()])
    conf = float(p.max())
    name = names[k]
    if conf < th or k == oos:
        return "oos", conf, "Sorry, I did not understand that. I can tell jokes, flip a coin, roll dice, give the date and time and do arithmetic."
    h = HANDLERS.get(name)
    return name, conf, (h(msg) if h else f"I understood that you want '{name.replace('_', ' ')}', but I cannot do that yet.")


if "chat" not in st.session_state:
    st.session_state.chat = [("assistant", "Hi! Ask me for a joke, a coin flip, the time, or a calculation.", "")]
for role, text, info in st.session_state.chat:
    with st.chat_message(role):
        st.write(text)
        if info:
            st.caption(info)
if msg := st.chat_input("Say something"):
    st.session_state.chat.append(("user", msg, ""))
    intent, conf, ans = reply(msg)
    st.session_state.chat.append(("assistant", ans, f"intent: {intent} (confidence {conf:.2f}; refuses below {th:.2f})"))
    st.rerun()
st.write("In-scope test accuracy (%):", M["in_scope_accuracy"])
st.write("Out-of-scope handling (test):")
st.dataframe(pd.DataFrame(M["oos"]).T.round(1), width="stretch")
st.write("Accuracy (%) when letters of the question are damaged:")
st.dataframe(pd.DataFrame(M["typos"]).T.round(1), width="stretch")
footer(27, "27 End to End Chatbot.ipynb")
