import streamlit as st
from main import clean_text, vectorizer, model
import time

# ------------------------------------------------
# 🎨 PAGE CONFIGURATION
# ------------------------------------------------
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered",
)

# ------------------------------------------------
# 💅 CUSTOM CSS STYLING
# ------------------------------------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        background-attachment: fixed;
        color: white;
    }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: white;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.1);
        color: white;
    }

    .main-title {
        text-align: center;
        font-size: 3em;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.4);
        animation: fadeIn 2s ease-in-out;
    }

    .subtext {
        text-align: center;
        font-size: 1.2em;
        color: #dbeafe;
        margin-top: -10px;
        margin-bottom: 30px;
        animation: fadeIn 2.5s ease-in-out;
    }

    textarea {
        border-radius: 10px !important;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
        background-color: #f9fafb;
        color: black !important;
    }

    .result-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5em;
        font-weight: bold;
        margin-top: 20px;
        animation: popIn 1s ease-in-out;
    }

    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(-10px);}
        to {opacity: 1; transform: translateY(0);}
    }

    @keyframes popIn {
        0% {transform: scale(0.8); opacity: 0;}
        100% {transform: scale(1); opacity: 1;}
    }

    .footer {
        text-align: center;
        color: #dbeafe;
        font-size: 0.9em;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# 🧠 TITLE & DESCRIPTION
# ------------------------------------------------
st.markdown("<div class='main-title'>📧 Spam Email Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Detect spam messages instantly using AI 🤖</div>", unsafe_allow_html=True)

# ------------------------------------------------
# ✍️ USER INPUT
# ------------------------------------------------
user_input = st.text_area("💬 Enter your message:", height=150, placeholder="Type something like 'Win $1000 now!' or 'Let's meet tomorrow at 5pm.'")

# ------------------------------------------------
# 🔮 PREDICTION
# ------------------------------------------------
if st.button("🔍 Predict"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a message before predicting.")
    else:
        with st.spinner("Analyzing message... 🔎"):
            time.sleep(1.5)
            msg = clean_text(user_input)
            vec = vectorizer.transform([msg])
            pred = model.predict(vec)[0]
            label = "🚫 Spam" if pred == "spam" else "✅ Not Spam"

        # 🎉 Animated output
        color = "#ef4444" if pred == "spam" else "#22c55e"
        st.markdown(
            f"<div class='result-box' style='background-color:{color}; color:white;'>{label}</div>",
            unsafe_allow_html=True
        )

# ------------------------------------------------
# 🪄 FOOTER
# ------------------------------------------------
st.markdown("<div class='footer'>Built by ❤️ Syed Ahamed Ali</div>", unsafe_allow_html=True)

