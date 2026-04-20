import streamlit as st
import pickle
import os

# ==============================
# Load model safely (important for deployment)
# ==============================
BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

# ==============================
# Keywords for basic detection
# ==============================
spam_keywords = ["win", "free", "lottery", "offer", "prize", "click"]

# ==============================
# Prediction function
# ==============================
def predict_spam(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]

    spam_score = probability[1]  # probability of spam

    detected_keywords = [word for word in spam_keywords if word in text.lower()]

    return prediction, spam_score, detected_keywords


# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Spam Detector", page_icon="📧")

st.title("📧 Smart Spam Email Detector")
st.write("Enter a message below to check whether it is Spam or Not Spam.")

msg = st.text_area("✉️ Enter your message:")

if st.button("🔍 Check"):

    if msg.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        prediction, spam_score, keywords = predict_spam(msg)

        # Result
        if prediction == 1:
            st.error("🚫 Spam Message")
        else:
            st.success("✅ Not Spam Message")

        # Probability
        st.info(f"📊 Spam Probability: {spam_score*100:.2f}%")

        # Keywords
        if keywords:
            st.warning(f"⚠️ Suspicious words detected: {', '.join(keywords)}")
