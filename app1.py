import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

spam_keywords = ["win", "free", "lottery", "offer", "prize", "click"]

def predict_spam(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]

    spam_score = probability[1]  # spam probability

    detected_keywords = [word for word in spam_keywords if word in text.lower()]

    return prediction, spam_score, detected_keywords


# UI
st.title("📧 Smart Spam Email Detector")

msg = st.text_area("Enter your message:")

if st.button("Check"):

    if msg.strip() == "":
        st.warning("Please enter a message")
    else:
        prediction, spam_score, keywords = predict_spam(msg)

        # 🔥 MAIN RESULT
        if prediction == 1:
            st.error("🚫 Spam Message")
        else:
            st.success("✅ Not Spam Message")

        # ⭐ CLEAR PERCENTAGE DISPLAY
        st.info(f"📊 Spam Probability: {spam_score*100:.2f}%")

        # ⚠️ Keywords
        if keywords:
            st.warning(f"Suspicious words: {', '.join(keywords)}")
