import streamlit as st
import pickle
import os

st.set_page_config(
    page_title="SentimentAI",
    page_icon="🧠",
    layout="centered"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

with open(os.path.join(BASE_DIR, "Sentiment_analysis.pkl"), "rb") as f:
    model = pickle.load(f)


st.title("🧠 SentimentAI")
st.write("Analyze whether your text is positive or negative.")

text = st.text_area(
    "Enter your text",
    placeholder="Example: I absolutely loved this product!",
    height=180
)

if st.button("Analyze Sentiment", type="primary"):

    if not text.strip():
        st.warning("Please enter some text.")
    else:
        features = vectorizer.transform([text])
        prediction = model.predict(features)[0]

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            confidence = max(probabilities) * 100
        else:
            confidence = None

        sentiment = str(prediction).lower()

        if sentiment == "positive":
            st.success(f"😊 Positive")
        else:
            st.error(f"😞 Negative")

        if confidence is not None:
            st.metric("Confidence", f"{confidence:.2f}%")
            st.progress(int(confidence))
