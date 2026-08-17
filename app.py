from flask import Flask, request, jsonify, render_template_string
import pickle
import os
import re

app = Flask(__name__)

# Load the trained artifacts once when the server starts.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

with open(os.path.join(BASE_DIR, "Sentiment_analysis.pkl"), "rb") as f:
    model = pickle.load(f)


HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SentimentAI | Sentiment Analysis</title>
    <meta name="description" content="AI-powered sentiment analysis using a trained NLP model.">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            min-height: 100vh;
            color: #eef2ff;
            background:
                radial-gradient(circle at 15% 15%, rgba(99,102,241,.28), transparent 30%),
                radial-gradient(circle at 85% 85%, rgba(14,165,233,.20), transparent 30%),
                #070b17;
            overflow-x: hidden;
        }
        .orb {
            position: fixed; width: 260px; height: 260px; border-radius: 50%;
            filter: blur(90px); opacity: .22; pointer-events: none;
        }
        .orb.one { background: #6366f1; top: -100px; left: -80px; }
        .orb.two { background: #06b6d4; right: -100px; bottom: -100px; }
        .container { width: min(960px, 92%); margin: 0 auto; padding: 42px 0 60px; position: relative; }
        .brand { display: flex; align-items: center; gap: 12px; margin-bottom: 48px; }
        .logo {
            width: 42px; height: 42px; border-radius: 13px; display: grid; place-items: center;
            background: linear-gradient(135deg, #818cf8, #22d3ee); color: #07101c;
            font-weight: 900; box-shadow: 0 10px 35px rgba(34,211,238,.18);
        }
        .brand h2 { font-size: 20px; letter-spacing: -.4px; }
        .brand span { color: #8b9ab7; font-size: 13px; display: block; margin-top: 2px; }
        .hero { text-align: center; margin-bottom: 34px; }
        .badge {
            display: inline-flex; padding: 7px 12px; border: 1px solid rgba(129,140,248,.25);
            border-radius: 999px; background: rgba(99,102,241,.09); color: #a5b4fc;
            font-size: 12px; font-weight: 700; margin-bottom: 17px;
        }
        h1 { font-size: clamp(38px, 7vw, 68px); line-height: .98; letter-spacing: -3px; }
        .gradient { background: linear-gradient(90deg, #a5b4fc, #67e8f9); -webkit-background-clip: text; color: transparent; }
        .hero p { max-width: 650px; margin: 18px auto 0; color: #9aa8c2; line-height: 1.7; font-size: 16px; }
        .card {
            border: 1px solid rgba(148,163,184,.15); background: rgba(15,23,42,.76);
            backdrop-filter: blur(18px); border-radius: 24px; padding: 25px;
            box-shadow: 0 25px 80px rgba(0,0,0,.28);
        }
        textarea {
            width: 100%; min-height: 190px; resize: vertical; border: 1px solid #26334d;
            border-radius: 17px; background: #0a1020; color: #f8fafc; padding: 18px;
            outline: none; font: inherit; line-height: 1.6; transition: .2s;
        }
        textarea:focus { border-color: #6366f1; box-shadow: 0 0 0 4px rgba(99,102,241,.11); }
        textarea::placeholder { color: #56657f; }
        .toolbar { display: flex; align-items: center; justify-content: space-between; gap: 14px; margin-top: 14px; }
        #counter { color: #61708b; font-size: 12px; }
        button {
            border: 0; cursor: pointer; border-radius: 13px; padding: 13px 21px;
            color: #07101c; font-weight: 800; font-size: 14px;
            background: linear-gradient(135deg, #a5b4fc, #67e8f9);
            box-shadow: 0 10px 30px rgba(103,232,249,.12); transition: transform .18s, box-shadow .18s;
        }
        button:hover { transform: translateY(-2px); box-shadow: 0 14px 35px rgba(103,232,249,.2); }
        button:disabled { opacity: .6; cursor: wait; transform: none; }
        .examples { margin-top: 18px; display: flex; flex-wrap: wrap; gap: 8px; }
        .example {
            background: rgba(30,41,59,.55); border: 1px solid #26334d; color: #9eacc4;
            padding: 8px 11px; border-radius: 10px; cursor: pointer; font-size: 12px;
        }
        .result { margin-top: 20px; display: none; }
        .result.show { display: block; animation: rise .35s ease; }
        @keyframes rise { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        .result-top { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
        .sentiment { font-size: 27px; font-weight: 850; }
        .sentiment.positive { color: #34d399; }
        .sentiment.negative { color: #fb7185; }
        .confidence { color: #9aa8c2; font-size: 13px; }
        .bar { height: 9px; background: #172136; border-radius: 99px; overflow: hidden; margin: 18px 0 10px; }
        .fill { height: 100%; width: 0%; border-radius: inherit; transition: width .6s ease; background: linear-gradient(90deg, #818cf8, #22d3ee); }
        .meter-labels { display: flex; justify-content: space-between; color: #64748b; font-size: 11px; }
        .error { color: #fda4af; background: rgba(244,63,94,.08); border: 1px solid rgba(244,63,94,.2); padding: 12px; border-radius: 12px; }
        .footer { text-align: center; margin-top: 28px; color: #53617a; font-size: 12px; }
        @media (max-width: 600px) {
            .container { padding-top: 26px; }
            .brand { margin-bottom: 35px; }
            .card { padding: 17px; border-radius: 19px; }
            .toolbar { align-items: flex-end; flex-direction: column; }
            button { width: 100%; }
        }
    </style>
</head>
<body>
    <div class="orb one"></div><div class="orb two"></div>
    <main class="container">
        <div class="brand">
            <div class="logo">S</div>
            <div><h2>SentimentAI</h2><span>NLP sentiment classifier</span></div>
        </div>

        <section class="hero">
            <div class="badge">TF-IDF + Multinomial Naive Bayes</div>
            <h1>Understand the <span class="gradient">feeling</span><br>behind the text.</h1>
            <p>Enter a review, comment, or message and let the trained machine-learning model classify it as positive or negative.</p>
        </section>

        <section class="card">
            <textarea id="text" maxlength="5000" placeholder="Example: I absolutely loved the product. The quality is excellent and delivery was fast!"></textarea>
            <div class="toolbar">
                <span id="counter">0 / 5000 characters</span>
                <button id="analyzeBtn" onclick="analyze()">Analyze Sentiment →</button>
            </div>
            <div class="examples">
                <span class="example" onclick="setExample('The product is amazing and I am very happy with my purchase!')">Positive example</span>
                <span class="example" onclick="setExample('Terrible experience. The product stopped working and support was useless.')">Negative example</span>
                <span class="example" onclick="setExample('The delivery arrived today. It works as expected.')">Try neutral-ish</span>
            </div>

            <div id="result" class="result">
                <div class="result-top">
                    <div id="sentiment" class="sentiment"></div>
                    <div id="confidence" class="confidence"></div>
                </div>
                <div class="bar"><div id="fill" class="fill"></div></div>
                <div class="meter-labels"><span>Lower confidence</span><span>Higher confidence</span></div>
            </div>
        </section>

        <div class="footer">Built with Flask • scikit-learn • TF-IDF • Multinomial Naive Bayes</div>
    </main>

<script>
const textBox = document.getElementById("text");
const counter = document.getElementById("counter");
textBox.addEventListener("input", () => counter.textContent = `${textBox.value.length} / 5000 characters`);

function setExample(text) {
    textBox.value = text;
    textBox.dispatchEvent(new Event("input"));
    textBox.focus();
}

async function analyze() {
    const text = textBox.value.trim();
    const btn = document.getElementById("analyzeBtn");
    const result = document.getElementById("result");

    if (!text) {
        showError("Please enter some text first.");
        return;
    }

    btn.disabled = true;
    btn.textContent = "Analyzing...";
    result.classList.remove("show");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({text})
        });

        const data = await response.json();

        if (!response.ok) throw new Error(data.error || "Prediction failed.");

        const sentiment = document.getElementById("sentiment");
        sentiment.textContent = data.emoji + " " + data.sentiment;
        sentiment.className = "sentiment " + data.sentiment.toLowerCase();
        document.getElementById("confidence").textContent = `Confidence: ${data.confidence}%`;
        document.getElementById("fill").style.width = data.confidence + "%";
        result.classList.add("show");
    } catch (error) {
        showError(error.message);
    } finally {
        btn.disabled = false;
        btn.textContent = "Analyze Sentiment →";
    }
}

function showError(message) {
    const result = document.getElementById("result");
    result.innerHTML = `<div class="error">${escapeHtml(message)}</div>`;
    result.classList.add("show");
}

function escapeHtml(value) {
    return value.replace(/[&<>"']/g, c => ({
        "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#039;"
    }[c]));
}

textBox.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") analyze();
});
</script>
</body>
</html>
"""

def predict_sentiment(text):
    # IMPORTANT: use the same vectorizer that was fitted during training.
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities) * 100)

    sentiment = str(prediction).strip().lower()
    if sentiment not in {"positive", "negative"}:
        sentiment = "positive" if sentiment in {"1", "true"} else "negative"

    return sentiment, confidence


@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")

        if not isinstance(text, str):
            return jsonify({"error": "Text must be a string."}), 400

        text = text.strip()

        if not text:
            return jsonify({"error": "Please enter some text."}), 400

        if len(text) > 5000:
            return jsonify({"error": "Text is too long. Maximum 5000 characters."}), 400

        sentiment, confidence = predict_sentiment(text)

        return jsonify({
            "sentiment": sentiment.capitalize(),
            "confidence": round(confidence, 2) if confidence is not None else None,
            "emoji": "😊" if sentiment == "positive" else "😞"
        })

    except Exception:
        app.logger.exception("Prediction error")
        return jsonify({"error": "The model could not process this text."}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": True})


if __name__ == "__main__":
    # Render provides PORT as an environment variable.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
