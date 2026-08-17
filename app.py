from flask import Flask, request, render_template_string
import os
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "Customer_churn.pkl")
model = joblib.load(MODEL_PATH)

FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

# These mappings match sklearn LabelEncoder's alphabetical ordering for the
# standard Telco Customer Churn categories used to train the supplied model.
OPTIONS = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "MultipleLines": {"No": 0, "No phone service": 1, "Yes": 2},
    "InternetService": {"DSL": 0, "Fiber optic": 1, "No": 2},
    "OnlineSecurity": {"No": 0, "No internet service": 1, "Yes": 2},
    "OnlineBackup": {"No": 0, "No internet service": 1, "Yes": 2},
    "DeviceProtection": {"No": 0, "No internet service": 1, "Yes": 2},
    "TechSupport": {"No": 0, "No internet service": 1, "Yes": 2},
    "StreamingTV": {"No": 0, "No internet service": 1, "Yes": 2},
    "StreamingMovies": {"No": 0, "No internet service": 1, "Yes": 2},
    "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
    "PaperlessBilling": {"No": 0, "Yes": 1},
    "PaymentMethod": {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3,
    },
}

DISPLAY = {
    "gender": "Gender",
    "SeniorCitizen": "Senior citizen",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "tenure": "Tenure (months)",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "Contract": "Contract",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
}

def selected_value(data, field):
    raw = data.get(field)
    if raw is None or raw == "":
        raise ValueError(f"Missing value for {DISPLAY[field]}.")
    return int(raw)

def build_input(data):
    row = {
        "gender": selected_value(data, "gender"),
        "SeniorCitizen": int(data.get("SeniorCitizen", 0)),
        "Partner": selected_value(data, "Partner"),
        "Dependents": selected_value(data, "Dependents"),
        "tenure": int(data.get("tenure", 0)),
        "PhoneService": selected_value(data, "PhoneService"),
        "MultipleLines": selected_value(data, "MultipleLines"),
        "InternetService": selected_value(data, "InternetService"),
        "OnlineSecurity": selected_value(data, "OnlineSecurity"),
        "OnlineBackup": selected_value(data, "OnlineBackup"),
        "DeviceProtection": selected_value(data, "DeviceProtection"),
        "TechSupport": selected_value(data, "TechSupport"),
        "StreamingTV": selected_value(data, "StreamingTV"),
        "StreamingMovies": selected_value(data, "StreamingMovies"),
        "Contract": selected_value(data, "Contract"),
        "PaperlessBilling": selected_value(data, "PaperlessBilling"),
        "PaymentMethod": selected_value(data, "PaymentMethod"),
        "MonthlyCharges": float(data.get("MonthlyCharges", 0)),
        "TotalCharges": float(data.get("TotalCharges", 0)),
    }

    if not 0 <= row["tenure"] <= 100:
        raise ValueError("Tenure must be between 0 and 100 months.")
    if row["MonthlyCharges"] < 0 or row["TotalCharges"] < 0:
        raise ValueError("Charges cannot be negative.")

    return pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)

HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Retention Intelligence</title>
<style>
:root{
  --bg:#05070b; --panel:rgba(12,17,25,.82); --line:rgba(110,231,255,.18);
  --cyan:#63e6ff; --cyan2:#16c7e8; --text:#eaf7ff; --muted:#8193a8;
  --green:#58e39b; --red:#ff647c; --amber:#ffd166;
}
*{box-sizing:border-box}
body{
  margin:0; min-height:100vh; color:var(--text);
  font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  background:
    radial-gradient(circle at 15% 10%,rgba(22,199,232,.12),transparent 28%),
    radial-gradient(circle at 90% 80%,rgba(99,230,255,.08),transparent 30%),
    linear-gradient(180deg,#060910,#030509);
}
body:before{
  content:""; position:fixed; inset:0; pointer-events:none; opacity:.23;
  background-image:linear-gradient(rgba(99,230,255,.055) 1px,transparent 1px),
                   linear-gradient(90deg,rgba(99,230,255,.055) 1px,transparent 1px);
  background-size:48px 48px;
  mask-image:linear-gradient(to bottom,black,transparent 90%);
}
.shell{width:min(1180px,94vw);margin:auto;padding:30px 0 60px;position:relative}
.topbar{display:flex;align-items:center;justify-content:space-between;margin-bottom:24px}
.brand{display:flex;gap:13px;align-items:center}
.mark{width:38px;height:38px;border:1px solid var(--cyan);border-radius:12px;position:relative;box-shadow:0 0 26px rgba(99,230,255,.18)}
.mark:before,.mark:after{content:"";position:absolute;background:var(--cyan);border-radius:50%}
.mark:before{width:8px;height:8px;left:14px;top:14px;box-shadow:0 0 15px var(--cyan)}
.mark:after{width:22px;height:1px;left:8px;top:18px;opacity:.55}
.brand h1{font-size:16px;letter-spacing:.18em;margin:0;text-transform:uppercase}
.brand span{display:block;color:var(--muted);font-size:11px;margin-top:3px;letter-spacing:.12em}
.status{display:flex;align-items:center;gap:8px;color:#a6b6c8;font-size:12px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 12px var(--green)}
.hero{display:grid;grid-template-columns:1.25fr .75fr;gap:18px;margin-bottom:18px}
.card{
  background:linear-gradient(145deg,rgba(16,24,34,.9),rgba(7,11,17,.82));
  border:1px solid var(--line);border-radius:22px;box-shadow:0 22px 70px rgba(0,0,0,.28);
  backdrop-filter:blur(18px);
}
.hero-copy{padding:30px}
.kicker{color:var(--cyan);font-size:11px;letter-spacing:.22em;text-transform:uppercase}
.hero h2{font-size:clamp(28px,4vw,48px);line-height:1.03;margin:10px 0 12px;letter-spacing:-.04em}
.hero p{color:var(--muted);max-width:680px;line-height:1.65;margin:0}
.orbit{min-height:190px;display:grid;place-items:center;overflow:hidden;position:relative}
.orbit-ring{width:135px;height:135px;border:1px solid rgba(99,230,255,.28);border-radius:50%;position:relative;box-shadow:0 0 50px rgba(22,199,232,.08)}
.orbit-ring:before{content:"";position:absolute;inset:19px;border:1px dashed rgba(99,230,255,.32);border-radius:50%;animation:spin 12s linear infinite}
.orbit-core{position:absolute;inset:47px;border-radius:50%;background:radial-gradient(circle,var(--cyan),rgba(99,230,255,.12) 45%,transparent 70%);box-shadow:0 0 36px rgba(99,230,255,.4)}
@keyframes spin{to{transform:rotate(360deg)}}
.form-card{padding:24px}
.section-title{display:flex;justify-content:space-between;align-items:end;margin:0 0 18px}
.section-title h3{margin:0;font-size:17px}
.section-title span{color:var(--muted);font-size:11px}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
.field label{display:block;font-size:11px;color:#9fb0c2;margin:0 0 7px 3px;letter-spacing:.03em}
input,select{
  width:100%;height:45px;border:1px solid rgba(145,177,200,.15);border-radius:12px;
  background:rgba(4,8,13,.72);color:var(--text);padding:0 13px;outline:none;
  transition:.2s;border-color .2s,box-shadow .2s;
}
input:focus,select:focus{border-color:rgba(99,230,255,.62);box-shadow:0 0 0 3px rgba(99,230,255,.08)}
select{appearance:none;background-image:linear-gradient(45deg,transparent 50%,#789 50%),linear-gradient(135deg,#789 50%,transparent 50%);background-position:calc(100% - 16px) 19px,calc(100% - 11px) 19px;background-size:5px 5px;background-repeat:no-repeat}
.toggle{display:flex;gap:8px}
.toggle label{margin:0;flex:1}
.toggle input{display:none}
.toggle span{display:flex;align-items:center;justify-content:center;height:45px;border:1px solid rgba(145,177,200,.15);border-radius:12px;color:#8fa1b4;font-size:12px;cursor:pointer;background:rgba(4,8,13,.72)}
.toggle input:checked+span{border-color:rgba(99,230,255,.62);color:var(--cyan);background:rgba(99,230,255,.07);box-shadow:inset 0 0 20px rgba(99,230,255,.04)}
.actions{display:flex;gap:12px;margin-top:20px}
button{
  flex:1;height:50px;border:0;border-radius:14px;color:#001018;font-weight:800;letter-spacing:.05em;
  cursor:pointer;background:linear-gradient(100deg,#63e6ff,#28cfe8);box-shadow:0 10px 35px rgba(22,199,232,.18);
  transition:transform .2s,box-shadow .2s;
}
button:hover{transform:translateY(-2px);box-shadow:0 15px 42px rgba(22,199,232,.27)}
.secondary{max-width:130px;background:rgba(99,230,255,.06);border:1px solid rgba(99,230,255,.18);color:#9bddec;box-shadow:none}
.result{margin-top:18px;padding:25px;display:grid;grid-template-columns:180px 1fr;gap:25px;align-items:center}
.score{height:150px;width:150px;border-radius:50%;display:grid;place-items:center;margin:auto;position:relative;background:conic-gradient(var(--cyan) {{ probability }}%,rgba(99,230,255,.08) 0)}
.score:before{content:"";position:absolute;inset:9px;border-radius:50%;background:#071019;border:1px solid rgba(99,230,255,.12)}
.score b,.score small{position:relative;z-index:1}
.score b{font-size:31px}.score small{font-size:10px;color:var(--muted);display:block;text-align:center}
.result h2{margin:0 0 7px;font-size:27px}
.result p{margin:0;color:var(--muted);line-height:1.6}
.badge{display:inline-flex;padding:6px 10px;border-radius:999px;font-size:10px;letter-spacing:.1em;text-transform:uppercase;margin-bottom:10px}
.badge.high{background:rgba(255,100,124,.1);color:var(--red);border:1px solid rgba(255,100,124,.25)}
.badge.low{background:rgba(88,227,155,.1);color:var(--green);border:1px solid rgba(88,227,155,.25)}
.error{margin-top:14px;padding:12px 14px;border:1px solid rgba(255,100,124,.25);background:rgba(255,100,124,.07);color:#ff9aac;border-radius:12px;font-size:12px}
.foot{text-align:center;color:#4f6173;font-size:10px;margin-top:20px;letter-spacing:.12em;text-transform:uppercase}
@media(max-width:850px){.hero{grid-template-columns:1fr}.grid{grid-template-columns:repeat(2,1fr)}.result{grid-template-columns:1fr}}
@media(max-width:560px){.shell{padding-top:18px}.grid{grid-template-columns:1fr}.topbar{align-items:flex-start}.status{display:none}.form-card,.hero-copy{padding:18px}}
</style>
</head>
<body>
<div class="shell">
  <div class="topbar">
    <div class="brand"><div class="mark"></div><div><h1>Retention Intelligence</h1><span>Predictive decision console</span></div></div>
    <div class="status"><i class="dot"></i> MODEL ONLINE</div>
  </div>

  <div class="hero">
    <div class="card hero-copy">
      <div class="kicker">Customer signal analysis</div>
      <h2>Estimate retention risk in seconds.</h2>
      <p>Enter the customer's current service profile and receive a probability-based churn assessment from the deployed predictive model.</p>
    </div>
    <div class="card orbit"><div class="orbit-ring"><div class="orbit-core"></div></div></div>
  </div>

  <form method="post" class="card form-card">
    <div class="section-title"><h3>Customer profile</h3><span>19 model inputs</span></div>
    <div class="grid">
      {% for field in fields %}
        <div class="field">
          <label>{{ labels[field] }}</label>
          {% if field in option_fields %}
            <select name="{{ field }}" required>
              {% for text, value in option_fields[field].items() %}
                <option value="{{ value }}" {% if form.get(field, '')|string == value|string %}selected{% endif %}>{{ text }}</option>
              {% endfor %}
            </select>
          {% elif field == "SeniorCitizen" %}
            <div class="toggle">
              <label><input type="radio" name="SeniorCitizen" value="0" {% if form.get('SeniorCitizen','0') != '1' %}checked{% endif %}><span>No</span></label>
              <label><input type="radio" name="SeniorCitizen" value="1" {% if form.get('SeniorCitizen') == '1' %}checked{% endif %}><span>Yes</span></label>
            </div>
          {% else %}
            <input name="{{ field }}" type="number" step="0.01" min="0"
                   value="{{ form.get(field, defaults[field]) }}" required>
          {% endif %}
        </div>
      {% endfor %}
    </div>
    <div class="actions">
      <button type="submit">RUN PREDICTION</button>
      <button type="reset" class="secondary" onclick="window.location.href='/'">RESET</button>
    </div>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
  </form>

  {% if result %}
  <div class="card result">
    <div class="score"><div><b>{{ result.percent }}%</b><small>CHURN PROBABILITY</small></div></div>
    <div>
      <div class="badge {{ 'high' if result.churn else 'low' }}">{{ result.risk }} RISK</div>
      <h2>{{ result.title }}</h2>
      <p>{{ result.message }}</p>
    </div>
  </div>
  {% endif %}

  <div class="foot">Secure prediction interface · No customer identifier required</div>
</div>
</body>
</html>
"""

DEFAULTS = {
    "tenure": "12",
    "MonthlyCharges": "70",
    "TotalCharges": "840",
}

@app.route("/", methods=["GET", "POST"])
def index():
    form = request.form.to_dict() if request.method == "POST" else {}
    result = None
    error = None

    if request.method == "POST":
        try:
            X = build_input(request.form)
            probability = float(model.predict_proba(X)[0][1])
            prediction = int(model.predict(X)[0])

            if prediction == 1:
                risk = "HIGH" if probability >= 0.65 else "MODERATE"
                title = "Customer likely to churn"
                message = (
                    "The model identifies elevated departure risk. "
                    "Consider proactive retention outreach and a review of service fit."
                )
                churn = True
            else:
                risk = "LOW"
                title = "Customer likely to stay"
                message = (
                    "The model identifies a lower churn likelihood for this profile. "
                    "Continue normal engagement and service monitoring."
                )
                churn = False

            result = {
                "percent": round(probability * 100, 1),
                "risk": risk,
                "title": title,
                "message": message,
                "churn": churn,
            }
        except Exception as exc:
            error = str(exc)

    return render_template_string(
        HTML,
        fields=FEATURES,
        labels=DISPLAY,
        option_fields=OPTIONS,
        defaults=DEFAULTS,
        form=form,
        result=result,
        error=error,
        probability=result["percent"] if result else 0,
    )

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
