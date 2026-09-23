"""Phase 2 UI extension. Run locally with: python app.py"""
from pathlib import Path
import csv
import math
import numpy as np
from flask import Flask, render_template, request
from saved_model import SavedLinearSVC

BASE = Path(__file__).resolve().parent
FEATURES = ("Glucose", "Insulin", "BMI", "Age")
LABELS = ("2-hour glucose", "2-hour serum insulin", "BMI", "Age")
UNITS = ("mg/dL", "microU/mL", "kg/m2", "years")
with (BASE / "diabetes.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))
DATA = np.array([[float(row[name]) for name in FEATURES] for row in rows])
LOW, HIGH = DATA.min(axis=0), DATA.max(axis=0)
MEDIANS = np.array([np.median(DATA[DATA[:, i] > 0, i]) for i in range(4)])
model = SavedLinearSVC(BASE / "model.pkl")
app = Flask(__name__)

@app.after_request
def private_response(response):
    response.headers["Cache-Control"] = "no-store"
    return response

def scale(values):
    # Match model.py's fitted feature order and raw-data min/max values.
    return (np.asarray(values, dtype=float) - LOW) / (HIGH - LOW)

def result_for(values):
    x = scale([values])
    prediction = int(model.predict(x)[0])
    # New visualization: each signed term is its contribution to the margin.
    terms = (model.coef_[0] * (x[0] - scale(MEDIANS))).tolist()
    largest = max(max(abs(value) for value in terms), 0.01)
    bars = [{"name": name, "value": round(value, 3),
             "width": round(abs(value) / largest * 46, 2),
             "positive": value >= 0}
            for name, value in zip(FEATURES, terms)]
    return {"prediction": prediction, "margin": float(model.decision_function(x)[0]),
            "bars": bars, "reference": dict(zip(FEATURES, MEDIANS.tolist())),
            "reference_margin": float(model.decision_function([scale(MEDIANS)])[0])}

@app.route("/", methods=["GET"])
@app.route("/predict", methods=["POST"])
def predict():
    values, errors, warnings = {}, {}, []
    result = None
    if request.method == "POST":
        for i, name in enumerate(FEATURES):
            values[name] = request.form.get(name, "").strip()
            try:
                value = float(values[name])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError
                if name == "Age" and (not value.is_integer() or value < 21):
                    errors[name] = "Enter a whole age of at least 21 for this dataset demo."
                elif value < LOW[i] or value > HIGH[i]:
                    errors[name] = "Outside this demo's observed data range. Check the units."
            except (ValueError, OverflowError):
                errors[name] = "Enter a positive number; do not use zero for a missing measurement."
        if not errors:
            result = result_for([float(values[name]) for name in FEATURES])
    fields = [{"name": name, "label": LABELS[i], "unit": UNITS[i],
               "step": "1" if name == "Age" else "any"}
              for i, name in enumerate(FEATURES)]
    return render_template("index.html", fields=fields, values=values,
                           errors=errors, result=result), (400 if errors else 200)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5059, debug=False)
