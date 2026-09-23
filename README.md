# Diabetes Prediction Phase 2

CSCE 5214 - Project Group 9

A local Flask extension of [Aditya Mankar's Diabetes Prediction project](https://github.com/Aditya-Mankar/Diabetes-Prediction), reviewed at upstream commit [`f380c23b79f18ae1845b5cf229ff0eb032e2da17`](https://github.com/Aditya-Mankar/Diabetes-Prediction/tree/f380c23b79f18ae1845b5cf229ff0eb032e2da17).

## Review the evidence

- [Verification scope and reproduction commands](VERIFICATION.md)
- [Actual automated verification output from September 23, 2026](verification/results.json)
- [Test script](test_extension.py)
- [Flask routes and contribution calculation](app.py)
- [User interface](templates/index.html)
- [Saved-model compatibility adapter](saved_model.py)

The 768/768 result means prediction agreement between two implementations. It is **not 100% classification accuracy**. The six-model accuracy table in the report comes from outputs saved in the upstream notebook; that six-model comparison was not rerun for this extension.

## Implemented changes

- Reuses the original saved linear SVC without fitting a replacement during application startup.
- Corrects the scaler's feature order to **Glucose, Insulin, BMI, Age**, matching the upstream training script.
- Adds persistent field labels, measurement units and server-side input validation.
- Shows a positive or negative model prediction and signed feature-contribution bars relative to nonzero dataset medians.
- Provides Clear form and About this result controls, with a responsive two-column layout.

The chart explains the numerical decision score. It does not show calibrated probabilities, causal effects or treatment recommendations. The application is an educational prototype, not a diagnostic tool. It does not save submitted measurements.

## Run locally

Use Python 3.12. Download this repository or clone it, then run the following commands from its root folder.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe app.py
```

### macOS or Linux

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

Open **http://127.0.0.1:5059**. The server listens on the local machine with debug mode disabled. GitHub hosts the source files; it does not run this Flask app.

Example fictional profile: glucose **148**, insulin **155**, BMI **33.6**, age **50**. It produces the positive model prediction and contribution values recorded in the verification output.

## Run the automated checks

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.venv\Scripts\python.exe test_extension.py
```

On macOS/Linux, replace `.venv\Scripts\python.exe` with `.venv/bin/python`. See [VERIFICATION.md](VERIFICATION.md) for exactly what these checks establish.

## Saved model and preprocessing

The upstream `model.pkl` was written with scikit-learn 0.20.1. `saved_model.py` checks that exact artifact's SHA-256 and uses a restricted reader to load its numerical state into a plain object. For the saved binary linear SVC, `coef = dual_coef @ support_vectors` reconstructs its decision function. The adapter exposes `predict` and `decision_function` without retraining the saved model.

The application reconstructs the raw-data minimum and maximum values used by upstream `model.py`, with the correct named features. This fixes the Insulin/BloodPressure mapping error but retains the original model's preprocessing limitations. The training-only imputation pipeline, cross-validation and alternative models discussed in the report remain proposals.

## Attribution

`model.pkl` and `diabetes.csv` are unchanged copies from the upstream `flask` directory at the commit above. The dataset is the public Pima Indians Diabetes Database. The original MIT license is retained in [UPSTREAM_LICENSE](UPSTREAM_LICENSE). The Flask extension, interface, compatibility adapter and verification script are the Phase 2 additions.
