import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from app import app, model, DATA, FEATURES, MEDIANS, rows, scale, result_for

x = scale(DATA)
y = np.array([int(r['Outcome']) for r in rows])
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y)
reference = SVC(kernel='linear', random_state=42).fit(x_train, y_train)
expected = reference.predict(x)
actual = model.predict(x)
assert np.array_equal(actual, expected)
expanded = (x @ model.support_vectors_.T) @ model.dual_coef_.ravel() + model.intercept_[0]
np.testing.assert_allclose(model.decision_function(x), expanded, atol=1e-10)
profile = [148, 155, 33.6, 50]
explanation = result_for(profile)
terms = model.coef_[0] * (scale(profile) - scale(MEDIANS))
np.testing.assert_allclose(terms.sum() + explanation['reference_margin'], explanation['margin'])
client = app.test_client()
assert client.get('/').status_code == 200
assert client.get('/predict').status_code == 405
valid = dict(zip(FEATURES, map(str, profile)))
response = client.post('/predict', data=valid)
assert response.status_code == 200
assert b'What changed the model score' in response.data
assert b'Positive model prediction' in response.data if explanation['prediction'] else b'Negative model prediction' in response.data
assert response.headers['Cache-Control'] == 'no-store'
assert client.post('/predict', data=dict(reversed(list(valid.items())))).data == response.data
for key, bad in [('Insulin',''),('Glucose','abc'),('BMI','nan'),('Insulin','inf'),
                 ('Glucose','0'),('Insulin','-1'),('BMI','100'),('Age','20'),('Age','30.5')]:
    payload = dict(valid)
    payload[key] = bad
    response_bad = client.post('/predict', data=payload)
    assert response_bad.status_code == 400, (key,bad)
    assert b'aria-invalid="true"' in response_bad.data
    assert b'What changed the model score' not in response_bad.data
summary = {'saved_model_rows_checked': len(DATA),
           'predictions_matching_independent_refit': int((actual == expected).sum()),
           'max_margin_difference_vs_refit': float(np.max(abs(model.decision_function(x)-reference.decision_function(x)))),
           'invalid_input_cases': 9, 'all_checks_passed': True,
           'example': explanation}
print(json.dumps(summary, indent=2))
