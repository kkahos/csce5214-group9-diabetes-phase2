# Verification record

## Automated run

- Date: September 23, 2026.
- Platform: Windows; Python 3.12.14.
- Key packages: Flask 3.1.3, NumPy 2.5.3, scikit-learn 1.9.1.
- Command: `python test_extension.py`, from an environment with `requirements-test.txt` installed.
- Exit status: 0.
- Output: [verification/results.json](verification/results.json).

The saved JSON is the actual script output, not an expected-output fixture. The source files published here match the files used in that run.

## Checks that passed

1. **Saved classifier integration.** On all 768 dataset rows, predictions from the original saved linear SVC's compatibility adapter match a linear SVC independently fitted with the original `model.py` procedure. The maximum absolute decision-score difference is approximately `2.77e-13`. This measures implementation consistency, not classification accuracy.
2. **Decision function.** The compact linear coefficient calculation agrees with the direct saved support-vector expansion, within the test's numerical tolerance.
3. **Contribution calculation.** For the fictional example `[148, 155, 33.6, 50]`, the unrounded contributions plus the reference score equal the decision score. The chart rounds displayed terms to three decimals.
4. **Flask request flow.** GET `/` and valid POST `/predict` return HTTP 200. GET `/predict` returns 405. Valid results include the predicted class and chart. Reordered form keys produce the same response because features are read by name. Responses use `Cache-Control: no-store`.
5. **Invalid inputs.** Nine cases return HTTP 400 and a field-level error without the contribution chart: missing insulin, nonnumeric glucose, NaN BMI, infinite insulin, zero glucose, negative insulin, BMI above the dataset range, age below 21 and fractional age.

## Browser check

On September 21, 2026, the local Flask server was started and its form was submitted in Chrome with glucose 148, insulin 155, BMI 33.6 and age 50. The page displayed a positive model prediction and contribution values of +0.773, -0.003, +0.059 and +0.331. The page layout and result were visually inspected. This paragraph records that manual check; the JSON file contains the automated checks above.

## Boundaries

These checks do not cover every possible input, browser or device and do not establish clinical validity or performance on new patients. The upstream notebook's six-model comparison was reviewed, not rerun. The notebook's KNN accuracy of 78.57% is a historical upstream result, not this extension's measured accuracy. New cross-validation, missing-value pipelines and alternative estimators remain proposed future work.

## Reproduce

Install Python 3.12 and run, from this repository's root:

```sh
python -m pip install -r requirements-test.txt
python test_extension.py
```

The script prints a JSON summary only after every assertion succeeds and exits with a nonzero status if a check fails. The independently refitted model is used only by the test; the app uses the saved model parameters.
