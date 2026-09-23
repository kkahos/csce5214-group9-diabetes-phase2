"""Compatibility adapter for this repository's saved binary linear SVC.

No fitting is performed. The saved support vectors, dual coefficients and
intercept define exactly the same linear decision function.
"""
from pathlib import Path
import hashlib
import pickle
import numpy as np
from numpy._core.multiarray import _reconstruct

MODEL_SHA256 = "25328cce4bb89a93dbc3c9002d8ab3900012764b5a0d195b46235fde08ebb972"

class _State:
    pass

class _Reader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("sklearn.svm.classes", "SVC"): _State,
            ("numpy.core.multiarray", "_reconstruct"): _reconstruct,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError("Unexpected object in model file")
        return allowed[module, name]

class SavedLinearSVC:
    def __init__(self, path):
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
            raise ValueError("Model file does not match the reviewed upstream artifact")
        with path.open("rb") as stream:
            state = _Reader(stream).load()
        if state.kernel != "linear" or not np.array_equal(state.classes_, [0, 1]):
            raise ValueError("This adapter supports only the reviewed binary linear SVC")
        self.support_vectors_ = state.support_vectors_
        self.dual_coef_ = state.dual_coef_
        self.coef_ = self.dual_coef_ @ self.support_vectors_
        self.intercept_ = state.intercept_
        self.classes_ = state.classes_
        if self.coef_.shape != (1, 4) or not np.isfinite(self.coef_).all():
            raise ValueError("Invalid coefficient array")

    def decision_function(self, features):
        features = np.asarray(features, dtype=float)
        return (features @ self.coef_.T).ravel() + self.intercept_[0]

    def predict(self, features):
        return self.classes_[(self.decision_function(features) >= 0).astype(int)]
