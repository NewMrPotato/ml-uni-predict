import numpy as np
import joblib
from unipredict import UniPredictor


reg_model = joblib.load('test_models/sklearn/rf_regressor.pkl')
feature_names = np.load('test_models/sklearn/feature_names_reg.npy', allow_pickle=True).tolist()
X = np.load('test_models/sklearn/X_reg.npy')

class SimpleThresholdClassifier:
    def __init__(self, threshold=0.0):
        self.threshold = threshold
    def predict(self, X):
        sums = X.sum(axis=1)
        return (sums > self.threshold).astype(int)

clf_model = SimpleThresholdClassifier(threshold=0.0)
clf_predictor = UniPredictor(model=clf_model, feature_names=feature_names)
reg_predictor = UniPredictor(model=reg_model, feature_names=feature_names)

def cascade_predict(data, clf_pred, reg_pred, threshold=0.5):
    classes = clf_pred.predict(data)
    if classes.ndim == 1:
        classes = classes.reshape(-1, 1)
    prob_class1 = classes.flatten()
    
    reg_result = reg_pred.predict(data)
    if reg_result.ndim == 1:
        reg_result = reg_result.reshape(-1, 1)
    
    mask = prob_class1 > threshold
    result = np.zeros_like(reg_result)
    result[mask] = reg_result[mask]
    return result

print("=== КАСКАДНЫЙ ПРИМЕР ===\n")
sample = {name: X[0, i] for i, name in enumerate(feature_names)}
result = cascade_predict(sample, clf_predictor, reg_predictor)
print(f"Одиночный объект: {result[0][0]:.4f}")

batch = {name: X[:5, i].tolist() for i, name in enumerate(feature_names)}
result_batch = cascade_predict(batch, clf_predictor, reg_predictor)
print(f"Батч: {result_batch.flatten()}")

negative_sample = {name: -1.0 for name in feature_names}
result_neg = cascade_predict(negative_sample, clf_predictor, reg_predictor)
print(f"Отрицательный объект: {result_neg[0][0]:.4f} (ожидается 0)")

print("\n✅ Пример каскада завершён.")