import pickle
import lightgbm as lgb

INPUT_FILE = "artifacts/best_classifier.pkl"
OUTPUT_FILE = "artifacts/best_classifier.txt"

print("Loading old classifier...")

with open(INPUT_FILE, "rb") as f:
    model = pickle.load(f)

print("Model type:", type(model))

if not isinstance(model, lgb.LGBMClassifier):
    raise TypeError(
        f"Expected LGBMClassifier, but found {type(model)}"
    )

print("Exporting LightGBM classifier to native format...")

model.booster_.save_model(OUTPUT_FILE)

print("SUCCESS!")
print("Created:", OUTPUT_FILE)