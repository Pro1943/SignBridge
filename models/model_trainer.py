import os
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# Paths (match your structure)
# =========================
ROOT = Path(__file__).resolve().parents[1]  # .../SignBridge
TRAIN_FILE = ROOT / "dataset" / "static" / "training" / "signbridge_landmarks_train.csv"
TEST_FILE  = ROOT / "dataset" / "static" / "testing" / "signbridge_landmarks_test.csv"

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_MODEL = MODELS_DIR / "signbridge_model.pkl"
OUT_PLOT = RESULTS_DIR / "accuracy.png"

# =========================
# Load data
# =========================
if not TRAIN_FILE.exists() or not TEST_FILE.exists():
    print("❌ Static CSVs not found.")
    print(f"Expected:\n- {TRAIN_FILE}\n- {TEST_FILE}")
    print("Run dataset/static/static_training_dataset.py and static_testing_dataset.py first.")
    raise SystemExit(1)

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"📊 Train samples: {len(train_df)} | Test samples: {len(test_df)}")
print("Train label counts:\n", train_df["label"].value_counts())

X_train = train_df.iloc[:, :-1].values  # 63 features
y_train = train_df["label"].values

X_test = test_df.iloc[:, :-1].values
y_test = test_df["label"].values

# =========================
# Train
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# =========================
# Evaluate
# =========================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100.0

print(f"\n🎯 ACCURACY: {acc:.1f}%")
print(classification_report(y_test, y_pred))

# Confusion matrix plot
labels = sorted(pd.unique(pd.concat([pd.Series(y_test), pd.Series(y_pred)])))
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title(f"Static Model Confusion Matrix (Acc: {acc:.1f}%)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=200)
plt.show()

# =========================
# Save
# =========================
joblib.dump(model, OUT_MODEL)
print(f"✅ Model saved: {OUT_MODEL}")
print(f"✅ Plot saved: {OUT_PLOT}")
print("✅ Model training complete.")
# =========================