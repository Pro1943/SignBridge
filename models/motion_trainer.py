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
ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "dataset" / "motion" / "training" / "motion_sequences_train.csv"
TEST_PATH  = ROOT / "dataset" / "motion" / "testing" / "motion_sequences_test.csv"

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_MODEL = MODELS_DIR / "signbridge_motion_model.pkl"
OUT_PLOT = RESULTS_DIR / "motion_confusion.png"

if not TRAIN_PATH.exists() or not TEST_PATH.exists():
    print("❌ Motion CSVs not found.")
    print(f"Expected:\n- {TRAIN_PATH}\n- {TEST_PATH}")
    print("Run dataset/motion/motion_training_dataset.py and motion_testing_dataset.py first.")
    raise SystemExit(1)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.iloc[:, :-1].values  # (N, 16*63)
y_train = train_df["label"].values

X_test = test_df.iloc[:, :-1].values
y_test = test_df["label"].values

print(f"📊 Motion Train: {len(train_df)} | Motion Test: {len(test_df)}")
print("Train label counts:\n", train_df["label"].value_counts())

print("\n🤖 Training motion model (sequence RandomForest)...")
model = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred) * 100.0

print(f"\n🎯 Motion model accuracy: {acc:.1f}%")
print(classification_report(y_test, pred))

labels = sorted(pd.unique(pd.concat([pd.Series(y_test), pd.Series(pred)])))
cm = confusion_matrix(y_test, pred, labels=labels)

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title(f"Motion Model Confusion Matrix (Acc: {acc:.1f}%)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=200)
plt.show()

joblib.dump(model, OUT_MODEL)
print(f"✅ Saved: {OUT_MODEL}")
print(f"✅ Plot saved: {OUT_PLOT}")
