import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

TRAIN_PATH = "dataset/motion/training/motion_sequences_train.csv"
TEST_PATH  = "dataset/motion/testing/motion_sequences_test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.iloc[:, :-1].values  # (N, 16*63)
y_train = train_df["label"].values

X_test = test_df.iloc[:, :-1].values
y_test = test_df["label"].values

print("🤖 Training motion model (sequence classifier)...")
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred) * 100

print(f"🎯 Motion model accuracy: {acc:.1f}%")
print(classification_report(y_test, pred))

joblib.dump(model, "signbridge_motion_model.pkl")
print("✅ Saved: signbridge_motion_model.pkl")