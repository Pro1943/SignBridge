import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load + combine datasets
os.makedirs("dataset/training", exist_ok=True)
os.makedirs("dataset/testing", exist_ok=True)

train_file = "dataset/training/signbridge_landmarks_train.csv"
test_file = "dataset/testing/signbridge_landmarks_test.csv"

if os.path.exists(train_file) and os.path.exists(test_file):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    df = pd.concat([train_df, test_df], ignore_index=True)
else:
    print("❌ Run training_dataset.py & testing_dataset.py first!")
    exit()

print(f"📊 Total: {len(df)} samples")
print(df['label'].value_counts())

X = df.iloc[:, :-1].values  # 63 landmarks
y = df['label'].values

# Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print(f"\n🎯 ACCURACY: {acc:.1f}%")
print(classification_report(y_test, y_pred))

# Plot
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title(f'Accuracy: {acc:.1f}%')
plt.savefig('accuracy.png')
plt.show()

joblib.dump(model, 'signbridge_model.pkl')
print("✅ Model saved!")
print("✅ Model training complete.")