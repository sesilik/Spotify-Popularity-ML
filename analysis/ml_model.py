import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import os

# === Paths ===
data_path = os.path.join("..", "data", "spotify_house_music.csv")
output_dir = os.path.join("..", "outputs")
model_dir = os.path.join("..", "models")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# === Load Data ===
df = pd.read_csv(data_path)

print("Dataset loaded. Shape:", df.shape)

# === Create a Popularity Label ===
# Let's turn popularity into a classification target:
#   1 = Popular (top 30%)
#   0 = Not Popular

threshold = df["popularity"].quantile(0.70)
df["popular_label"] = (df["popularity"] >= threshold).astype(int)

print("\nPopularity Threshold:", threshold)
print("Counts:\n", df["popular_label"].value_counts())

# === Feature Selection ===
features = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms"
]

X = df[features]
y = df["popular_label"]

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# === Scale Features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Train Model ===
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# === Predictions ===
y_pred = model.predict(X_test_scaled)

# === Evaluation ===
acc = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print("\nModel Accuracy:", round(acc, 4))
print("\nConfusion Matrix:\n", conf_mat)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# === Feature Importance Plot ===
importances = model.feature_importances_
feat_imp = pd.DataFrame({
    "feature": features,
    "importance": importances
}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(x="importance", y="feature", data=feat_imp, palette="viridis")
plt.title("Feature Importance — What Makes a Track Popular?")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importance.png"))
plt.show()

# === Save model ===
import joblib
joblib.dump(model, os.path.join(model_dir, "spotify_popularity_model.pkl"))
print("\nModel saved successfully!")


