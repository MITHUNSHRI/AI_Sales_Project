import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "customers.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "lead_model.pkl")

# Load data
data = pd.read_csv(DATA_PATH)

X = data[['engagement', 'company_size', 'visits']]
y = data['converted']

# Train
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Save model
joblib.dump(model, MODEL_PATH)

print("✅ Model trained and saved!")
