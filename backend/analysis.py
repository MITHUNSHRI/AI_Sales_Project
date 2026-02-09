import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "customers.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "lead_model.pkl")

data = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

X = data[['engagement','company_size','visits']]
y = data['converted']

pred = model.predict(X)

acc = accuracy_score(y, pred)

print("✅ Model Accuracy:", acc*100, "%")
