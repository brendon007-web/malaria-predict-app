import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle # Make sure this is imported

# --- 1. Load and Prepare Data (No changes here) ---
df = pd.read_csv('malaria_vitals_large_dataset.csv')
X = df.drop('Malaria_Positive', axis=1)
y = df['Malaria_Positive']

# --- 2. Split Data (No changes here) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# --- 3. Train the Model (No changes here) ---
print("Training the Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Model training complete.")

# --- 4. Evaluate Model (No changes here) ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# --- 5. Classification Report (No changes here) ---
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['No Malaria', 'Malaria']))

# --- 6. NEW: Save the Trained Model ---
# This saves your model into a single file named 'malaria_model.pkl'.
with open('malaria_model.pkl', 'wb') as file:
    pickle.dump(model, file)

