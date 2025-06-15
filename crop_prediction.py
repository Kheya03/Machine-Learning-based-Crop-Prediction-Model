import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load dataset
file_path = r"C:\Users\bhara\OneDrive\Documents\Crop_recommendation.csv" 
try:
    df = pd.read_csv(file_path)
    print("\n✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("\n❌ Error: Dataset file not found. Check file name and path.")
    exit()

# Display dataset summary
print("\n🔹 First few rows of dataset:\n", df.head())
print("\n🔹 Column names:", df.columns)
print("\n🔹 Missing Values:\n", df.isnull().sum())

# ✅ Display additional statistics: Standard Deviation, Min, Max
print("\n🔹 Statistical Summary:")
print(df.describe().T[['std', 'min', 'max']])  # Shows std, min, max for each column

# Splitting features and target variable
X = df.drop(columns=['crop'])  
y = df['crop']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model & scaler
with open("crop_prediction_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ✅ Model Evaluation
y_pred = model.predict(X_test_scaled)
print("\n🔹 Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))
# ✅ Calculate Sensitivity (Recall) and Specificity

# Extract True Negatives (TN), False Positives (FP), False Negatives (FN), and True Positives (TP)
from sklearn.metrics import multilabel_confusion_matrix

# For multi-class classification, use multilabel_confusion_matrix
mcm = multilabel_confusion_matrix(y_test, y_pred)

# Initialize lists to store sensitivity and specificity for each class
sensitivity_list = []
specificity_list = []

# Calculate sensitivity and specificity for each class
for idx, matrix in enumerate(mcm):
    tn, fp, fn, tp = matrix.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    sensitivity_list.append(sensitivity)
    specificity_list.append(specificity)

# Display average sensitivity and specificity
avg_sensitivity = np.mean(sensitivity_list)
avg_specificity = np.mean(specificity_list)

print(f"\n🔹 Average Sensitivity (Recall): {avg_sensitivity:.4f}")
print(f"🔹 Average Specificity: {avg_specificity:.4f}")

# Optional: Display sensitivity and specificity for each class
for idx, label in enumerate(np.unique(y)):
    print(f"\nClass: {label}")
    print(f"   Sensitivity (Recall): {sensitivity_list[idx]:.4f}")
    print(f"   Specificity: {specificity_list[idx]:.4f}")


# ✅ Predict Crop (Ensuring It Works)
sample_input = np.array([[80, 40, 45, 28.0, 70.0, 5.8, 200]])  
sample_input_scaled = scaler.transform(sample_input)
predicted_crop = model.predict(sample_input_scaled)

# ✅ Print predicted crop BEFORE confusion matrix
print("\n✅ Predicted Crop:", predicted_crop[0])

# ✅ Generate & Display Confusion Matrix **after everything else**
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
