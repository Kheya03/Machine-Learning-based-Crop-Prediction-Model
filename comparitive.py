import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
file_path = r"C:\Users\bhara\OneDrive\Documents\Crop_recommendation.csv" 
df = pd.read_csv(file_path)

# Check dataset structure
print(df.head())

# Splitting features and target
X = df.drop(columns=['crop'])  # Assuming 'crop' is the target column
y = df['crop']

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardizing the features for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
knn = KNeighborsClassifier(n_neighbors=5)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# Train and predict
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

# Accuracy scores
acc_knn = accuracy_score(y_test, y_pred_knn)
acc_gb = accuracy_score(y_test, y_pred_gb)

print("KNN Accuracy:", acc_knn)
print("Gradient Boosting Accuracy:", acc_gb)

# Classification reports
print("\nKNN Classification Report:\n", classification_report(y_test, y_pred_knn))
print("\nGradient Boosting Classification Report:\n", classification_report(y_test, y_pred_gb))

# Sensitivity and Specificity Calculation Function
def calculate_sensitivity_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sensitivity = np.mean(np.diag(cm) / np.sum(cm, axis=1))
    specificity = np.mean([
        (np.sum(cm) - (np.sum(cm, axis=0)[i] + np.sum(cm, axis=1)[i] - cm[i, i])) /
        (np.sum(cm) - np.sum(cm, axis=1)[i])
        for i in range(len(cm))
    ])
    return sensitivity, specificity

# Calculate sensitivity and specificity
sensitivity_knn, specificity_knn = calculate_sensitivity_specificity(y_test, y_pred_knn)
sensitivity_gb, specificity_gb = calculate_sensitivity_specificity(y_test, y_pred_gb)

# Print sensitivity and specificity
print("\nKNN Sensitivity (Recall): {:.4f}".format(sensitivity_knn))
print("KNN Specificity: {:.4f}".format(specificity_knn))

print("\nGradient Boosting Sensitivity (Recall): {:.4f}".format(sensitivity_gb))
print("Gradient Boosting Specificity: {:.4f}".format(specificity_gb))
# Comparative accuracy plot
models = ['KNN', 'Gradient Boosting']
accuracies = [acc_knn, acc_gb]

plt.figure(figsize=(6, 4))
sns.barplot(x=models, y=accuracies, palette="viridis")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Comparison of KNN and Gradient Boosting")
plt.ylim(0, 1)
plt.show()
# Confusion matrix visualization
plt.figure(figsize=(10, 4))

for i, (model_name, y_pred) in enumerate(zip(['KNN', 'Gradient Boosting'], [y_pred_knn, y_pred_gb])):
    plt.subplot(1, 2, i+1)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

plt.tight_layout()
plt.show()
