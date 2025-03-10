import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset (update the filename as needed)
file_path = "traffic.csv"  # Replace with your actual CSV file
df = pd.read_csv(file_path)

# Clean column names to remove leading/trailing spaces
df.columns = df.columns.str.strip()

# Encode the 'Label' column
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])

# Select relevant features for visualization and modeling
features = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
    'Flow Bytes/s', 'Flow Packets/s', 'SYN Flag Count', 'FIN Flag Count',
    'Packet Length Mean', 'Packet Length Std'
]

# Use .copy() to avoid potential chained assignment issues
df_selected = df[features + ['Label']].copy()

# --- Data Cleaning ---
# Convert feature columns to numeric (force non-numeric to NaN)
df_selected[features] = df_selected[features].apply(pd.to_numeric, errors='coerce')

# Replace infinite values with NaN
df_selected.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN values
df_selected.dropna(inplace=True)

# Cap extreme values at the 99th percentile
for col in features:
    cap = df_selected[col].quantile(0.99)
    df_selected[col] = np.where(df_selected[col] > cap, cap, df_selected[col])

# --- Normalization ---
scaler = StandardScaler()
df_selected[features] = scaler.fit_transform(df_selected[features])

# --- Visualization: Correlation Heatmap ---
plt.figure(figsize=(12, 6))
sns.heatmap(df_selected.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

# --- Visualization: Distribution of Normal vs Malicious Traffic ---
plt.figure(figsize=(8, 5))
sns.countplot(x=df_selected['Label'], palette=['blue', 'red'])
plt.title("Distribution of Normal vs Malicious Traffic")
plt.xlabel("Traffic Type (0 = Normal, 1 = Malicious)")
plt.ylabel("Count")
plt.show()

# --- Split dataset for training and testing ---
X = df_selected[features]
y = df_selected['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train a Random Forest Classifier ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# --- Evaluation Metrics ---
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Visualization: Feature Importance ---
feature_importances = pd.Series(model.feature_importances_, index=features)
feature_importances.sort_values().plot(kind='barh', figsize=(10, 6), title='Feature Importance in Malicious Packet Detection')
plt.show()

# ===== Additional Visualizations =====

# 1. ROC Curve Visualization
from sklearn.metrics import roc_curve, auc

# Get the predicted probabilities for the positive class
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.show()

# 2. Improved Parallel Coordinates Plot
from pandas.plotting import parallel_coordinates

# --- A) Pick Top Features for Clarity ---
# Sort features by importance
importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]  # descending order
top_n = 6  # Number of features to visualize
top_features = [features[i] for i in sorted_idx[:top_n]]

# Create a smaller DataFrame with top features + Label
df_top = df_selected[top_features + ['Label']].copy()

# --- B) Sample Data to Avoid Overcrowding ---
df_sampled = df_top.sample(n=200, random_state=42)  # adjust n if needed

# --- C) Plot Parallel Coordinates ---
plt.figure(figsize=(12, 8))
# Use distinct colors for classes, plus alpha for transparency
parallel_coordinates(
    df_sampled, 
    class_column='Label', 
    color=['#1f77b4', '#ff7f0e'], 
    alpha=0.4
)

plt.title('Parallel Coordinates Plot (Top Features)', fontsize=14)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Normalized Value', fontsize=12)
plt.legend(title='Traffic Type\n(0: Normal, 1: Malicious)', fontsize=10, loc='upper right')
plt.grid(True)
plt.show()
