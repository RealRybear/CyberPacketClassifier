import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import necessary modules for model training and evaluation
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# ============================
# STEP 1: Load the Dataset
# ============================
file_path = "traffic.csv"  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# ============================
# STEP 2: Clean Column Names
# ============================
# Remove any leading or trailing spaces from the column names
df.columns = df.columns.str.strip()

# ============================
# STEP 3: Encode the Target Variable
# ============================
# Assume that the 'Label' column indicates if a packet is evil (malicious) or not.
# We use LabelEncoder to convert text labels (if any) into numeric values.
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])

# ============================
# STEP 4: Select Relevant Features
# ============================
# Choose a subset of features that might be useful for classification.
features = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
    'Flow Bytes/s', 'Flow Packets/s', 'SYN Flag Count', 'FIN Flag Count',
    'Packet Length Mean', 'Packet Length Std'
]

# Create a new DataFrame with only the selected features and the target label.
df_selected = df[features + ['Label']].copy()

# ============================
# STEP 5: Data Cleaning and Preprocessing
# ============================
# a) Convert the feature columns to numeric, coercing errors into NaN
df_selected[features] = df_selected[features].apply(pd.to_numeric, errors='coerce')

# b) Replace infinite values with NaN
df_selected.replace([np.inf, -np.inf], np.nan, inplace=True)

# c) Drop any rows with missing values
df_selected.dropna(inplace=True)

# d) Cap extreme values at the 99th percentile to reduce the impact of outliers
for col in features:
    cap = df_selected[col].quantile(0.99)
    df_selected[col] = np.where(df_selected[col] > cap, cap, df_selected[col])

# ============================
# STEP 6: Normalize the Features
# ============================
# Scale the features so that they have a mean of 0 and a standard deviation of 1.
scaler = StandardScaler()
df_selected[features] = scaler.fit_transform(df_selected[features])

# =========================================================
# STEP 7: Split the Data into Training and Testing Sets
# ========================================================
# We use 80% of the data for training and 20% for testing.
X = df_selected[features]  # The features used for training
y = df_selected['Label']   # The target variable (0 = not evil, 1 = evil)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================================
# STEP 8: Train a Random Forest Classifier
# =========================================
# Create and train the Random Forest model.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ============================
# STEP 9: Evaluate the Model
# ============================
# a) Make predictions on the test set
y_pred = model.predict(X_test)

# b) Print the classification report and confusion matrix to evaluate performance
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ========================================
# STEP 10: Plot the ROC Curve (Optional)
# =======================================
# This curve helps visualize the trade-off between the True Positive Rate and False Positive Rate.
y_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class (evil)
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve for Evil Packet Classification', fontsize=14)
plt.legend(loc="lower right")
plt.show()
