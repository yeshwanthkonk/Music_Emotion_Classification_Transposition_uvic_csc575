import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Function to convert space-separated feature strings into numpy arrays
def parse_feature(feature_str):
    return np.array([float(x) for x in feature_str.strip("[]").split()])

# Load dataset (replace with actual file path)
df = pd.read_csv("../midi_features_with_label.csv")  # Replace with actual dataset file path

# Convert feature columns from string to numerical arrays
df['mfcc_mean'] = df['mfcc_mean'].apply(parse_feature)
df['chroma_mean'] = df['chroma_mean'].apply(parse_feature)
df['spectral_contrast_mean'] = df['spectral_contrast_mean'].apply(parse_feature)

# Flatten the features
mfcc_features = np.vstack(df['mfcc_mean'].values)
chroma_features = np.vstack(df['chroma_mean'].values)
spectral_features = np.vstack(df['spectral_contrast_mean'].values)

# Combine all features into a single feature matrix
X = np.hstack([mfcc_features, chroma_features, spectral_features, df[['tempo', 'loudness_variation']].values])

# Target variable
y = df['Emotion_Description'].values

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM model with RBF kernel
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)

# Predictions
y_pred = svm_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)
