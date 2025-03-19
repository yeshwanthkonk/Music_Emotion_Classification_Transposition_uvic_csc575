import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import ast

# Function to convert space-separated feature strings into numpy arrays
def parse_feature(feature_str):
    return np.array([float(x) for x in feature_str.strip("[]").split()])

# Load dataset (assuming it's provided in a CSV or similar format)
# df1 = pd.read_csv("./training/midi_features_with_label.csv")  # Replace with actual file path
df = pd.read_csv("./training/features_with_quadrants.csv")
# df = pd.concat([df1, df2])

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

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)

conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix and save
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - KNN")
plt.savefig("confusion_matrix_knn.png")
plt.close()

# Generate classification report dictionary
report_dict = classification_report(y_test, y_pred, output_dict=True)

# Convert classification report to DataFrame and remove unwanted rows
report_df = pd.DataFrame(report_dict).transpose().drop(index=['accuracy', 'macro avg', 'weighted avg'])

# Create a figure with two subplots for better clarity
plt.figure(figsize=(8, 5))

# Precision and Recall Plot
sns.lineplot(data=report_df, x=report_df.index, y=report_df["precision"], marker="o", label="Precision")
sns.lineplot(data=report_df, x=report_df.index, y=report_df["recall"], marker="s", label="Recall")
plt.ylabel("Score")
plt.title("Precision and Recall for KNN")
plt.legend()
plt.grid(True)

# Save Precision-Recall plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("precision_recall_knn.png")
plt.close()

# F1-Score Plot
plt.figure(figsize=(8, 5))
sns.lineplot(data=report_df, x=report_df.index, y=report_df["f1-score"], marker="^", label="F1-Score")
plt.xlabel("Class Labels")
plt.ylabel("Score")
plt.title("F1-Score for KNN")
plt.legend()
plt.grid(True)

# Save F1-Score plot
plt.xticks(rotation=45)
plt.savefig("f1_score_knn.png")
plt.close()