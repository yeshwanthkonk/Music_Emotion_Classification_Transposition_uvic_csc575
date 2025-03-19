import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns

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

# Create DataFrame with features and filename for reference
feature_names = [f'mfcc_{i}' for i in range(mfcc_features.shape[1])] + \
                [f'chroma_{i}' for i in range(chroma_features.shape[1])] + \
                [f'spectral_{i}' for i in range(spectral_features.shape[1])] + \
                ['tempo', 'loudness_variation']

X_df = pd.DataFrame(X, columns=feature_names)
X_df['filename'] = df['filename'].values  # Add filename for reference
X_df['Emotion_Description'] = df['Emotion_Description'].values  # Add labels

# Save to CSV for review
# X_df.to_csv("processed_features.csv", index=False)

# Plot feature distributions
plt.figure(figsize=(15, 10))
for i, feature in enumerate(feature_names[:10]):  # Plot first 10 features
    plt.subplot(5, 2, i + 1)
    sns.histplot(X_df, x=feature, hue='Emotion_Description', bins=30, kde=True, alpha=0.7)
    plt.title(f"Distribution of {feature}")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Pairplot of selected features
selected_features = feature_names[:5] + ['Emotion_Description']
sns.pairplot(X_df[selected_features], hue="Emotion_Description", diag_kind="kde")
plt.show()
