import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def load_and_analyze_features(features_csv):
    """
    Load features from CSV and perform basic analysis.
    
    Args:
        features_csv: Path to the CSV file with extracted features
        
    Returns:
        DataFrame with features and analysis results
    """
    # Load features
    df = pd.read_csv(features_csv)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Number of classes: {df['class'].nunique()}")
    print(f"Class distribution:\n{df['class'].value_counts()}")
    
    # Count number of segments per file
    file_counts = df['file_path'].value_counts()
    print(f"Average segments per file: {file_counts.mean():.2f}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("Columns with missing values:")
        print(missing[missing > 0])
    
    return df

def visualize_feature_distributions(df, output_dir):
    """
    Create visualizations of feature distributions.
    
    Args:
        df: DataFrame with features
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get feature columns (exclude metadata)
    feature_cols = [col for col in df.columns if col not in ['file_path', 'class', 'segment_id']]
    
    # If there are too many features, select a subset
    if len(feature_cols) > 20:
        # Focus on MFCC means as they're often most informative
        mfcc_mean_cols = [col for col in feature_cols if 'mfcc_mean' in col]
        if len(mfcc_mean_cols) > 0:
            selected_features = mfcc_mean_cols[:20]
        else:
            selected_features = feature_cols[:20]
    else:
        selected_features = feature_cols
    
    # Visualize distributions for each class
    for feature in selected_features:
        plt.figure(figsize=(12, 6))
        for class_name in df['class'].unique():
            class_data = df[df['class'] == class_name][feature]
            sns.kdeplot(class_data, label=class_name)
        
        plt.title(f'Distribution of {feature} by Class')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{feature}_distribution.png'))
        plt.close()
    
    # Correlation matrix of features
    plt.figure(figsize=(20, 16))
    corr_matrix = df[feature_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=False, 
                linewidths=.5, vmin=-1, vmax=1)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlation.png'))
    plt.close()

def visualize_feature_importance(df, output_dir):
    """
    Visualize feature importance using variance and PCA.
    
    Args:
        df: DataFrame with features
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in ['file_path', 'class', 'segment_id']]
    
    # Calculate variance
    feature_variance = df[feature_cols].var().sort_values(ascending=False)
    
    # Plot top 20 features by variance
    plt.figure(figsize=(12, 8))
    feature_variance.head(20).plot(kind='bar')
    plt.title('Top 20 Features by Variance')
    plt.ylabel('Variance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_variance.png'))
    plt.close()
    
    # PCA for dimensionality reduction
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'pca_explained_variance.png'))
    plt.close()
    
    # Plot first two principal components
    plt.figure(figsize=(12, 10))
    for class_name in df['class'].unique():
        mask = df['class'] == class_name
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=class_name, alpha=0.7)
    
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA Projection of Audio Features')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'pca_projection.png'))
    plt.close()
    
    # t-SNE for more complex manifold visualization (on a sample if dataset is large)
    if len(df) > 5000:
        sample_idx = np.random.choice(len(df), 5000, replace=False)
        X_sample = X_scaled[sample_idx]
        classes_sample = df['class'].values[sample_idx]
    else:
        X_sample = X_scaled
        classes_sample = df['class'].values
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_sample)
    
    # Plot t-SNE projection
    plt.figure(figsize=(12, 10))
    for class_name in np.unique(classes_sample):
        mask = classes_sample == class_name
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=class_name, alpha=0.7)
    
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Projection of Audio Features')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'tsne_projection.png'))
    plt.close()

if __name__ == "__main__":
    # Load and analyze features
    features_csv = '../../data/processed/features_dataset.csv'
    df = load_and_analyze_features(features_csv)
    
    # Visualize feature distributions
    output_dir = '../../data/analysis'
    visualize_feature_distributions(df, output_dir)
    visualize_feature_importance(df, output_dir) 