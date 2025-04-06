import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def train_rf_model(features_file, output_dir, test_size=0.2, random_state=42):
    """
    Train a Random Forest model on the extracted features.
    
    Args:
        features_file: Path to the CSV file containing features
        output_dir: Directory to save the model and results
        test_size: Proportion of dataset to include in test split
        random_state: Random seed for reproducibility
    
    Returns:
        Trained model and evaluation metrics
    """
    # Load features
    df = pd.read_csv(features_file)
    
    # Separate metadata
    file_paths = df['file_path'].values
    y = df['class'].values
    
    # Remove non-feature columns
    X = df.drop(['file_path', 'class'], axis=1)
    
    # Handle any non-numeric columns
    for col in X.columns:
        if X[col].dtype == object:
            X = X.drop(col, axis=1)
    
    # Split the data
    X_train, X_test, y_train, y_test, train_paths, test_paths = train_test_split(
        X, y, file_paths, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Feature importance
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and scaler
    joblib.dump(clf, os.path.join(output_dir, 'rf_model.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    
    # Save metrics and results
    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix.tolist(),
        'feature_importances': feature_importances.to_dict(),
        'test_files': test_paths.tolist(),
        'test_actual': y_test.tolist(),
        'test_predicted': y_pred.tolist()
    }
    
    # Save to CSV for easy analysis
    pd.DataFrame({
        'file_path': test_paths,
        'actual': y_test,
        'predicted': y_pred
    }).to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    
    feature_importances.to_csv(os.path.join(output_dir, 'feature_importances.csv'), index=False)
    
    print(f"Model training complete. Accuracy: {accuracy:.4f}")
    print(f"Results saved to {output_dir}")
    
    return clf, results

if __name__ == "__main__":
    train_rf_model(
        features_file="../../data/features/audio_features.csv",
        output_dir="../../results/rf_model"
    ) 