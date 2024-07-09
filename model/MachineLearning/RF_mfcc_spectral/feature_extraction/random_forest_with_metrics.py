import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import os

# Function to parse the string representation of lists
def parse_list_string(s):
    if isinstance(s, str):
        return np.fromstring(s.strip('[]"'), sep=' ')
    return np.array([])

# Function to load and preprocess data
def load_and_preprocess_data(file_path, is_excel=True):
    if is_excel:
        df = pd.read_excel(file_path, header=0)
    else:
        df = pd.read_csv(file_path, header=0)
    
    # Columns to parse as lists
    columns_to_parse = [
        'mfcc_mean', 'mfcc_var', 'spectral_centroid_mean', 'spectral_bandwidth_mean',
        'spectral_contrast_mean', 'spectral_rolloff_mean', 'zero_crossing_rate_mean'
    ]
    
    # Parse the list columns
    for column in columns_to_parse:
        df[column] = df[column].apply(parse_list_string)
    
    # Concatenate the features into a single array
    features = np.hstack([
        np.vstack(df[column].values) for column in columns_to_parse
    ])
    
    # Extract labels
    labels = df['label'].values
    
    return features, labels

# Function to construct file paths for the datasets
def get_data_paths(base_path, data_type):
    return {
        'train': os.path.join(base_path, 'audio_features', data_type, 'train.xlsx'),
        'test': os.path.join(base_path, 'audio_features', data_type, 'test.xlsx'),
        'val': os.path.join(base_path, 'audio_features', data_type, 'val.xlsx')
    }

# Function to load datasets
def load_data(base_path, data_type):
    paths = get_data_paths(base_path, data_type)
    X_train, y_train = load_and_preprocess_data(paths['train'])
    X_test, y_test = load_and_preprocess_data(paths['test'])
    X_val, y_val = load_and_preprocess_data(paths['val'])
    return X_train, y_train, X_test, y_test, X_val, y_val

# Function to evaluate the model
def evaluate_model(y_true, y_pred, y_pred_proba=None):
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1-score: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    
    if y_pred_proba is not None:
        print(f"ROC AUC: {roc_auc_score(y_true, y_pred_proba[:, 1]):.4f}")

# Function to perform cross-validation
def cross_validate_model(X, y, model, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"Cross-validation scores: {scores}")
    print(f"Mean CV score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Function to train and evaluate the Random Forest classifier
def train_and_evaluate(base_path, data_type, top_feature_indices=[8, 21, 1, 15, 23, 9, 25]):
    # Load data
    X_train, y_train, X_test, y_test, X_val, y_val = load_data(base_path, data_type)

    # Select top features
    X_train = X_train[:, top_feature_indices]
    X_test = X_test[:, top_feature_indices]
    X_val = X_val[:, top_feature_indices]

    # Combine all data for cross-validation
    X = np.vstack((X_train, X_test, X_val))
    y = np.concatenate((y_train, y_test, y_val))

    # Initialize and train the Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Cross-validation
    cross_validate_model(X, y, rf_classifier)

    # Make predictions on training set
    y_train_pred = rf_classifier.predict(X_train)
    y_train_pred_proba = rf_classifier.predict_proba(X_train)

    # Make predictions on test set
    y_test_pred = rf_classifier.predict(X_test)
    y_test_pred_proba = rf_classifier.predict_proba(X_test)

    # Calculate accuracy for training set
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f'Accuracy on training set: {train_accuracy * 100:.2f}%')

    # Calculate accuracy for test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f'Accuracy on test set: {test_accuracy * 100:.2f}%')

    # Feature importance
    feature_importances = rf_classifier.feature_importances_
    print("Top Feature Importances:")
    for i, importance in enumerate(feature_importances):
        print(f'Top Feature {top_feature_indices[i]}: {importance}')

    # Confusion Matrix for test set
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Test Set')
    plt.show()

    # Evaluate the model on training set using additional metrics
    print("\nTraining Set Evaluation:")
    evaluate_model(y_train, y_train_pred, y_train_pred_proba)

    # Evaluate the model on test set using additional metrics
    print("\nTest Set Evaluation:")
    evaluate_model(y_test, y_test_pred, y_test_pred_proba)

if __name__ == '__main__':
    base_path = r'C:\Users\seera\OneDrive\Desktop\B2AI\data\proceessing'
    # Specify the data type (FIMO, RP, Deep, or Reg)
    data_type = 'FIMO'
    # data_type = 'RP'
    # data_type = 'Deep'
    # data_type = 'Reg'
    train_and_evaluate(base_path, data_type)
