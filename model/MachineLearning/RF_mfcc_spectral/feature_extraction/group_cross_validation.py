import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score, GroupKFold
import matplotlib.pyplot as plt
import os
import re  # For regex operations

# Function to parse the string representation of lists
def parse_list_string(s):
    if isinstance(s, str):
        return np.fromstring(s.strip('[]"'), sep=' ')
    return np.array([])

# Function to extract or assign patient IDs
def extract_patient_id_from_filename(filename):
    # Example regex to extract patient ID from filename
    match = re.search(r'(Control-\d+|Patient-\d+)', filename)
    if match:
        return match.group(1)
    else:
        return None  # Handle cases where no ID is found

# Function to load and preprocess data, including patient IDs
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
    
    patient_ids = df['record_number'].values
    

    return features, labels, patient_ids

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
    X_train, y_train, train_patient_ids = load_and_preprocess_data(paths['train'])
    X_test, y_test, test_patient_ids = load_and_preprocess_data(paths['test'])
    X_val, y_val, val_patient_ids = load_and_preprocess_data(paths['val'])
    return X_train, y_train, X_test, y_test, X_val, y_val, train_patient_ids, test_patient_ids, val_patient_ids

# Function to evaluate the model
def evaluate_model(y_true, y_pred, y_pred_proba=None):
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1-score: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    
    if y_pred_proba is not None:
        print(f"ROC AUC: {roc_auc_score(y_true, y_pred_proba[:, 1]):.4f}")

# Function to perform grouped cross-validation
def grouped_cross_validate_model(X, y, groups, model, cv=5):
    group_kfold = GroupKFold(n_splits=cv)
    scores = []
    for train_index, test_index in group_kfold.split(X, y, groups=groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Fit the model
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Evaluate metrics
        scores.append({
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba[:, 1]) if y_pred_proba is not None else None
        })
    
    # Print cross-validation results
    for i, score in enumerate(scores):
        print(f"\nFold {i+1} - Accuracy: {score['accuracy']:.4f}, Precision: {score['precision']:.4f}, Recall: {score['recall']:.4f}, F1-score: {score['f1_score']:.4f}")
        if score['roc_auc'] is not None:
            print(f"Fold {i+1} - ROC AUC: {score['roc_auc']:.4f}")
    
    # Compute mean and standard deviation of scores
    accuracy_scores = [score['accuracy'] for score in scores]
    precision_scores = [score['precision'] for score in scores]
    recall_scores = [score['recall'] for score in scores]
    f1_scores = [score['f1_score'] for score in scores]
    roc_auc_scores = [score['roc_auc'] for score in scores if score['roc_auc'] is not None]
    
    print(f"\nMean CV scores: Accuracy - {np.mean(accuracy_scores):.4f}, Precision - {np.mean(precision_scores):.4f}, Recall - {np.mean(recall_scores):.4f}, F1-score - {np.mean(f1_scores):.4f}")
    if roc_auc_scores:
        print(f"Mean CV score: ROC AUC - {np.mean(roc_auc_scores):.4f} (+/- {np.std(roc_auc_scores) * 2:.4f})")

# Function to train and evaluate the Random Forest classifier
def train_and_evaluate(base_path, data_type, top_feature_indices=[8, 21, 1, 15, 23, 9, 25]):
    # Load data
    X_train, y_train, X_test, y_test, X_val, y_val, train_patient_ids, test_patient_ids, val_patient_ids = load_data(base_path, data_type)

    # Select top features
    X_train = X_train[:, top_feature_indices]
    X_test = X_test[:, top_feature_indices]
    X_val = X_val[:, top_feature_indices]

    # Initialize and train the Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Grouped cross-validation on training and validation sets
    print("\nGrouped Cross-validation on Training and Validation Sets:")
    X = np.vstack((X_train, X_val))
    y = np.concatenate((y_train, y_val))
    groups = np.concatenate((train_patient_ids, val_patient_ids))
    grouped_cross_validate_model(X, y, groups, rf_classifier)

    # Fit on the entire training set (including validation set)
    rf_classifier.fit(np.vstack((X_train, X_val)), np.concatenate((y_train, y_val)))

    # Make predictions on test set
    y_pred = rf_classifier.predict(X_test)
    y_pred_proba = rf_classifier.predict_proba(X_test)

    # Calculate accuracy for test set
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f'\nAccuracy on test set: {test_accuracy * 100:.2f}%')

    # Feature importance
    feature_importances = rf_classifier.feature_importances_
    print("\nTop Feature Importances:")
    for i, importance in enumerate(feature_importances):
        print(f'Top Feature {top_feature_indices[i]}: {importance}')

    # Confusion Matrix for test set
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Test Set')
    plt.show()

    # Evaluate the model on test set using additional metrics
    print("\nTest Set Evaluation:")
    evaluate_model(y_test, y_pred, y_pred_proba)

if __name__ == '__main__':
    base_path = r'C:\Users\seera\OneDrive\Desktop\B2AI\data\proceessing'
    # Specify the data type (FIMO, RP, Deep, or Reg)
    data_type = 'FIMO'
    # data_type = 'RP'
    # data_type = 'Deep'
    # data_type = 'Reg'
    train_and_evaluate(base_path, data_type)
