import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import joblib
import config as config

def load_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    paths = df.iloc[:, 0].values  # First column as paths
    X = df.iloc[:, 1:-1].values  # All columns except the first (path) and the last one (label)
    y = df.iloc[:, -1].values    # The last column
    return paths, X, y, df.columns[1:-1]

def handle_missing_values(X):
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    return X_imputed

def standardize_features(X):
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    return X_standardized, scaler

def correlation_based_feature_selection(X, y, n_features=10,correlation_threshold=0.7):
    mi_scores = mutual_info_classif(X, y)
    corr_matrix = np.abs(np.corrcoef(X.T))
    
    selected_features = []
    remaining_features = list(range(X.shape[1]))
    
    while len(selected_features) < n_features and remaining_features:
        best_feature = max(remaining_features, key=lambda i: mi_scores[i])
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        
        remaining_features = [
            i for i in remaining_features
            if not any(corr_matrix[i][j] > 0.7 for j in selected_features)
        ]
    
    return selected_features

def save_selected_features(selected_features, save_path, filename):
    print(os.path.join(save_path, filename))
    selected_features.to_csv(os.path.join(save_path, filename), index=False)

def load_annotations(annotation_file):
    df = pd.read_csv(annotation_file, sep=' ', header=None, names=['path', 'label'])
    return df['path'].values, df['label'].values
def process_fold(fold):
    annotations_dir = os.path.join(config.MAIN_DATA_DIR, config.DATA_TYPE, f'fold{fold}')
    train_annotations_file = os.path.join(annotations_dir, 'train.txt')
    train_paths, y_train = load_annotations(train_annotations_file)
    
    combine_csv_file_path = os.path.join(config.MAIN_SAVE_DIR, config.DATA_TYPE, f'{config.DATA_TYPE}_combined_features{config.WAYS}.csv')
    combined_paths, X_combined, y_combined, feature_names = load_data(combine_csv_file_path)

    train_indices = np.where(np.isin(combined_paths, train_paths))[0]
    X_train_full = X_combined[train_indices]
    y_train_full = y_combined[train_indices]

    save_base_path = os.path.join(config.MAIN_SAVE_DIR, config.DATA_TYPE, 'CFS', f'fold{fold}')
    os.makedirs(save_base_path, exist_ok=True)
    
    X_train_full = handle_missing_values(X_train_full)

    # Standardize features if required
    if config.Standard:
        X_train_full, scaler = standardize_features(X_train_full)
        joblib.dump(scaler, os.path.join(save_base_path, 'scaler.pkl'))

    for threshold in config.thredhold:
        save_path = os.path.join(save_base_path, f'threshold_{threshold}')
        os.makedirs(save_path, exist_ok=True)

        # Apply CFS with the current threshold
        n_features_to_select = min(config.MAX_FEATURES, X_train_full.shape[1])
        print(f'number of feature to select: {n_features_to_select}')
        selected_features_indices = correlation_based_feature_selection(X_train_full, y_train_full, n_features=n_features_to_select, correlation_threshold=threshold)
        selected_features = feature_names[selected_features_indices]
        feature_count = len(selected_features)
        print(f"Number of selected features: {feature_count}")
        print(f"Selected features: {selected_features}")

        # Save the selected features
        save_selected_features(pd.DataFrame(selected_features, columns=["Selected Features"]), save_path, f'{config.DATA_TYPE}_fold{fold}_selected_features_{config.FEATURE}_{config.MAX_FEATURES}_threshold{threshold}_std{config.Standard}.csv')

        # Print feature importances
        mi_scores = mutual_info_classif(X_train_full, y_train_full)
        feature_importances = sorted(zip(mi_scores, feature_names), reverse=True)
        print("\nFeature Importances:")
        for score, name in feature_importances[:20]:  # Print top 20 features
            print(f"{name}: {score:.4f}")

def main():
    for fold in config.Fold:
        print(f"Processing fold {fold}")
        process_fold(fold)

if __name__ == "__main__":
    main()