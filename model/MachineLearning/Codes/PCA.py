import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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

def apply_pca(X, n_components=None):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

def save_selected_features(selected_features, save_path, filename):
    selected_features.to_csv(os.path.join(save_path, filename), index=False)

def print_pca_components(pca, feature_names):
    for i, component in enumerate(pca.components_):
        print(f'{pca.explained_variance_ratio_}')
        print(f"Principal Component {i} {pca.explained_variance_ratio_[i]} :")
        component_features = sorted(zip(np.abs(component), feature_names), reverse=True)
        for value, name in component_features:
            print(f"{name}: {value:.4f}")
        print("\n")

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

    save_path = os.path.join(config.MAIN_SAVE_DIR, config.DATA_TYPE, f'PCA_{config.PCA_component}')
    os.makedirs(save_path, exist_ok=True)

    X_train_full = handle_missing_values(X_train_full)

    # Standardize features if required
    if config.Standard:
        X_train_full, scaler = standardize_features(X_train_full)
        joblib.dump(scaler, os.path.join(save_path, f'scaler_fold{fold}.pkl'))

    # Apply PCA
    X_pca, pca = apply_pca(X_train_full, n_components=config.PCA_component)
    pca_components = pd.DataFrame(pca.components_, columns=feature_names)
    print(f"PCA Components for fold {fold}: \n{pca_components}")

    # Identify top contributing features to the first principal component
    top_features_indices = np.argsort(np.abs(pca.components_[0]))[::-1]
    top_features = feature_names[top_features_indices]
    selected_features = top_features[:config.MAX_FEATURES]
    print(f"Selected top features for fold {fold}: {selected_features}")

    # Print PCA components and feature contributions
    print_pca_components(pca, feature_names)

    # Save the selected features
    save_selected_features(pd.DataFrame(selected_features, columns=["Selected Features"]), save_path, f'{config.DATA_TYPE}_fold{fold}_selected_features_{config.FEATURE}_{config.MAX_FEATURES}_std{config.Standard}.csv')

def main():
    for fold in config.Fold:
        print(f"Processing fold {fold}")
        process_fold(fold)

if __name__ == "__main__":
    main()
