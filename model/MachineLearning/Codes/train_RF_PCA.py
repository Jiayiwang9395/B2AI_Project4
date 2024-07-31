import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import config as config
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

def load_selected_features(csv_file_path, num_features):
    df = pd.read_csv(csv_file_path)
    selected_features = df['Selected Features'].values[:num_features]  # Select the top N features
    return selected_features

def load_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    paths = df.iloc[:, 0].values  # First column as paths
    X = df.iloc[:, 1:-1].values  # All columns except the first (path) and the last one (label)
    y = df.iloc[:, -1].values   # The last column
    return paths, X, y, df.columns[1:-1]

def load_annotations(annotation_file):
    df = pd.read_csv(annotation_file, sep=' ', header=None, names=['path', 'label'])
    return df['path'].values, df['label'].values

def standardize_features(X, scaler):
    X_standardized = scaler.transform(X)
    return X_standardized

def plot_confusion_matrix(cm, classes, save_path, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.close()

def write_test_results(y_test, y_pred, paths, save_path):
    df = pd.DataFrame({
        'recording_path': paths,
        'Prediction': y_pred,
        'Actual': y_test
    })
    results_path = os.path.join(save_path, f'test_results.txt')
    df.to_csv(results_path, index=False, sep='\t')
    print(f'Test results written to {results_path}')

def main():
    random_state = 42  # Set a fixed random state for reproducibility

    main_path = os.path.join(config.MAIN_SAVE_DIR, config.DATA_TYPE, f'PCA_{config.PCA_component}')

    summary_results = {}

    for fold in config.Fold:
        fold_results = {}
        for num_features in config.NUMBER_FEATURES:
            print(f'************************ Fold {fold} - Features {num_features} *****************************************')
            save_path = os.path.join(config.MAIN_SAVE_DIR, config.DATA_TYPE, f'PCA_{config.PCA_component}', f'fold{fold}', f'features_{num_features}')
            os.makedirs(save_path, exist_ok=True)

            # Load selected features
            selected_features_path = os.path.join(main_path, f'{config.DATA_TYPE}_fold{fold}_selected_features_{config.FEATURE}_{config.MAX_FEATURES}_std{config.Standard}.csv')
            selected_features_names = load_selected_features(selected_features_path, num_features)

            # Load the combined data
            combine_csv_file_path = os.path.join(config.MAIN_SAVE_DIR, config.DATA_TYPE, f'{config.DATA_TYPE}_combined_features{config.WAYS}.csv')
            combined_paths, X_combined, y_combined, feature_names = load_data(combine_csv_file_path)

            annotations_dir = os.path.join(config.MAIN_DATA_DIR, config.DATA_TYPE, f'fold{fold}')
            train_annotations_file = os.path.join(annotations_dir, 'train.txt')
            test_annotations_file = os.path.join(annotations_dir, 'test.txt')

            train_paths, y_train = load_annotations(train_annotations_file)
            test_paths, y_test = load_annotations(test_annotations_file)

            # Find the indices of the training and test paths in the combined data
            train_indices = np.where(np.isin(combined_paths, train_paths))[0]
            test_indices = np.where(np.isin(combined_paths, test_paths))[0]

            # Extract the corresponding features for training and testing
            X_train_full = X_combined[train_indices]
            train_path_full = combined_paths[train_indices]
            y_train = y_combined[train_indices]  # Ensure y_train corresponds to train_indices
            X_test_full = X_combined[test_indices]
            test_path_full = combined_paths[test_indices]
            y_test = y_combined[test_indices]  # Ensure y_test corresponds to test_indices

            # Convert selected feature names to indices
            selected_features_indices = [list(feature_names).index(feature) for feature in selected_features_names]

            # Select the top features from the datasets
            X_train_selected = X_train_full[:, selected_features_indices]
            X_test_selected = X_test_full[:, selected_features_indices]

            if config.OVERSAMPLING == 'SMOTE':
                smote = SMOTE(random_state=random_state)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)
            else:
                X_train_resampled, y_train_resampled = X_train_selected, y_train

            if config.Standard:
                # Standardize the datasets using the previously saved scaler
                scaler_path = os.path.join(main_path, f'scaler_fold{fold}.pkl')
                scaler = joblib.load(scaler_path)
                X_train_standardized = standardize_features(X_train_resampled, scaler)
                X_test_standardized = standardize_features(X_test_selected, scaler)
            else:
                X_train_standardized = X_train_resampled
                X_test_standardized = X_test_selected

            if config.GridSearch:
                grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=random_state), param_grid=config.PARAM_GRID, cv=3, n_jobs=-1, verbose=2)
                # Fit the grid search to the resampled data
                grid_search.fit(X_train_standardized, y_train_resampled)

                # Print the best parameters
                print(f"Best parameters: {grid_search.best_params_}")

                # Use the best model for predictions
                best_rf = grid_search.best_estimator_
                y_pred = best_rf.predict(X_test_standardized)
            else:
                # Train RandomForest model using the selected features
                rf_classifier = RandomForestClassifier(random_state=random_state, n_estimators=200, max_features='sqrt', max_depth=20, min_samples_split=5, min_samples_leaf=2, bootstrap=True)
                rf_classifier.fit(X_train_standardized, y_train_resampled)

                # Use the trained model for predictions
                y_pred = rf_classifier.predict(X_test_standardized)

            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            auc = roc_auc_score(y_test, rf_classifier.predict_proba(X_test_standardized)[:, 1]) if len(np.unique(y_test)) == 2 else 'N/A'  # AUC is only meaningful for binary classification
            report = classification_report(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            print(f'Test Accuracy: {accuracy}')

            # Save the best model
            joblib.dump(rf_classifier, os.path.join(save_path, f'rf_model_{num_features}_features_{config.OVERSAMPLING}_GRIDSEARCH{config.GridSearch}.pkl'))

            # Save the classification report
            with open(os.path.join(save_path, f'classification_report_{num_features}_features_{config.OVERSAMPLING}_GRIDSEARCH{config.GridSearch}.txt'), 'w') as f:
                f.write(f'Test Accuracy: {accuracy}\n')
                f.write(f'F1 Score: {f1}\n')
                f.write(f'Precision: {precision}\n')
                f.write(f'Recall: {recall}\n')
                f.write(f'AUC: {auc}\n\n')
                f.write(report)

            # Write the test results to a file
            write_test_results(y_test, y_pred, test_path_full, save_path)

            # Save the confusion matrix plot
            plot_confusion_matrix(cm, classes=[0, 1], save_path=os.path.join(save_path, f'confusion_matrix_{num_features}_features_{config.OVERSAMPLING}_GRIDSEARCH{config.GridSearch}.png'))

            # Feature importance analysis
            feature_importances = rf_classifier.feature_importances_
            sorted_idx = np.argsort(feature_importances)[::-1]
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(feature_importances)), feature_importances[sorted_idx], align='center')
            plt.xticks(range(len(feature_importances)), np.array(selected_features_names)[sorted_idx], rotation=90)
            plt.xlabel('Feature Importance')
            plt.ylabel('Feature')
            plt.title(f'Feature Importance Analysis - {num_features} Features')
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'feature_importance_{num_features}_features_{config.OVERSAMPLING}_GRIDSEARCH{config.GridSearch}.png'))
            plt.close()

            # Collect results
            fold_results[num_features] = {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'auc': auc
            }

        summary_results[fold] = fold_results

    # Generate summary table
    summary_table = pd.DataFrame(columns=['Fold'] + config.NUMBER_FEATURES)

    for fold in config.Fold:
        fold_data = summary_results[fold]
        row = {'Fold': fold}
        for num_features in config.NUMBER_FEATURES:
            row[num_features] = fold_data[num_features]['accuracy']
        summary_table = summary_table.append(row, ignore_index=True)

    # Save summary table
    summary_save_path = os.path.join(config.MAIN_SAVE_DIR, config.DATA_TYPE, 'summary_results.csv')
    summary_table.to_csv(summary_save_path, index=False)
    print(f'Summary results saved to {summary_save_path}')

if __name__ == "__main__":
    main()
