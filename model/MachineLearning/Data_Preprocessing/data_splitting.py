# import os
# import pandas as pd
# import numpy as np

# def stratified_split_by_recordings(data, train_size, test_size, random_state=42):
#     assert train_size + test_size == 1, "The sum of train_size and test_size must be 1."
    
#     patients = data['patient'].unique()
#     np.random.seed(random_state)
#     np.random.shuffle(patients)

#     train_patients, test_patients = [], []
#     train_count, test_count = 0, 0
#     total_count = len(data)

#     for patient in patients:
#         patient_data = data[data['patient'] == patient]
#         patient_recordings = len(patient_data)

#         if train_count / total_count < train_size:
#             train_patients.append(patient)
#             train_count += patient_recordings
#         else:
#             test_patients.append(patient)
#             test_count += patient_recordings

#     train_data = data[data['patient'].isin(train_patients)]
#     test_data = data[data['patient'].isin(test_patients)]
    
#     return train_data, test_data, train_patients, test_patients

# def check_labels(data):
#     labels = data['label'].unique()
#     return set(labels) == {0, 1}

# def main():
#     main_path = '/data/jiayiwang/summerschool/models/Data_preprocessing'
#     data_type = 'FIMO'
#     fold = 1
#     main_file_path = os.path.join(main_path, f'{data_type}_chunk_Stridor.txt')
#     output_dir = os.path.join(main_path, data_type,f'fold{fold}')
    
#     os.makedirs(output_dir, exist_ok=True)

#     data = pd.read_csv(main_file_path, sep=' ', header=None, names=['path', 'label'])
#     data['patient'] = data['path'].apply(lambda x: x.split('/')[-2])

#     train_data, test_data, train_patients, test_patients = stratified_split_by_recordings(
#         data, train_size=0.80, test_size=0.20)

#     if check_labels(train_data) and check_labels(test_data):
#         print("Each dataset contains both labels.")
#     else:
#         print("One or more datasets are missing a label.")

#     train_data[['path', 'label']].to_csv(os.path.join(output_dir, 'train.txt'), sep=' ', header=False, index=False)
#     test_data[['path', 'label']].to_csv(os.path.join(output_dir, 'test.txt'), sep=' ', header=False, index=False)

#     print("Data split completed and saved to txt files.")

#     print("Label distribution for training set:")
#     print(train_data['label'].value_counts().sort_index())
#     print("Label distribution for test set:")
#     print(test_data['label'].value_counts().sort_index())

#     print("\nNumber of patients in training set:", len(train_patients))
#     print("Patients in training set:", train_patients)
#     print("\nNumber of patients in test set:", len(test_patients))
#     print("Patients in test set:", test_patients)

# if __name__ == "__main__":
#     main()
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

def check_labels(data):
    labels = data['label'].unique()
    return set(labels) == {0, 1}

def stratified_kfold_split(data, n_splits=5, random_state=42):
    patients = data['patient'].unique()
    np.random.seed(random_state)
    np.random.shuffle(patients)
    
    patient_labels = data.groupby('patient')['label'].agg(lambda x: x.value_counts().idxmax())
    patient_labels = patient_labels.loc[patients]
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    folds = []
    for train_idx, test_idx in skf.split(patients, patient_labels):
        train_patients, test_patients = patients[train_idx], patients[test_idx]
        
        train_data = data[data['patient'].isin(train_patients)]
        test_data = data[data['patient'].isin(test_patients)]
        
        folds.append((train_data, test_data, train_patients, test_patients))
    
    return folds

def main():
    main_path = '/data/jiayiwang/summerschool/models/Data_preprocessing'
    data_type = 'Deep'
    main_file_path = os.path.join(main_path, f'{data_type}_chunk_Stridor.txt')
    output_dir = os.path.join(main_path, data_type)
    
    os.makedirs(output_dir, exist_ok=True)

    data = pd.read_csv(main_file_path, sep=' ', header=None, names=['path', 'label'])
    data['patient'] = data['path'].apply(lambda x: x.split('/')[-2])
    
    folds = stratified_kfold_split(data, n_splits=5)
    
    for i, (train_data, test_data, train_patients, test_patients) in enumerate(folds):
        fold_dir = os.path.join(output_dir, f'fold{i+1}')
        print(fold_dir)
        os.makedirs(fold_dir, exist_ok=True)
        
        if check_labels(train_data) and check_labels(test_data):
            print(f"Each dataset in fold {i+1} contains both labels.")
        else:
            print(f"One or more datasets in fold {i+1} are missing a label.")

        train_data[['path', 'label']].to_csv(os.path.join(fold_dir, 'train.txt'), sep=' ', header=False, index=False)
        test_data[['path', 'label']].to_csv(os.path.join(fold_dir, 'test.txt'), sep=' ', header=False, index=False)

        print(f"Data split for fold {i+1} completed and saved to txt files.")

        print(f"Label distribution for training set in fold {i+1}:")
        print(train_data['label'].value_counts().sort_index())
        print(f"Label distribution for testing set in fold {i+1}:")
        print(test_data['label'].value_counts().sort_index())

        print(f"\nNumber of patients in training set for fold {i+1}:", len(train_patients))
        print(f"Patients in training set for fold {i+1}:", train_patients)
        print(f"\nNumber of patients in testing set for fold {i+1}:", len(test_patients))
        print(f"Patients in testing set for fold {i+1}:", test_patients)

if __name__ == "__main__":
    main()
