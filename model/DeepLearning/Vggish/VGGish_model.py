import os
import torch
from torch.utils.data import DataLoader
from AudioDataset import AudioDataset
from torchvggish import vggish
import torch.nn as nn
import torch.nn.functional as F
import easydict
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import shutil
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
"""seed and setup"""
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device found and selected: {device}')
# random seed assign
seed = 2024
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(seed)

class VGGishBinaryClassifier(nn.Module):
    def __init__(self):
        super(VGGishBinaryClassifier, self).__init__()
        self.vggish = vggish()
        self.fc = nn.Linear(128, 1)
    
    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        #print(f'Input device: {x.device}')
        x = self.vggish(x)
        #print(f'Internal VGGish output device: {x.device}')
        x = self.fc(x)
        return torch.sigmoid(x)

    def to(self, device):
        super().to(device)
        self.vggish.pproc._pca_matrix = self.vggish.pproc._pca_matrix.to(device)
        self.vggish.pproc._pca_means = self.vggish.pproc._pca_means.to(device)
        return self

'''saving models'''
def save_ckp(state, is_best, checkpoint_path, best_model_path):
    f_path = checkpoint_path
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(f_path, best_fpath)
## created for loading model
def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_loss_min = checkpoint['valid_loss_min']
    return model, optimizer, checkpoint['epoch'],valid_loss_min

'''*********************************************************************'''

'''plot ACC figure'''
def plot_acc(result_save_path,epoch_train_acc, epoch_valid_acc):
    
    pic_save_path = os.path.join(result_save_path, f"acc.png")
    file_save_path = os.path.join(result_save_path,f"acc.csv")
    '''saving loss into csv file '''
    final_loss = pd.DataFrame({'train_acc':epoch_train_acc, 'valid_acc':epoch_valid_acc})
    final_loss.to_csv(file_save_path)

    '''plot the picture'''
    epochs_list = list(range(1, len(epoch_train_acc) + 1))
    plt.clf()
    plt.plot(epochs_list, epoch_train_acc, color='blue', label='train_acc')
    plt.plot(epochs_list, epoch_valid_acc, color='red', label='valid_acc')
    plt.ylabel(' Accuracy')
    plt.xlabel('Epochs')
    plt.text(0.8, 1.1, 'Train_acc', color='blue', transform=plt.gca().transAxes)
    plt.text(0.8, 1.05, 'Valid_acc', color='red', transform=plt.gca().transAxes)
    plt.savefig(pic_save_path)
'''*********************************************************************'''


'''plot loss figure'''
def plot_loss(result_save_path,epoch_train_loss, epoch_valid_loss):
    
    pic_save_path = os.path.join(result_save_path, f"loss.png")
    file_save_path = os.path.join(result_save_path,f"loss.csv")
    '''saving loss into csv file '''
    final_loss = pd.DataFrame({'train_loss':epoch_train_loss, 'valid_loss':epoch_valid_loss})
    final_loss.to_csv(file_save_path)

    '''plot the picture'''
    epochs_list = list(range(1, len(epoch_train_loss) + 1))
    plt.clf()
    plt.plot(epochs_list, epoch_train_loss, color='blue', label='train_loss')
    plt.plot(epochs_list, epoch_valid_loss, color='red', label='valid_loss')
    plt.ylabel(' Loss')
    plt.xlabel('Epochs')
    plt.text(0.8, 1.1, 'Train_loss', color='blue', transform=plt.gca().transAxes)
    plt.text(0.8, 1.05, 'Valid_loss', color='red', transform=plt.gca().transAxes)
    plt.savefig(pic_save_path)
'''*********************************************************************'''

def model_training(model_args, model, train_loader, val_loader, optimizer, loss_fn, save_path,scheduler):
    valid_loss_min = np.Inf
    model.to(model_args.device)
    epochs = range(model_args.max_iter)
    epoch_train_loss = []
    epoch_valid_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []

    for epoch in epochs:
        model.train()
        train_loss = 0.0
        train_correct_count = 0
        train_total_count = 0
        
        train_iterator = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{model_args.max_iter}")
        for i, batch_data in train_iterator:
            optimizer.zero_grad()
            _path,features, sample_rate, label = batch_data
            #print(f'features shape is {features.shape}')
            #features = features.to(model_args.device).squeeze(1)  # Remove the extra dimension
            # features = features.permute(0, 1, 3, 2)
            # features = F.interpolate(features, size=(96, 64))
            features = features.to(model_args.device)
            label = label.to(model_args.device).float().view(-1, 1)
            
            # print(f'train features shape: {features.shape}')
            # print(f'train label shape: {label.shape}')
            
            output = model(features)
            # print(f'label is {label}')
            # print(f'output is {output}')
            #print(f'train output shape: {output.shape}')
            loss = loss_fn(output, label)
            
            correct_predictions = (output > 0.5).float() == label.view_as(output)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            train_correct_count += correct_predictions.sum().item()
            train_total_count += label.size(0)
            train_iterator.set_postfix({"train_loss": float(loss)})
        
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct_count / train_total_count
        epoch_train_loss.append(train_loss)
        epoch_train_acc.append(train_acc)
        
        print(f"Epoch {epoch+1}/{model_args.max_iter} - Train loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        model.eval()
        eval_loss = 0.0
        valid_correct_count = 0
        valid_total_count = 0
        valid_iterator = tqdm(enumerate(val_loader), total=len(val_loader), desc="valid")
        with torch.no_grad():
            for i, batch_data in valid_iterator:
                _path,features, sample_rate, label = batch_data
                # features = features.permute(0, 1, 3, 2)
                # features = F.interpolate(features, size=(96, 64))
                features = features.to(model_args.device)  # Remove the extra dimension
                label = label.to(model_args.device).float().view(-1, 1)
                
                # print(f'valid features shape: {features.shape}')
                # print(f'valid label shape: {label.shape}')
                
                output = model(features)
                #print(f'valid output shape: {output.shape}')
                loss = loss_fn(output, label)
                correct_predictions = (output > 0.5).float() == label.view_as(output)
                eval_loss += loss.item()
                valid_correct_count += correct_predictions.sum().item()
                valid_total_count += label.size(0)
                valid_iterator.set_postfix({"eval_loss": float(loss)})
            
            eval_loss = eval_loss / len(val_loader)
            valid_acc = valid_correct_count / valid_total_count
            epoch_valid_loss.append(eval_loss)
            epoch_valid_acc.append(valid_acc)
            scheduler.step(eval_loss)
            
            print(f"Evaluation Loss: {eval_loss:.4f}")
            print(f"Evaluation Acc: {valid_acc:.4f}")
        '''***********************'''
        '''saveing checkpoint and best model'''
        checkpoint = {
          'epoch': epoch,
          'valid_loss_min': eval_loss,
          'state_dict': model.state_dict(),
          'optimizer': optimizer.state_dict(),
        }
        checkpoint_path = os.path.join(save_path,'checkpoint')
        os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_path,'last-checkpoint.pth')
        bestmodel_path = os.path.join(save_path,'bestmodel')
        os.makedirs(bestmodel_path, exist_ok=True)
        bestmodel_path =bestmodel_path + f"/epoch-{str(epoch).zfill(3)}" + '-loss-' +  str("{:.2f}".format(eval_loss)) + '.pth'
        save_ckp(checkpoint,False,checkpoint_path,bestmodel_path)
        if eval_loss < valid_loss_min:
            print('Validation loss decreased ({:.6f} ---> {:.6f}). Saving model...'.format(valid_loss_min, eval_loss))
            save_ckp(checkpoint,True, checkpoint_path, bestmodel_path)
            valid_loss_min = eval_loss
        plot_loss(save_path,epoch_train_loss, epoch_valid_loss)
        plot_acc(save_path,epoch_train_acc, epoch_valid_acc)
'''Testing process'''
def model_testing(model_args, model, test_loader, optimizer, save_path, model_path):
    model, optimizer, last_epoch, valid_loss_min = load_ckp(model_path, model, optimizer)
    print("optimizer = ", optimizer)
    print("last_epoch = ", last_epoch)
    print("valid_loss_min = ", valid_loss_min)
    
    file_save_path = os.path.join(save_path, 'prediction.txt')
    print(f"file_save_path is {file_save_path}")
    result_save_file = open(file_save_path, "w+")
    
    all_preds = []
    all_labels = []
    
    test_iterator = tqdm(enumerate(test_loader), total=len(test_loader), desc="test")
    with torch.no_grad():
        for i, batch_data in test_iterator:
            _path, features, _sample_rate, label = batch_data
            # features = features.permute(0, 1, 3, 2)
            # features = F.interpolate(features, size=(96, 64))
            features = features.to(model_args.device)  # Remove the extra dimension
            label = label.to(model_args.device).float().view(-1, 1)
            output = model(features)
            path = _path[0]
            
            prediction = (output > 0.5).float().item()
            result_save_file.write("{},{},{}\n".format(path, label.item(),prediction))
            
            all_preds.append(prediction)
            all_labels.append(label.item())
    
    result_save_file.close()
    
    # Calculate accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc:.4f}")
    
    # Save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_save_path = os.path.join(save_path, 'confusion_matrix.png')
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Accuracy: {acc:.4f}')
    plt.savefig(cm_save_path)
    plt.close()
'''*********************************************************************'''

def main():
    main_data_dir = '/data/jiayiwang/summerschool/models/Data_preprocessing/'
    data_type = 'FIMO'
    annotations_dir = os.path.join(main_data_dir, data_type)
    main_save_dir = '/data/jiayiwang/summerschool/models/saved/VGGish'
    target_sample_rate = 44100
    feature_type = 'mel_spectrogram'
    save_path = os.path.join(main_save_dir, data_type,feature_type)
    os.makedirs(save_path, exist_ok=True)

    test_annotations_file = os.path.join(annotations_dir, 'test.txt')
    val_annotations_file = os.path.join(annotations_dir, 'val.txt')
    train_annotations_file = os.path.join(annotations_dir, 'train.txt')

    Train = False

    batch_size = 16
    batch_size_valid = 16
    batch_size_test = 1

   

    if Train:
        train_dataset = AudioDataset(annotations_file=train_annotations_file, target_sample_rate=target_sample_rate, feature_type=feature_type)
        val_dataset = AudioDataset(annotations_file=val_annotations_file, target_sample_rate=target_sample_rate, feature_type=feature_type)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size_valid, shuffle=False)
    else:
        test_dataset = AudioDataset(annotations_file=test_annotations_file, target_sample_rate=target_sample_rate, feature_type=feature_type)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

    model_args = easydict.EasyDict({
        "batch_size": batch_size,
        "learning_rate": 1e-5,
        "max_iter": 30,
        "device": device,
    })

    model = VGGishBinaryClassifier()
    model.to(device)

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 10, verbose = True)
    if Train:
        model_save_path = os.path.join(save_path)
        print(f'model_save_path is {model_save_path}')
        os.makedirs(model_save_path, exist_ok=True)
        model_training(model_args, model, train_loader, val_loader, optimizer, loss_fn, model_save_path,scheduler)
    else:
        model_path = os.path.join(save_path,'bestmodel' ,'epoch-006-loss-8.94.pth')
        model_testing(model_args,model,test_loader,optimizer,save_path,model_path)


if __name__ == "__main__":
    main()
