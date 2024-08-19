# In[1]
# Include / import python libraries

# Author: Siyu (Steven) Lin, Haowen Zhou

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import models, transforms

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import copy
from torch.utils.data import Dataset

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

from scipy.interpolate import griddata

from PIL import Image

import warnings
warnings.filterwarnings('ignore')


cudnn.benchmark = True
plt.ion()   # interactive mode


# In[2]
# Utility Functions
def soft(x,y):
    return 1/(1 + np.exp(y-x))

def sig(x):
    return 1/(1 + np.exp(-x))

# Return the slide numbers used for validation
# Same number of BM and C slides will be used for validation
def shuffle(slides_BM, slides_C, nfold = 1, num_val = 20, overlap = 0):
    validation_slide_split = np.zeros((nfold, num_val))
    np.random.seed(1)
    np.random.shuffle(slides_BM)
    np.random.shuffle(slides_C)
    for fold in range(nfold):
        validation_slide_split[fold][:] = np.concatenate((np.array(slides_BM[fold * (num_val // 2 - overlap): fold * (num_val // 2 - overlap) + num_val // 2]) , 
                                                          np.array(slides_C [fold * (num_val // 2 - overlap): fold * (num_val // 2 - overlap) + num_val // 2])))
    return validation_slide_split.astype('int')
        
    
    

# In[3]
# datatype = 'train' / 'val'
# file = 000000_000.png
class NSCLC_Dataset(Dataset):
    def __init__(self, datafolder, datatype, transform, validation_slides, testing_slides,downsample):
        self.datafolder = datafolder
        self.image_files_list = {}
        self.categories = ['BM', 'C']
        self.dataset_image = []
        self.count = {}
        self.downsample = downsample

        for category in self.categories:
            label = 0 if category == 'BM' else 1
            self.image_files_list[category] = [os.listdir(datafolder + s + '/') for s in self.categories]

            for img in self.image_files_list[category][label]:
                row_idx = int(np.floor( float(int(img[:-4])-1) / tile_per_slide))
                slide = row_idx + 1
                if datatype == 'train' and slide not in validation_slides and slide not in testing_slides:
                    if not slide in self.count:
                        self.count[slide] = 1
                    else:
                        self.count[slide] += 1
                    self.dataset_image.append((img, label, slide))
                elif datatype == 'val' and slide in validation_slides:
                    if not slide in self.count:
                        self.count[slide] = 1
                    else:
                        self.count[slide] += 1
                    self.dataset_image.append((img, label, slide))
        self.transform = transform
        

    def __len__(self):
        return len(self.dataset_image)

    def __getitem__(self, idx):
        if self.dataset_image[idx][1] == 0:
            img_name = os.path.join(self.datafolder + 'BM/',
                                    self.dataset_image[idx][0])
        elif self.dataset_image[idx][1] == 1:
            img_name = os.path.join(self.datafolder + 'C/',
                                    self.dataset_image[idx][0])
               
        image = Image.open(img_name)
        image = np.array(image)
        if isinstance(self.downsample, int):
            image = image[::self.downsample,::self.downsample,:]
            image = Image.fromarray(np.uint8(image))
        else:
            # print(image)
            (img_x, img_y, channels) = np.shape(image) 
            t = transforms.Resize(int((img_x / self.downsample)))
            image = Image.fromarray(np.uint8(image))
            image = t(image)
            
        image = self.transform(image)
        # Same for the labels files
        return image, self.dataset_image[idx][1], self.dataset_image[idx][2]
    
    
# In[]
if __name__ == '__main__':
    
    # Hyper Parameters and Paths
    Cpath = 'RootPath'
    root_name = 'DataFolder' 
    model_abbr = 'Resenet18_'
    magnif = '20'
    lr = 1e-3
    offset = 0
    num_train_test_splits = 3
    nfold = 3 # 5
    tile_per_slide = 1000
    num_test = 40 # 20
    num_val = 30
    
    batch_size = 200 ###### Originally: 200
    test_batch_size = 1
    
    momentum = 0.9
    num_epochs = 30
    weight_decay = 0.1
    
    
    '''
    Change This
    '''
    downsampling_ratios = [40,50]
    '''
    Change This
    '''
    for downsample in downsampling_ratios:
        tile_size = int(256/downsample)

        data_dir = os.path.join(Cpath, root_name)
        indexpath = os.path.join(data_dir, 'Index') 
        save_dir = os.path.join(data_dir, 'test_results_train_test_splits_' + str(tile_size))
        
        # Storing the final tile and slide level accuracies on each fold of cross validation
        nfold_val_tile_acc =  []
        nfold_val_slide_acc = []
        nfold_val_tile_auc =  []
        nfold_val_slide_auc = []
        
        
        # Directory to store testing results
        os.makedirs(save_dir, exist_ok = True)
        
        
        # Get the slide numbers for BM and C
        # Combine the train and testing index files
        
        
        iminfo_train = pd.read_csv(os.path.join(indexpath, 'iminfo_train.csv'))
        iminfo_test = pd.read_csv(os.path.join(indexpath, 'iminfo_test.csv'))
        iminfo_list = pd.concat([iminfo_train, iminfo_test], ignore_index=True)
        iminfo_list = iminfo_list.sort_values(by = ['Slide', 'Index'])
        
        slides_BM = []
        slides_C  = []
        for idx in range(len(iminfo_list) // tile_per_slide):
            if iminfo_list['Class'][idx * tile_per_slide] == 0: 
                slides_BM.append(iminfo_list['Slide'][idx * tile_per_slide])
            elif iminfo_list['Class'][idx * tile_per_slide] == 1:
                slides_C.append(iminfo_list['Slide'][idx * tile_per_slide])
            else:
                print('Error')
        slides_BM = np.array(slides_BM)
        slides_C  = np.array(slides_C)
        
        # In[]
        # Train Test Splits
        validation_slide_splits = shuffle(slides_BM, slides_C, num_val = num_test, nfold = num_train_test_splits, overlap = 0)
        
        # np.save(os.path.join(indexpath,'nfold_splits_nfold_'+ str(nfold) + '_offset_' + str(offset) + '.npy'), validation_slide_splits)

        # In[]
        for train_test_fold in range(3):
        # train_test_fold = 2 # Change  this!
        
            # Remove the testing slides
            cv_slides_BM = []
            cv_slides_C  = []
            for slide in slides_BM:
                if slide not in validation_slide_splits[train_test_fold]:
                    cv_slides_BM.append(slide)
                    
            for slide in slides_C:
                if slide not in validation_slide_splits[train_test_fold]:
                    cv_slides_C.append(slide) 
                    
            cv_slides_BM = np.array(cv_slides_BM)
            cv_slides_C  = np.array(cv_slides_C)
            
            # Generate CV splits
            cv_slide_splits = shuffle(cv_slides_BM, cv_slides_C, num_val = num_val, nfold = nfold, overlap = 0)
            
            
            
            
            
            # In[]
            for fold in range(nfold): # nfold
            
                iminfo_train_cv = iminfo_list.drop(np.arange(0, len(iminfo_list), 1))
                iminfo_val_cv = iminfo_list.drop(np.arange(0, len(iminfo_list), 1))
                
                slide_num = np.zeros(len(iminfo_list)//tile_per_slide)
                for row_idx in range(len(slide_num)):
                    slide_n = int(iminfo_list['Index'][row_idx*tile_per_slide][7:])
                    slide_num[row_idx] = slide_n
                    if slide_n in cv_slide_splits[fold]: 
                        iminfo_val_cv = pd.concat([ iminfo_val_cv,iminfo_list.loc[row_idx*tile_per_slide : (row_idx+1)*tile_per_slide - 1] ])
                    elif slide_n not in cv_slide_splits[fold] and slide_n not in validation_slide_splits[train_test_fold]: 
                        iminfo_train_cv = pd.concat([ iminfo_train_cv,iminfo_list.loc[row_idx*tile_per_slide : (row_idx+1)*tile_per_slide - 1] ])
                    # else:
                    #     print('Error: Slide split issue!')
                        
                slide_num = slide_num.astype('int')
                        
                # In[]
                mean_r = np.mean(iminfo_train_cv['mean_r'])
                mean_g = np.mean(iminfo_train_cv['mean_g'])
                mean_b = np.mean(iminfo_train_cv['mean_b'])
                std_r = np.sqrt(np.mean(iminfo_train_cv['var_r']))
                std_g = np.sqrt(np.mean(iminfo_train_cv['var_g']))
                std_b = np.sqrt(np.mean(iminfo_train_cv['var_b']))
                
                print(mean_r,mean_g,mean_b,std_r,std_g,std_b)
                
                
                # In[5]
                # Data augmentation and normalization for training
                # Just normalize for validation
                data_transforms = {
                    'train': transforms.Compose([
                                            # transforms.Resize(256),
                                            # transforms.RandomCrop(tile_size), 
                                            transforms.Resize(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.RandomRotation(90),
                                            transforms.ToTensor(),
                                            transforms.Normalize([mean_r,mean_g,mean_b], [std_r,std_g,std_b]) 
                                            ]),
                    'val': transforms.Compose([
                                            # transforms.Resize(256),
                                            # transforms.CenterCrop(tile_size),
                                            transforms.Resize(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([mean_r,mean_g,mean_b], [std_r,std_g,std_b])
                                            ]),
                    }
                
                # In[] 
                # Dataset Loader
                image_datasets = {x: NSCLC_Dataset(data_dir + '/train/', x, data_transforms[x], cv_slide_splits[fold], validation_slide_splits[train_test_fold],downsample) for x in ['train', 'val']}
                dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=2) for x in ['train', 'val'] }
                dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
                   
                device = torch.device("cuda:0")
                
                print(device)
                print(dataset_sizes)
                   
                # In[6]
                # Set model
                save_name = root_name + '_lr_' + str(lr) + '_model_' + model_abbr + 'fold_' + str(train_test_fold) +'_' + str(int(fold + offset))
                
                   
                
                if model_abbr == 'Resenet18_': 
                   model_ft = models.resnet18(pretrained=True)
                   num_ftrs = model_ft.fc.in_features
                   model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 2),
                               )   # Add a softmax layer nn.Softmax(dim=1)
                   # print(model_ft)
                   
                elif model_abbr == 'Wideresnet50_': 
                   model_ft = models.wide_resnet50_2(pretrained=True)
                   num_ftrs = model_ft.fc.in_features
                   model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 2),
                               )   # Add a softmax layer nn.Softmax(dim=1)
                   # print(model_ft)
                   
                elif model_abbr == 'Densenet121_': 
                   model_ft = models.densenet121(pretrained=True)
                   # num_ftrs = model_ft.fc.in_features
                   model_ft.classifier = nn.Sequential(nn.Linear(1024, 1000),
                                       nn.Sequential(nn.Linear(1000, 2)))
                          
                   # print(model_ft)
                   
                elif model_abbr == 'Efficientnetv2_': 
                   model_ft = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)
                   model_ft.classifier = model_ft.classifier.append(nn.Linear(model_ft.classifier[3].out_features, 2))
                   # print(model_ft)
                
                
                model_ft = model_ft.to(device)
                
                criterion = nn.CrossEntropyLoss()
                
                # Observe that all parameters are being optimized
                # optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=momentum)
                optimizer_ft = optim.Adam(model_ft.parameters(), lr=lr, weight_decay=weight_decay)
                
                # Decay LR by a factor of 0.1 every xx epochs 
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=200, gamma=0.1)
                
                # Number of parameters
                num_params = sum(param.numel() for param in model_ft.parameters())
                print('  ')
                print('Magnification: ' + magnif + ' | Model: ' + model_abbr + ' | Learning Rate: ' + str(lr))
                print('Number of parameters: ',num_params)
                print('Total Number of Epochs : ', num_epochs)
                   
                   # Model structure
                   
                   # print(model_ft)
                   
                   # In[8]
                   # Training starts here !!!!!
                   
                since = time.time()
                
                best_model_wts = copy.deepcopy(model_ft.state_dict())
                best_acc = 0.0
                best_epoch = 0
                train_loss =  []
                val_loss = []
                train_acc = []
                val_acc = []
                for epoch in range(num_epochs):
                    t = time.time()
                    print('-' * 40)
                    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                    
                    
                    # Training Phase
                    model_ft.train()
                    
                            
                    running_loss = 0.0
                    running_corrects = 0
                 
                    # Iterate over data.
                    for inputs, labels, _ in dataloaders['train']:
                        # plt.figure
                        # I = inputs.detach().numpy()
                        # I = np.squeeze(I[0,:,:,:])
                        # I = np.moveaxis(I, 0, -1)
                        # I[:,:,0] = I[:,:,0]*std_r + mean_r
                        # I[:,:,1] = I[:,:,1]*std_g + mean_g
                        # I[:,:,2] = I[:,:,2]*std_b + mean_b
                        # plt.imshow(I)
                        # plt.show()
                        
                        
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        # zero the parameter gradients
                        optimizer_ft.zero_grad()
                        # forward pass
                        # track history if only in train
                        with torch.set_grad_enabled(True):
                            outputs = model_ft(inputs)
                            preds_score, preds_class = torch.max(outputs,1)
                            loss = criterion(outputs, labels)
                         
                            # backward + optimize only if in training phase
                            loss.backward()
                            optimizer_ft.step()
                   
                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds_class == labels.data)
                    
                        exp_lr_scheduler.step()
                    
                    epoch_loss = (running_loss / dataset_sizes['train'])
                    epoch_acc = (running_corrects / dataset_sizes['train']).cpu()
                         
                 
                    train_loss.append(epoch_loss)
                    train_acc.append(epoch_acc)
                 
                 
                    
                    elapsed = time.time() - t
                    print('Training Time per epoch: ',elapsed)
                    print('{} Loss: {:.4f} Acc: {:.4f} '.format(
                        'Train', epoch_loss, epoch_acc))
                    
                
                    
                    # Validation Phase
                    t = time.time()
                    model_ft.eval()
                    
                    running_loss = 0.0
                    running_corrects = 0
                    
                    for inputs, labels, _ in dataloaders['val']:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
        
                        outputs = model_ft(inputs)
                        
                        preds_score, preds_class = torch.max(outputs,1)
                        loss = criterion(outputs, labels)
                          
                    
                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds_class == labels.data)
                                 
                    epoch_loss = (running_loss / dataset_sizes['val'])
                    epoch_acc = (running_corrects / dataset_sizes['val']).cpu()
                         
                 
                    val_loss.append(epoch_loss)
                    val_acc.append(epoch_acc)
                 
                    
                    
                    elapsed = time.time() - t
                    print('Validation Time per epoch: ',elapsed)
                    print('{} Loss: {:.4f} Acc: {:.4f} '.format(
                        'Validation', epoch_loss, epoch_acc))
                    
                    # Save the model with the best validation performance
                    if epoch_acc > best_acc:
                        best_epoch = epoch
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model_ft.state_dict())
                        
          
                time_elapsed = time.time() - since
                print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
               
                # load best model weights
               
                model_ft.load_state_dict(best_model_wts)
                # In[9]
                # Save model
                  
                torch.save(model_ft, os.path.join(save_dir, save_name + 
                                                   '_whole_model.pt'))
                 
                np.save(os.path.join(save_dir,'Train_loss_' + save_name + '.npy'), np.array(train_loss))
                np.save(os.path.join(save_dir,'Train_acc_' + save_name + '.npy'), np.array(train_acc))
                np.save(os.path.join(save_dir,'Val_loss_' + save_name + '.npy'), np.array(val_loss))
                np.save(os.path.join(save_dir,'Val_acc_' + save_name + '.npy'), np.array(val_acc))
                np.save(os.path.join(save_dir,'Best_epoch_' + save_name + '.npy'), np.array(best_epoch))
                
                
                # In[]
                # Get the slide-level accuracy based on the best model
                slide_accuracies = {}
                model_ft.eval()
                running_corrects = 0
                
                tile_y_true = np.array([])
                tile_y_pred = np.array([])
                
                for inputs, labels, slides in dataloaders['val']:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
        
                    outputs = model_ft(inputs)
                    preds_score, preds_class = torch.max(outputs,1)
                    loss = criterion(outputs, labels)
                    
                    tile_scores = soft(outputs[:,0].detach().cpu().numpy(),outputs[:,1].detach().cpu().numpy())
                    slide_nums = slides.detach().cpu().numpy()
                    
                    tile_y_pred = np.append(tile_y_pred, preds_class.detach().cpu().numpy())
                    tile_y_true = np.append(tile_y_true, labels.detach().cpu().numpy())
                    
                    
                    for i in range(np.size(tile_scores)):
                        slide = slide_nums[i]
                        if slide not in slide_accuracies:
                            slide_accuracies[slide] = []
                        slide_accuracies[slide].append(tile_scores[i])
                    
                      
                
                    # statistics
                    running_corrects += torch.sum(preds_class == labels.data)
                 
                overall_val_accuracy = (running_corrects / dataset_sizes['val']).cpu().numpy()
                tile_auc_score = roc_auc_score(tile_y_true, tile_y_pred)
                nfold_val_tile_auc.append(tile_auc_score)
                # In[]
                fpr, tpr, thres = roc_curve(tile_y_true, tile_y_pred)
                plt.plot(fpr,tpr)
                plt.title('Fold ' + str(fold) + ' Tile-Level ROC Curve')
                plt.ylabel('True Positive Rate')
                plt.xlabel('False Positive Rate')
                plt.savefig(os.path.join(save_dir, 'Fold ' + str(fold) + ' Tile-Level ROC Curve.png'))
                plt.show()
                print('Fold ' + str(fold) + ' Tile-Level AUC Score ' + str(tile_auc_score))
                
                # In[]
                count_slide_pred_correct = 0
                slide_y_true = []
                slide_y_pred = []
                
                
                df = pd.DataFrame.from_dict(slide_accuracies)
                df.to_csv(os.path.join(save_dir, save_name + '_tile_accuracies.csv'), index=True)
                
                
                for slide, tile_preds in slide_accuracies.items():
                    slide_pred = np.median(np.array(tile_preds))
                    # print(slide_pred)
                    slide_pred = np.sign(slide_pred - 0.5)
                    gt = 0
                    if slide in slides_BM:
                        gt = 1
                    elif slide in slides_C:
                        gt = -1
                    # print('Prediction: ', slide_pred, 'GT: ', gt)
                    if slide_pred == gt:
                        count_slide_pred_correct += 1
                    slide_y_true.append(gt)
                    slide_y_pred.append(slide_pred)
                        
                slide_y_true = np.array(slide_y_true)
                slide_y_pred = np.array(slide_y_pred)
                slide_auc_score = roc_auc_score(slide_y_true, slide_y_pred)
                fpr, tpr, thres = roc_curve(slide_y_true, slide_y_pred)
                plt.plot(fpr,tpr)
                plt.title('Fold ' + str(fold) + ' Slide-Level ROC Curve')
                plt.ylabel('True Positive Rate')
                plt.xlabel('False Positive Rate')
                plt.savefig(os.path.join(save_dir, 'Fold ' + str(fold) + ' Slide-Level ROC Curve.png'))
                plt.show()
                print('Fold ' + str(fold) + ' Slide-Level AUC Score ' + str(slide_auc_score))
                # In[]
                        
                nfold_val_slide_acc.append(count_slide_pred_correct / len(slide_accuracies))
                nfold_val_tile_acc.append(overall_val_accuracy)
                nfold_val_slide_auc.append(slide_auc_score)
                print('Overall slide level accuracy: ', count_slide_pred_correct / len(slide_accuracies))
                print('Overall tile level accuracy: ', overall_val_accuracy)
            df = pd.DataFrame({'Tile Accuracy':  nfold_val_tile_acc,
                               'Slide Accuracy': nfold_val_slide_acc,
                               'Tile AUC':  nfold_val_tile_auc,
                               'Slide AUC': nfold_val_slide_auc
                               })
            df.to_csv(os.path.join(save_dir, 'nfold_validation_accuracy.csv'), index=True)
                        
                
        
        
