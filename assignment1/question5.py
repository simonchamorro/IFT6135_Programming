import random
import sys
import os
import shutil
import warnings
import numpy as np
import h5py
import re
import seaborn as sns

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

import solution


def plot_roc(true, pred, title=''):
    tpr_list = []
    fpr_list = []

    # Compute fpr and tpr for each k value
    k_list = np.linspace(0, 1, num=200, endpoint=False)
    for k in k_list:
        y_pred = pred > k
        results = solution.compute_fpr_tpr(true, y_pred.astype(int))
        fpr_list.append(results['fpr'])
        tpr_list.append(results['tpr'])

    plt.figure()
    plt.plot(fpr_list, tpr_list)
    plt.title(title)
    plt.xlabel('False Positive Rate', fontsize=10)
    plt.ylabel('True Positive Rate', fontsize=10)
    plt.savefig('roc.png')
    plt.show()


def auc_roc(model, loader, device, title=''):
    # Compute AUC and plot ROC
    y_pred = []
    y_true = []
    for batch in tqdm(loader):
        y_model = model(batch['sequence'].to(device))
        pred = torch.sigmoid(y_model)
        y_pred_batch = pred.view(-1).detach().cpu().numpy()
        true = batch['target'].view(-1).cpu().numpy()
        y_pred.extend(y_pred_batch)
        y_true.extend(true)

    out = solution.compute_auc(np.array(y_true, dtype=np.int), np.array(y_pred))
    plot_roc(np.array(y_true, dtype=np.int), np.array(y_pred), title=title)
    return out['auc']


def plot_pwm(pwm, idx, corr):
    plt.figure()
    plt.imshow(pwm)
    plt.colorbar()
    plt.title('Kernel # ' + str(idx) + ' p = ' + str(corr))
    plt.savefig(str(idx) + '.png')


def main():
    # The hyperparameters we will use
    batch_size = 64
    learning_rate = 0.002

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # set RNG
    seed = 42
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type=='cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # unzip data
    if not os.path.exists('er.h5'):
        os.system('unzip er.zip')

    do_q1 = False
    do_q2 = False

    ########################################################################

    # QUESTION 5.1

    # Compute AUC and plot ROC of untrained model
    # Load trained model and compute AUC + plot ROC

    # Load untrained model
    if do_q1:
        model = solution.Basset().to(device)
        model.eval()

        # Load test data
        basset_dataset_test = solution.BassetDataset(path='.', f5name='er.h5', split='test')
        basset_dataloader_test = DataLoader(basset_dataset_test,
                                            batch_size=128,
                                            drop_last=True,
                                            shuffle=False,
                                            num_workers=1)
        
        auc = auc_roc(model, basset_dataloader_test, device, title='Untrained Model ROC')
        print("\n\nuntrained auc: " + str(auc) + '\n\n')

        # Load trained model
        model = torch.load('model_params.pt', map_location=device)
        model.eval()
        auc = auc_roc(model, basset_dataloader_test, device, title='Trained Model ROC')
        print("\n\trained auc: " + str(auc) + '\n\n')


    # QUESTION 5.2

    # Open CTCF motif in MA0139.1.jaspar
    ctcf = np.zeros((4, 19))
    with open('MA0139.1.jaspar') as file:
        rows = file.readlines()
        for i in range(len(rows)):
            data = rows[i].split()
            for j in range(len(data)):
                number = re.sub('\D', '', data[j])
                ctcf[i,j] = float(number)

    # Normalize so columns sum to 1
    for i in range(ctcf.shape[1]):
        ctcf[:,i] = ctcf[:,i] / np.sum(ctcf[:,i])

    plt.figure()
    plt.imshow(ctcf)
    plt.colorbar()
    plt.title('CTCF Normalized Motif')
    plt.savefig('ctcf.png')
    
    
    # QUESTION 5.3
    # Convert 300 filters of first layer to normalized PWMs
    # Determine maximum activated value in test set
    model = torch.load('model_params.pt', map_location=device)
    model.eval()

    # Load test data
    basset_dataset_test = solution.BassetDataset(path='.', f5name='er.h5', split='test')
    basset_dataloader_test = DataLoader(basset_dataset_test,
                                        batch_size=128,
                                        drop_last=True,
                                        shuffle=False,
                                        num_workers=1)
    
    # Get max activations
    max_activations = np.zeros((300))
    for batch in tqdm(basset_dataloader_test):
        activations = model.get_kernel_activation(batch['sequence'].to(device))
        max_activations = np.maximum(activations, max_activations)    

    # Loader with smaller batch size
    basset_dataloader_test = DataLoader(basset_dataset_test,
                                        batch_size=1,
                                        drop_last=True,
                                        shuffle=False,
                                        num_workers=1)

    # Get base pair counts
    shape = (300, 19, 4)
    i = 0
    base_pair_count = np.zeros(shape)
    for batch in tqdm(basset_dataloader_test):
        base_pair_count += model.count(batch['sequence'].to(device), max_activations, shape)    
        i += 1
        if i == 300:
            break

    # Normalize PWMs
    for i in range(base_pair_count.shape[0]):
        for j in range(base_pair_count.shape[1]):
            base_pair_count[i,j,:] = base_pair_count[i,j,:] / np.sum(base_pair_count[i,j,:])
    base_pair_count = base_pair_count.swapaxes(1, 2)

    # Check correlation
    correlation = np.zeros(base_pair_count.shape[0])
    for i in range(base_pair_count.shape[0]):
        correlation[i] = np.abs(np.corrcoef(ctcf.flatten(), base_pair_count[i].flatten())[0][1])

    # Keep 3 best and plot
    best_idx = np.argsort(correlation)
    for i in range(10):
        idx = best_idx[-i]
        plot_pwm(base_pair_count[idx], idx, correlation[idx])



if __name__ == '__main__':
    main()