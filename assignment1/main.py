import random
import sys
import os
import shutil
import warnings
import numpy as np
import h5py

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

import solution



def main():
    # The hyperparameters we will use
    batch_size = 64
    learning_rate = 0.002

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    ########################################################################

    # QUESTION 1

    ########################################################################

    # investigate your data
    f = h5py.File('er.h5', 'r')
    print('Dataset keys')
    print(f.keys())
    f.close()

    # Load data
    basset_dataset_train = solution.BassetDataset(path='.', f5name='er.h5', split='train')
    basset_dataset_valid = solution.BassetDataset(path='.', f5name='er.h5', split='valid')
    basset_dataset_test = solution.BassetDataset(path='.', f5name='er.h5', split='test')
    basset_dataloader_train = DataLoader(basset_dataset_train,
                                         batch_size=batch_size,
                                         drop_last=True,
                                         shuffle=True,
                                         num_workers=1)
    basset_dataloader_valid = DataLoader(basset_dataset_valid,
                                         batch_size=batch_size,
                                         drop_last=True,
                                         shuffle=False,
                                         num_workers=1)
    basset_dataloader_test = DataLoader(basset_dataset_test,
                                        batch_size=batch_size,
                                        drop_last=True,
                                        shuffle=False,
                                        num_workers=1)

    ########################################################################

    # QUESTION 2

    ########################################################################

    model = solution.Basset().to(device)

    ########################################################################

    # QUESTION 3

    ########################################################################
    out = solution.compute_fpr_tpr_dumb_model()
    # plt.figure()
    # plt.plot(out['fpr_list'], out['tpr_list'])
    # plt.show()

    out = solution.compute_fpr_tpr_smart_model()
    # plt.figure()
    # plt.plot(out['fpr_list'], out['tpr_list'])
    # plt.show()
    
    out = solution.compute_auc_both_models()
    out = solution.compute_auc_untrained_model(model, basset_dataloader_test, device)

if __name__ == '__main__':
    main()