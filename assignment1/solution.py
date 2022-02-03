# This script contains the helper functions you will be using for this assignment

import os
import random

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class BassetDataset(Dataset):
    """
    BassetDataset class taken with permission from Dr. Ahmad Pesaranghader

    We have already processed the data in HDF5 format: er.h5
    See https://www.h5py.org/ for details of the Python package used

    We used the same data processing pipeline as the paper.
    You can find the code here: https://github.com/davek44/Basset
    """

    # Initializes the BassetDataset
    def __init__(self, path='./data/', f5name='er.h5', split='train', transform=None):
        """
        Args:
            :param path: path to HDF5 file
            :param f5name: HDF5 file name
            :param split: split that we are interested to work with
            :param transform (callable, optional): Optional transform to be applied on a sample
        """

        self.split = split

        split_dict = {'train': ['train_in', 'train_out'],
                      'test': ['test_in', 'test_out'],
                      'valid': ['valid_in', 'valid_out']}

        assert self.split in split_dict, "'split' argument can be only defined as 'train', 'valid' or 'test'"

        # Open hdf5 file where one-hoted data are stored
        self.dataset = h5py.File(os.path.join(path, f5name.format(self.split)), 'r')

        # Keeping track of the names of the target labels
        self.target_labels = self.dataset['target_labels']

        # Get the list of volumes
        self.inputs = self.dataset[split_dict[split][0]]
        self.outputs = self.dataset[split_dict[split][1]]

        self.ids = list(range(len(self.inputs)))
        if self.split == 'test':
            self.id_vars = np.char.decode(self.dataset['test_headers'])

    def __getitem__(self, i):
        """
        Returns the sequence and the target at index i

        Notes:
        * The data is stored as float16, however, your model will expect float32.
          Do the type conversion here!
        * Pay attention to the output shape of the data.
          Change it to match what the model is expecting
          hint: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        * The target must also be converted to float32
        """

        idx = self.ids[i]

        # Sequence & Target
        output = {'sequence': None, 'target': None}

        # WRITE CODE HERE
        breakpoint()

        return output

    def __len__(self):
        return self.outputs.shape[0]

    def get_seq_len(self):
        """
        Answer to Q1 part 2
        """
        # WRITE CODE HERE
        breakpoint()
        return 0

    def is_equivalent(self):
        """
        Answer to Q1 part 3
        """
        # WRITE CODE HERE
        breakpoint()
        return 0


class Basset(nn.Module):
    """
    Basset model
    Architecture specifications can be found in the supplementary material
    You will also need to use some Convolution Arithmetic
    """

    def __init__(self):
        super(Basset, self).__init__()

        self.dropout = nn.Dropout(0.3)
        self.num_cell_types = 164

        # TODO dimensions
        # self.conv1 = nn.Conv2d(1, 300, (19, ?), stride=(1, 1), padding=(9, 0))
        # self.conv2 = nn.Conv2d(300, 200, (11, 1), stride=(1, 1), padding=(?, 0))
        # self.conv3 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(4, 0))

        self.bn1 = nn.BatchNorm2d(300)
        self.bn2 = nn.BatchNorm2d(200)
        self.bn3 = nn.BatchNorm2d(200)
        self.maxpool1 = nn.MaxPool2d((3, 1))
        self.maxpool2 = nn.MaxPool2d((4, 1))
        self.maxpool3 = nn.MaxPool2d((4, 1))

        # TODO hbn4 and bn5 needed?
        self.fc1 = nn.Linear(13*200, 1000)
        self.bn4 = nn.BatchNorm1d(1000)

        self.fc2 = nn.Linear(1000, 1000)
        self.bn5 = nn.BatchNorm1d(1000)

        self.fc3 = nn.Linear(1000, self.num_cell_types)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxpool3(x)

        x = self.fc1(x.view(x.shape[0], -1))
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        output = self.fc3(x)

        return output


def compute_fpr_tpr(y_true, y_pred):
    """
    Computes the False Positive Rate and True Positive Rate
    Args:
        :param y_true: groundtruth labels (np.array of ints)
        :param y_pred: model decisions (np.array of ints)

    :Return: dict with tpr, fpr (values are floats)
    """
    output = {'fpr': 0., 'tpr': 0.}

    # True positives 
    tp = np.sum(np.logical_and(y_pred == 1, y_true == 1))
 
    # True negatives 
    tn = np.sum(np.logical_and(y_pred == 0, y_true == 0))
 
    # False positives 
    fp = np.sum(np.logical_and(y_pred == 1, y_true == 0))
     
    # False negatives 
    fn = np.sum(np.logical_and(y_pred == 0, y_true == 1))

    output['tpr'] = tp / (tp + fn)
    output['fpr'] = fp / (tn + fp)

    return output


def compute_fpr_tpr_dumb_model():
    """
    Simulates a dumb model and computes the False Positive Rate and True Positive Rate

    :Return: dict with tpr_list, fpr_list.
             These lists contain the tpr and fpr for different thresholds
             fpr and tpr values in the lists should be floats
             Order the lists such that:
                 output['fpr_list'][0] corresponds to k=0.
                 output['fpr_list'][1] corresponds to k=0.05 
                 ...
            Do the same for output['tpr_list']
             
    """
    output = {'fpr_list': [], 'tpr_list': []}

    # Sample 1000 data points and 1000 targets 
    samples = np.random.uniform(size=1000)
    labels = np.random.randint(0, high=2, size=1000)
    
    # Compute fpr and tpr for each k value
    k_list = np.linspace(0, 1, num=21, endpoint=True)
    for k in k_list:
        y_pred = samples > k
        results = compute_fpr_tpr(labels, y_pred.astype(int))
        output['fpr_list'].append(results['fpr'])
        output['tpr_list'].append(results['tpr'])

    return output


def compute_fpr_tpr_smart_model():
    """
    Simulates a smart model and computes the False Positive Rate and True Positive Rate

    :Return: dict with tpr_list, fpr_list.
             These lists contain the tpr and fpr for different thresholds
             fpr and tpr values in the lists should be floats
             Order the lists such that:
                 output['fpr_list'][0] corresponds to k=0.
                 output['fpr_list'][1] corresponds to k=0.05 
                 ...
            Do the same for output['tpr_list']
    """
    output = {'fpr_list': [], 'tpr_list': []}

    # Sample 1000 targets 
    labels = np.random.randint(0, high=2, size=1000)

    # Sample data according to target
    samples = np.zeros(labels.shape)
    for i in range(samples.shape[0]):
        if labels[i] == 1:
            samples[i] = np.random.uniform(low=0.6, high=1.0)
        else:
            samples[i] = np.random.uniform(low=0.0, high=0.4)
    
    # Compute fpr and tpr for each k value
    k_list = np.linspace(0, 1, num=21, endpoint=True)
    for k in k_list:
        y_pred = samples > k
        results = compute_fpr_tpr(labels, y_pred.astype(int))
        output['fpr_list'].append(results['fpr'])
        output['tpr_list'].append(results['tpr'])

    return output


def compute_auc_both_models():
    """
    Simulates a dumb model and a smart model and computes the AUC of both

    :Return: dict with auc_dumb_model, auc_smart_model.
             These contain the AUC for both models
             auc values in the lists should be floats
    """
    output = {'auc_dumb_model': 0., 'auc_smart_model': 0.}

    # Sample 1000 targets 
    labels = np.random.randint(0, high=2, size=1000)

    # Sample data 
    dumb_samples = np.random.uniform(size=1000)
    smart_samples = np.zeros(labels.shape)
    for i in range(smart_samples.shape[0]):
        if labels[i] == 1:
            smart_samples[i] = np.random.uniform(low=0.6, high=1.0)
        else:
            smart_samples[i] = np.random.uniform(low=0.0, high=0.4)

    # Compute AUC
    k = 0.5
    smart_preds = smart_samples > 0.5
    dumb_preds = dumb_samples > 0.5
    output['auc_smart_model'] = compute_auc(labels, smart_preds.astype(int))
    output['auc_dumb_model'] = compute_auc(labels, dumb_preds.astype(int))

    return output


def compute_auc_untrained_model(model, dataloader, device):
    """
    Computes the AUC of your input model

    Dont forget to re-apply your output activation!

    :Return: dict with auc_dumb_model, auc_smart_model.
             These contain the AUC for both models
             auc values should be floats

    Make sure this function works with arbitrarily small dataset sizes!
    """
    output = {'auc': 0.}

    # WRITE CODE HERE
    breakpoint()

    return output


def compute_auc(y_true, y_model):
    """
    Computes area under the ROC curve
    auc returned should be float
    Args:
        :param y_true: groundtruth labels (np.array of ints)
        :param y_pred: model decisions (np.array of ints)
    """
    output = {'auc': 0.}

    # TODO: actually compute auc
    # return mAP for now
    output['auc'] = np.sum(np.equal(y_model, y_true)) / y_model.shape[0]
 
    return output


def get_critereon():
    """
    Picks the appropriate loss function for our task
    criterion should be subclass of torch.nn
    """

    # WRITE CODE HERE
    breakpoint()

    return critereon


def train_loop(model, train_dataloader, device, optimizer, criterion):
    """
    One Iteration across the training set
    Args:
        :param model: solution.Basset()
        :param train_dataloader: torch.utils.data.DataLoader
                                 Where the dataset is solution.BassetDataset
        :param device: torch.device
        :param optimizer: torch.optim
        :param critereon: torch.nn (output of get_critereon)

    :Returns: output dict with keys: total_score, total_loss
    values of each should be floats
    (if you want to record losses or scores within the, you may print them to screen)

    Make sure your loop works with arbitrarily small dataset sizes!
    """

    output = {'total_score': 0.,
              'total_loss': 0.}

    # WRITE CODE HERE
    breakpoint()

    return output['total_score'], output['total_loss']


def valid_loop(model, valid_dataloader, device, optimizer, criterion):
    """
    One Iteration across the validation set
    Args:
        :param model: solution.Basset()
        :param valid_dataloader: torch.utils.data.DataLoader
                                 Where the dataset is solution.BassetDataset
        :param device: torch.device
        :param optimizer: torch.optim
        :param critereon: torch.nn (output of get_critereon)

    :Returns: output dict with keys: total_score, total_loss
    values of each should be floats
    (if you want to record losses or scores within the, you may print them to screen)

    Make sure your loop works with arbitrarily small dataset sizes!
    """

    output = {'total_score': 0.,
              'total_loss': 0.}

    # WRITE CODE HERE
    breakpoint()

    return output['total_score'], output['total_loss']
