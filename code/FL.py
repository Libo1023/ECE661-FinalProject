import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import argparse
import os, sys
import pickle
import time
import datetime
from tqdm import tqdm_notebook as tqdm
import copy

class DatasetSplit(Dataset):
    """
    According to the idxs to split the original dataset and allocate to the local device

    init:    given dataset and the idxs of allocated data
    return:  dataset
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        inputs, targets = self.dataset[self.idxs[item]]
        return torch.tensor(inputs), torch.tensor(targets)

    ###################################  Modified  ####################


class LocalUpdate(object):
    def __init__(self, dataset, idxs, Gweight):
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        split the dataset and guarantee the length of data

        input para: given dataset and its idxs. idxs means the local device's id
        Return:     train, validation and test dataloaders
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8 * len(idxs))]
        idxs_val = idxs[int(0.8 * len(idxs)):int(0.9 * len(idxs))]
        idxs_test = idxs[int(0.9 * len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=50, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val) / 10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test) / 10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, Gweight):
        # Set mode to train model
        model.train()
        if Gweight != None:
            model.load_state_dict(Gweight)
        epoch_loss = []

        # Set optimizer for the local updates
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)  ## 0.01

        for iter in range(3):
            batch_loss = []
            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                model.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, targets.long())
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_weights_prox(self, model, global_net, Gweight, mu):
        # Set mode to train model
        model.train()
        epoch_loss = []
        if Gweight != None:
            model.load_state_dict(Gweight)
        global_weight_collector = list(global_net.to(self.device).parameters())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

        for iter in range(5):
            batch_loss = []
            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                model.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, targets.long())
                ###  Change the loss function with weight regularization  ###
                loss2 = 0
                for param_index, param in enumerate(model.parameters()):
                    loss2 += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss += loss2

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """
        send back the accuracy and loss

        input para: model
        return:     accuracy and loss
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Inference
            outputs = model(inputs)
            batch_loss = self.criterion(outputs, targets.long())
            loss += batch_loss.item()

            # Prediction
            _, pred_targets = torch.max(outputs, 1)
            pred_targets = pred_targets.view(-1)
            correct += torch.sum(torch.eq(pred_targets, targets)).item()
            total += len(targets)

        accuracy = correct / total
        return accuracy, loss


def average_weights(w):
    """
    Calculate weights' average

    input para: weight
    return:     average weight
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg