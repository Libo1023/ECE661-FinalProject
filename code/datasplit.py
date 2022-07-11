import random
import numpy as np


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    input:   param dataset
             param num_users
    return:  dict of data index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    input:   param dataset
             param num_users
    return:  dict of data index
    """
    num_shards, num_imgs = 200, 225
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    targets = np.array(dataset.targets)

    # sort labels
    idxs_targets = np.vstack((idxs, targets))
    idxs_targets = idxs_targets[:, idxs_targets[1, :].argsort()]
    idxs = idxs_targets[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def cifar_noniid_simple(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset, it is highly unbalanced non-iid, with only one class data in each device
    input:    param dataset
              param num_users
    return:   dict of data index
    """
    dict_type, all_idxs = {}, [i for i in range(10)]
    List2=[]
    for j in range(10):
        for i in range(len(dataset)):
            if(dataset.targets[i] == j):
                List2.append(i)
        dict_type[j] = List2
        List2 = []
    return dict_type


def cifar_noniid_simple2(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset, it is highly unbalanced non-iid, with only two class data in each device
    input:    param dataset
              param num_users
    return:   dict of data index
    """
    dict_type, all_idxs = {}, [i for i in range(10)]
    List2=[]
    List5 = []
    [List5.append(i) for i in range (10)]
    for j in range(num_users):
        random.shuffle(List5)
        for i in range(45000):
#             if(len(List2) < 4500):
            if(dataset.targets[i] == List5[0] or dataset.targets[i] == List5[1]):
                List2.append(i)
        dict_type[j] = List2
        List2 = []
    return dict_type


def cifar_noniid_simple3(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset, it is highly unbalanced non-iid, with only three class data in each device
    input:    param dataset
              param num_users
    return:   dict of data index
    """
    dict_type, all_idxs = {}, [i for i in range(10)]
    List2=[]
    List5 = []
    [List5.append(i) for i in range (10)]
    for j in range(num_users):
        random.shuffle(List5)
        for i in range(45000):
            if(dataset.targets[i] == List5[0] or dataset.targets[i] == List5[1] or dataset.targets[i] == List5[2]):
                List2.append(i)
        dict_type[j] = List2
        List2 = []
    return dict_type


