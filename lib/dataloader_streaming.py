import os
import pickle as pkl
import sys

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from lib.utils import Scaler_NYC, Scaler_Chi

# high frequency time
high_fre_hour = [6, 7, 8, 15, 16, 17, 18]


def split_and_norm_data_time_continual(all_data, splits=[0.3, 0.175, 0.175, 0.175, 0.175], recent_prior=3, week_prior=4,
                                       one_day_period=24,
                                       days_of_week=7, pre_len=1):
    num_of_time, channel, _, _ = all_data.shape
    split_indices = np.cumsum([int(s * num_of_time) for s in splits])
    split_indices = np.insert(split_indices, 0, 0)

    for index, (start, end) in enumerate(zip(split_indices[:-1], split_indices[1:])):
        if index == 0:
            if channel == 48:  # NYC
                scaler = Scaler_NYC(all_data[start:end, :, :, :])
            elif channel == 41:  # Chicago
                scaler = Scaler_Chi(all_data[start:end, :, :, :])
        norm_data = scaler.transform(all_data[start:end, :, :, :])
        X, Y, target_time = [], [], []
        high_X, high_Y, high_target_time = [], [], []
        for i in range(len(norm_data) - week_prior * days_of_week * one_day_period - pre_len + 1):
            t = i + week_prior * days_of_week * one_day_period
            label = norm_data[t:t + pre_len, 0, :, :]
            period_list = []
            for week in range(week_prior):
                period_list.append(i + week * days_of_week * one_day_period)
            for recent in list(range(1, recent_prior + 1))[::-1]:
                period_list.append(t - recent)
            feature = norm_data[period_list, :, :, :]
            X.append(feature)
            Y.append(label)
            target_time.append(norm_data[t, 1:33, 0, 0])
            if list(norm_data[t, 1:25, 0, 0]).index(1) in high_fre_hour:
                high_X.append(feature)
                high_Y.append(label)
                high_target_time.append(norm_data[t, 1:33, 0, 0])
        yield np.array(X), np.array(Y), np.array(target_time), np.array(high_X), np.array(high_Y), np.array(
            high_target_time), scaler


def normal_and_generate_dataset_time_continual(all_data_filename, splits=[0.3, 0.175, 0.175, 0.175, 0.175],
                                               recent_prior=3, week_prior=4,
                                               one_day_period=24, days_of_week=7, pre_len=1):
    all_data = pkl.load(open(all_data_filename, 'rb')).astype(np.float32)
    for i in split_and_norm_data_time_continual(all_data,
                                                splits=splits,
                                                recent_prior=recent_prior,
                                                week_prior=week_prior,
                                                one_day_period=one_day_period,
                                                days_of_week=days_of_week,
                                                pre_len=pre_len):
        yield i


def generate_dataloader_continual(
        all_data_filename,
        grid_node_data_filename,
        batch_size,
        splits=[0.3, 0.175, 0.175, 0.175, 0.175],
        recent_prior=3,
        week_prior=4,
        one_day_period=24,
        days_of_week=7,
        pre_len=1,
        test=False,
        north_south_map=20,
        west_east_map=20):
    train_loaders = []
    val_loaders = []
    test_loaders = []
    high_test_loaders = []
    scaler_r = ""
    train_data_shape_r = ""
    time_shape_r = ""
    graph_feature_shape_r = ""
    for idx, (x, y, target_times, high_x, high_y, high_target_times, scaler) in enumerate(
            normal_and_generate_dataset_time_continual(
                all_data_filename,
                splits=splits,
                recent_prior=recent_prior,
                week_prior=week_prior,
                one_day_period=one_day_period,
                days_of_week=days_of_week,
                pre_len=pre_len)):

        if test:
            len = 64
            x = x[:len]
            y = y[:len]
            target_times = target_times[:len]
            high_x = high_x[:len]
            high_y = high_y[:len]
            high_target_times = high_target_times[:len]

        grid_node_map = get_grid_node_map_maxtrix(grid_node_data_filename)
        if 'nyc' in all_data_filename:
            graph_x = x[:, :, [0, 46, 47], :, :].reshape((x.shape[0], x.shape[1], -1, north_south_map * west_east_map))
            high_graph_x = high_x[:, :, [0, 46, 47], :, :].reshape(
                (high_x.shape[0], high_x.shape[1], -1, north_south_map * west_east_map))
            graph_x = np.dot(graph_x, grid_node_map)
            high_graph_x = np.dot(high_graph_x, grid_node_map)
        if 'chicago' in all_data_filename:
            graph_x = x[:, :, [0, 39, 40], :, :].reshape((x.shape[0], x.shape[1], -1, north_south_map * west_east_map))
            high_graph_x = high_x[:, :, [0, 39, 40], :, :].reshape(
                (high_x.shape[0], high_x.shape[1], -1, north_south_map * west_east_map))
            graph_x = np.dot(graph_x, grid_node_map)
            high_graph_x = np.dot(high_graph_x, grid_node_map)

        print("feature:", str(x.shape), "label:", str(y.shape), "time:", str(target_times.shape),
              "high feature:", str(high_x.shape), "high label:", str(high_y.shape))
        print("graph_x:", str(graph_x.shape), "high_graph_x:", str(high_graph_x.shape))

        if idx == 0:  # first sub-dataset
            scaler_r = scaler
            train_data_shape_r = x.shape
            time_shape_r = target_times.shape
            graph_feature_shape_r = graph_x.shape

        split_train_len = int(x.shape[0] * 0.6)
        split_val_len = int(x.shape[0] * 0.2)

        train_loaders.append(DataLoader(
            TensorDataset(
                torch.from_numpy(x[:split_train_len]),
                torch.from_numpy(target_times[:split_train_len]),
                torch.from_numpy(graph_x[:split_train_len]),
                torch.from_numpy(y[:split_train_len])
            ),
            batch_size=batch_size,
            shuffle=True
        ))

        val_loaders.append(DataLoader(
            TensorDataset(
                torch.from_numpy(x[split_train_len:split_train_len + split_val_len]),
                torch.from_numpy(target_times[split_train_len:split_train_len + split_val_len]),
                torch.from_numpy(graph_x[split_train_len:split_train_len + split_val_len]),
                torch.from_numpy(y[split_train_len:split_train_len + split_val_len])
            ),
            batch_size=batch_size,
            shuffle=False
        ))

        test_loaders.append(DataLoader(
            TensorDataset(
                torch.from_numpy(x[split_train_len + split_val_len:]),
                torch.from_numpy(target_times[split_train_len + split_val_len:]),
                torch.from_numpy(graph_x[split_train_len + split_val_len:]),
                torch.from_numpy(y[split_train_len + split_val_len:])
            ),
            batch_size=batch_size,
            shuffle=False
        ))

        high_test_loaders.append(DataLoader(
            TensorDataset(
                torch.from_numpy(high_x),
                torch.from_numpy(high_target_times),
                torch.from_numpy(high_graph_x),
                torch.from_numpy(high_y)
            ),
            batch_size=batch_size,
            shuffle=False
        ))

    return scaler_r, train_data_shape_r, time_shape_r, graph_feature_shape_r, train_loaders, val_loaders, test_loaders, high_test_loaders


def get_mask(mask_path):
    """
    Arguments:
        mask_path {str} -- mask filename
    
    Returns:
        {np.array} -- mask matrix，shape(W,H)
    """
    mask = pkl.load(open(mask_path, 'rb')).astype(np.float32)
    return mask


def get_adjacent(adjacent_path):
    """
    Arguments:
        adjacent_path {str} -- adjacent matrix path
    
    Returns:
        {np.array} -- shape:(N,N)
    """
    adjacent = pkl.load(open(adjacent_path, 'rb')).astype(np.float32)
    return adjacent


def get_grid_node_map_maxtrix(grid_node_path):
    """
    Arguments:
        grid_node_path {str} -- filename
    
    Returns:
        {np.array} -- shape:(W*H,N)
    """
    grid_node_map = pkl.load(open(grid_node_path, 'rb')).astype(np.float32)
    return grid_node_map


def get_trans(trans_path):
    """
    Arguments:
        trans_path {str} -- filename

    Returns:
        {np.array} -- shape:(N_f,N_c)
    """
    trans = pkl.load(open(trans_path, 'rb')).astype(np.float32)
    return trans


def is_image_file(filename):
    """
    determines whether the arguments are an image file
    @param filename: string of the files path
    @return: a boolean indicating whether the file is an image file
    """
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def image_loader(image_name, transformation, add_fake_batch_dimension=True):
    """
    loads an image
    :param image_name: the path of the image
    :param transformation: the transformation done on the image
    :param add_fake_batch_dimension: should add a 4th batch dimension
    :return: the image on the current device
    """
    image = Image.open(image_name).convert('RGB')
    # fake batch dimension required to fit network's input dimensions
    if add_fake_batch_dimension:
        image = transformation(image).unsqueeze(0)
    else:
        image = transformation(image)
    return image


class rsdataset(Dataset):
    """
    dataset class for remote sensing image dataset
    """

    def __init__(self, root_dir, loader):
        self.root_dir = root_dir
        self.loader = loader
        self.image_list = [x for x in os.listdir(root_dir) if is_image_file(x)]
        self.image_list.sort(key=lambda x: int(x[7:-4]))
        self.len = len(self.image_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = image_loader(img_name, self.loader, add_fake_batch_dimension=False)
        sample = {'image': image}

        return sample


def get_remote_sensing_dataloader(norm, imsize, path):
    # loaders
    loaders = {
        'std': transforms.Compose(
            [transforms.Resize((imsize, imsize)),
             # transforms.RandomResizedCrop(256),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        'no_norm': transforms.Compose(
            [transforms.Resize((imsize, imsize)),
             # transforms.RandomResizedCrop(imsize/2),
             transforms.ToTensor()])
    }
    loader = loaders[norm]
    remote_sensing_dateset = rsdataset(path, loader)

    # remote_sensing_dataloader = DataLoader(remote_sensing_dateset, batch_size=400, shuffle=True, num_workers=16)
    remote_sensing_dataloader = DataLoader(remote_sensing_dateset, batch_size=400)

    return remote_sensing_dataloader
