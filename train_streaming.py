import torch
import torch.optim as optim

import numpy as np
import json
import copy
import time
import argparse
import random

import sys
import os

from tqdm import tqdm

from lib.dataloader_streaming import get_mask, get_adjacent, get_grid_node_map_maxtrix, \
    get_trans, get_remote_sensing_dataloader, generate_dataloader_continual
from lib.early_stop import EarlyStopping
from model.MGHSTN_r import MGHSTN
from lib.utils import mask_loss, compute_loss, predict_and_evaluate

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file')
parser.add_argument("--gpus", type=str, help="test program")
parser.add_argument("--test", action="store_true", help="test program")
parser.add_argument("--nors", action="store_true", help="no remote sensing")
parser.add_argument("--infer", action="store_true", help="load model from file and infer")

args = parser.parse_args()
config_filename = args.config
with open(config_filename, 'r') as f:
    config = json.loads(f.read())
print(json.dumps(config, sort_keys=True, indent=4))

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if config['seed'] is not None:
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def loadConfig(config):
    patience = config['patience']
    delta = config['delta']

    train_rate = config['train_rate']
    valid_rate = config['valid_rate']
    recent_prior = config['recent_prior']
    week_prior = config['week_prior']
    one_day_period = config['one_day_period']
    days_of_week = config['days_of_week']
    pre_len = config['pre_len']
    seq_len = recent_prior + week_prior
    training_epoch = config['training_epoch']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    num_of_transformer_layers = config['num_of_transformer_layers']
    transformer_hidden_size = config['transformer_hidden_size']
    gcn_num_filter = config['gcn_num_filter']

    remote_sensing_image_path = config['remote_sensing_image_path']
    num_of_heads = config['num_of_heads']
    augment_channel = config['augment_channel']

    all_data_filename = []
    mask_filename = []
    road_adj_filename = []
    risk_adj_filename = []
    poi_adj_filename = []
    sum_adj_filename = []
    grid_node_filename = []
    north_south_map = []
    west_east_map = []

    trans20_10_filename = config['trans20_10_filename']
    trans10_5_filename = config['trans10_5_filename']
    trans5_2_filename = config['trans5_2_filename']

    trans = [trans20_10_filename, trans10_5_filename, trans5_2_filename]
    for i in range(4):
        all_data_filename.append(config['all_data_filename_{}'.format(i + 1)])
        mask_filename.append(config['mask_filename_{}'.format(i + 1)])
        road_adj_filename.append(config['road_adj_filename_{}'.format(i + 1)])
        risk_adj_filename.append(config['risk_adj_filename_{}'.format(i + 1)])
        poi_adj_filename.append(config['poi_adj_filename_{}'.format(i + 1)])
        sum_adj_filename.append(config['sum_adj_filename_{}'.format(i + 1)])
        grid_node_filename.append(config['grid_node_filename_{}'.format(i + 1)])
        north_south_map.append(config['north_south_map_{}'.format(i + 1)])
        west_east_map.append(config['west_east_map_{}'.format(i + 1)])

    bfc_20_10_filename = config['bfc_20_10_filename']

    return (patience, delta, train_rate, valid_rate, recent_prior, week_prior, one_day_period, days_of_week, pre_len,
            seq_len, training_epoch, batch_size, learning_rate, num_of_transformer_layers, transformer_hidden_size,
            gcn_num_filter, remote_sensing_image_path,
            all_data_filename, mask_filename, road_adj_filename, risk_adj_filename, poi_adj_filename, sum_adj_filename,
            grid_node_filename, north_south_map, west_east_map, trans, bfc_20_10_filename, num_of_heads, augment_channel)


class PrioritizedExperienceReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity  # 经验池的最大容量
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)  # 初始化优先级
        self.alpha = alpha  # 优先级的平滑系数
        self.position = 0

    def add(self, experience, error):
        priority = (error + 1e-5) ** self.alpha  # 基于误差计算优先级
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        # 使用误差设定优先级
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity  # 环形队列方式存储

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        probabilities = priorities
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        # 计算重要性采样权重
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_errors):
        for idx, error in zip(batch_indices, batch_errors):
            self.priorities[idx] = (error + 1e-5) ** self.alpha  # 基于新的误差更新优先级

    def current_capacity(self):
        """
        返回当前缓冲区中存储的经验数量。
        """
        return len(self.buffer)


class DualMemoryExperienceReplayBuffer:
    def __init__(self, stm_capacity, ltm_capacity, alpha=0.6):
        # 短期记忆缓冲区（STM）
        self.stm_capacity = stm_capacity
        self.stm_buffer = []
        self.stm_priorities = np.zeros((stm_capacity,), dtype=np.float32)
        self.stm_position = 0

        # 长期记忆缓冲区（LTM）
        self.ltm_capacity = ltm_capacity
        self.ltm_buffer = []
        self.ltm_priorities = np.zeros((ltm_capacity,), dtype=np.float32)
        self.ltm_position = 0

        self.alpha = alpha  # 优先级的平滑系数

    def add(self, experience, error, to_ltm=False):
        """
        添加经验到 STM 或 LTM

        :param experience: 经验数据
        :param error: 该经验的误差，用于计算优先级
        :param to_ltm: 是否添加到长期记忆缓冲区
        """
        priority = (error + 1e-5) ** self.alpha  # 基于误差计算优先级

        if to_ltm:
            # 添加到 LTM
            if len(self.ltm_buffer) < self.ltm_capacity:
                self.ltm_buffer.append(experience)
            else:
                self.ltm_buffer[self.ltm_position] = experience

            self.ltm_priorities[self.ltm_position] = priority
            self.ltm_position = (self.ltm_position + 1) % self.ltm_capacity  # 环形存储
        else:
            # 添加到 STM
            if len(self.stm_buffer) < self.stm_capacity:
                self.stm_buffer.append(experience)
            else:
                self.stm_buffer[self.stm_position] = experience

            self.stm_priorities[self.stm_position] = priority
            self.stm_position = (self.stm_position + 1) % self.stm_capacity  # 环形存储

    def sample(self, batch_size, beta=0.4, stm_ratio=0.75):
        """
        从 STM 和 LTM 中采样经验

        :param batch_size: 采样的总批次大小
        :param beta: 重要性采样权重的参数
        :param stm_ratio: 从 STM 中采样的比例（0 到 1 之间）
        :return: 采样的经验、对应的索引和重要性采样权重
        """
        stm_batch_size = int(batch_size * stm_ratio)
        ltm_batch_size = batch_size - stm_batch_size

        samples = []
        indices = []
        weights = []

        # 从 STM 中采样
        if len(self.stm_buffer) > 0 and stm_batch_size > 0:
            stm_samples, stm_indices, stm_weights = self._sample_from_buffer(
                self.stm_buffer, self.stm_priorities, stm_batch_size, beta)
            samples.extend(stm_samples)
            indices.extend([('stm', idx) for idx in stm_indices])
            weights.extend(stm_weights)

        # 从 LTM 中采样
        if len(self.ltm_buffer) > 0 and ltm_batch_size > 0:
            ltm_samples, ltm_indices, ltm_weights = self._sample_from_buffer(
                self.ltm_buffer, self.ltm_priorities, ltm_batch_size, beta)
            samples.extend(ltm_samples)
            indices.extend([('ltm', idx) for idx in ltm_indices])
            weights.extend(ltm_weights)

        return samples, indices, weights

    def _sample_from_buffer(self, buffer, priorities, batch_size, beta):
        """
        从指定的缓冲区中采样

        :param buffer: STM 或 LTM 缓冲区
        :param priorities: 对应的优先级数组
        :param batch_size: 采样的批次大小
        :param beta: 重要性采样权重的参数
        :return: 采样的经验、对应的索引和重要性采样权重
        """
        if len(buffer) == len(priorities):
            current_priorities = priorities
        else:
            current_priorities = priorities[:len(buffer)]

        probabilities = current_priorities
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(buffer), batch_size, p=probabilities)
        samples = [buffer[idx] for idx in indices]

        # 计算重要性采样权重
        total = len(buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_errors):
        """
        更新经验的优先级

        :param batch_indices: 经验的索引列表，包含 ('stm', idx) 或 ('ltm', idx)
        :param batch_errors: 对应经验的新误差列表
        """
        for (buffer_type, idx), error in zip(batch_indices, batch_errors):
            priority = (error + 1e-5) ** self.alpha
            if buffer_type == 'stm':
                self.stm_priorities[idx] = priority
            elif buffer_type == 'ltm':
                self.ltm_priorities[idx] = priority

    def transfer_to_ltm(self, num_experiences):
        """
        将 STM 中的重要经验转移到 LTM

        :param num_experiences: 转移的经验数量
        """
        if len(self.stm_buffer) == 0:
            return

        # 根据优先级选择 top-k 的经验
        num_experiences = min(num_experiences, len(self.stm_buffer))
        indices = np.argsort(self.stm_priorities[:len(self.stm_buffer)])[-num_experiences:]

        for idx in indices:
            experience = self.stm_buffer[idx]
            priority = self.stm_priorities[idx]
            self.add(experience, priority, to_ltm=True)

    def clear_stm(self):
        """清空短期记忆缓冲区"""
        self.stm_buffer = []
        self.stm_priorities = np.zeros((self.stm_capacity,), dtype=np.float32)
        self.stm_position = 0

    def current_capacity(self):
        """返回 STM 和 LTM 当前存储的经验数量"""
        return len(self.stm_buffer), len(self.ltm_buffer)


def training_continual(net,
                       training_epoch,
                       train_loaders,
                       val_loaders,
                       test_loaders,
                       high_test_loaders,
                       road_adj,
                       risk_adj,
                       poi_adj,
                       sum_adj,
                       risk_mask,
                       grid_node_map,
                       trainer,
                       early_stop,
                       device,
                       scaler,
                       trans,
                       bfc,
                       data_type='nyc'):
    global_step = 1
    results = []
    replay_buffer = PrioritizedExperienceReplayBuffer(capacity=50)  # 初始化优先级经验池
    alpha, beta = 0.6, 0.4  # 优先级和重要性采样权重的参数

    for split_idx in range(len(train_loaders)):
        print(f"\nStarting training for Split {split_idx + 1}...\n")

        split_iterator = tqdm(range(1, training_epoch + 1), desc=f"Split {split_idx + 1}", unit="epoch")

        for epoch in split_iterator:
            net.train()

            # 从经验池中抽样
            if len(replay_buffer.buffer) > 0:
                replay_samples, replay_indices, replay_weights = replay_buffer.sample(16, beta=beta)
                for experience, weight in zip(replay_samples, replay_weights):

                    train_feature, target_time, graph_feature, train_label = experience
                    train_feature = [t.to(device) for t in train_feature]
                    target_time = [t.to(device) for t in target_time]
                    graph_feature= [t.to(device) for t in graph_feature]
                    train_label = [t.to(device) for t in train_label]


                    # # 使用经验池中的样本进行训练
                    # for i in range(4):
                    #     print("train_feature", train_feature[i].shape)
                    #     print("target_time", target_time[i].shape)
                    #     print("graph_feature", graph_feature[i].shape)
                    #     print("train_label", train_label[i].shape)

                    final_output, classification_output, consistency_loss = net(
                        train_feature, target_time, graph_feature, road_adj, risk_adj, poi_adj, sum_adj, grid_node_map, trans)
                    replay_loss = mask_loss(final_output, classification_output, train_label, risk_mask, bfc, data_type) + consistency_loss
                    loss = replay_loss * weight
                    trainer.zero_grad()
                    loss.backward()
                    trainer.step()

            # 当前split数据的训练
            epoch_iterator = tqdm(zip(train_loaders[split_idx][0],
                                      train_loaders[split_idx][1],
                                      train_loaders[split_idx][2],
                                      train_loaders[split_idx][3]),
                                  desc=f"Epoch {epoch}/{training_epoch}",
                                  unit="batch", leave=False)

            for batch_1, batch_2, batch_3, batch_4 in epoch_iterator:
                batch = [batch_1, batch_2, batch_3, batch_4]
                train_feature = []
                target_time = []
                graph_feature = []
                train_label = []
                for i in range(4):
                    t_train_feature, t_target_time, t_graph_feature, t_train_label = batch[i]
                    t_train_feature, t_target_time, t_graph_feature, t_train_label = t_train_feature.to(device), \
                        t_target_time.to(device), t_graph_feature.to(device), t_train_label.to(device)
                    train_feature.append(t_train_feature)
                    target_time.append(t_target_time)
                    graph_feature.append(t_graph_feature)
                    train_label.append(t_train_label)

                # 模型训练
                final_output, classification_output, consistency_loss = net(train_feature, target_time, graph_feature, road_adj, risk_adj,
                                                          poi_adj, sum_adj, grid_node_map, trans)
                l = mask_loss(final_output, classification_output, train_label, risk_mask, bfc, data_type) + consistency_loss
                trainer.zero_grad()
                l.backward()
                trainer.step()
                training_loss = l.cpu().item()

                epoch_iterator.set_postfix({"Training Loss": f"{training_loss:.6f}"})

                # # 在每个 batch 的训练后，将数据从 GPU 转移到 CPU，确保不会占用大量 GPU 内存
                # train_feature_cpu = [t.cpu() for t in train_feature]
                # target_time_cpu = [t.cpu() for t in target_time]
                # graph_feature_cpu = [t.cpu() for t in graph_feature]
                # train_label_cpu = [t.cpu() for t in train_label]
                #
                # # 计算每个样本的误差，用于更新优先级经验池
                # batch_errors = np.abs(training_loss).tolist()
                #
                # # 将从 GPU 移动到 CPU 后的数据添加到经验池
                # replay_buffer.add((train_feature_cpu, target_time_cpu, graph_feature_cpu, train_label_cpu),
                #                   batch_errors)

                global_step += 1

            # 每个 epoch 结束后进行验证
            val_loss = compute_loss(net, val_loaders[split_idx], risk_mask, road_adj, risk_adj, poi_adj, sum_adj,
                                    grid_node_map, trans, device, bfc, data_type)
            split_iterator.set_postfix({"Validation Loss": f"{val_loss:.6f}"})

            if epoch == 1 or val_loss < early_stop.best_score:
                test_rmse, test_recall, test_map, test_inverse_trans_pre, test_inverse_trans_label = \
                    predict_and_evaluate(net, test_loaders[split_idx], risk_mask, road_adj, risk_adj, poi_adj, sum_adj,
                                         grid_node_map, trans, scaler, device)

                high_test_rmse, high_test_recall, high_test_map, _, _ = \
                    predict_and_evaluate(net, high_test_loaders[split_idx], risk_mask, road_adj, risk_adj, poi_adj, sum_adj,
                                         grid_node_map, trans, scaler, device)

                print(
                    f'Test Results for Split {split_idx + 1}, Epoch {epoch}/{training_epoch}: RMSE={test_rmse:.4f}, Recall={test_recall:.2f}%, MAP={test_map:.4f}')
                print(
                    f'High Test Results: RMSE={high_test_rmse:.4f}, Recall={high_test_recall:.2f}%, MAP={high_test_map:.4f}',
                    flush=True)

            flag = early_stop(val_loss, test_rmse, test_recall, test_map, high_test_rmse, high_test_recall,
                              high_test_map, test_inverse_trans_pre, test_inverse_trans_label)
            if flag:
                torch.save(net, data_type + '_full_model.pth')
                print(f"Model saved in Split {split_idx + 1}, Epoch {epoch}", flush=True)

            if early_stop.early_stop:
                print(f"Early Stopping in global step {global_step}, Split {split_idx + 1}, Epoch {epoch}", flush=True)
                print('Best Test RMSE: {:.4f}, Best Test Recall: {:.2f}%, Best Test MAP: {:.4f}'.format(
                    early_stop.best_rmse, early_stop.best_recall, early_stop.best_map))
                print('Best High Test RMSE: {:.4f}, Best High Test Recall: {:.2f}%, Best High Test MAP: {:.4f}'.format(
                    early_stop.best_high_rmse, early_stop.best_high_recall, early_stop.best_high_map))
                break

        # 在每个 split 结束后，将当前 split 的数据添加到经验池
        print(f"\nAdding samples from Split {split_idx + 1} to experience replay buffer...\n")
        for batch in zip(train_loaders[split_idx][0], train_loaders[split_idx][1], train_loaders[split_idx][2],
                         train_loaders[split_idx][3]):
            train_feature = []
            target_time = []
            graph_feature = []
            train_label = []
            for i in range(4):
                t_train_feature, t_target_time, t_graph_feature, t_train_label = batch[i]
                t_train_feature, t_target_time, t_graph_feature, t_train_label = t_train_feature.to(device), \
                    t_target_time.to(device), t_graph_feature.to(device), t_train_label.to(device)
                train_feature.append(t_train_feature)
                target_time.append(t_target_time)
                graph_feature.append(t_graph_feature)
                train_label.append(t_train_label)

            # 计算损失以作为优先级
            with torch.no_grad():
                final_output, classification_output, consistency_loss = net(
                    train_feature, target_time, graph_feature, road_adj, risk_adj, poi_adj, sum_adj, grid_node_map,
                    trans)
                loss = mask_loss(final_output, classification_output, train_label, risk_mask, bfc,
                                 data_type) + consistency_loss
                priority = loss.cpu().item()
            train_feature_cpu = [t.cpu() for t in train_feature]
            target_time_cpu = [t.cpu() for t in target_time]
            graph_feature_cpu = [t.cpu() for t in graph_feature]
            train_label_cpu = [t.cpu() for t in train_label]

            replay_buffer.add((
                train_feature_cpu,
                target_time_cpu,
                graph_feature_cpu,
                train_label_cpu
            ), priority)

        print(f"Experience replay buffer size: {replay_buffer.current_capacity()}\n")

        # 在每个 split 结束后记录结果
        results.append({
            'split': split_idx + 1,
            'best_test_rmse': early_stop.best_rmse,
            'best_test_recall': early_stop.best_recall,
            'best_test_map': early_stop.best_map,
            'best_high_test_rmse': early_stop.best_high_rmse,
            'best_high_test_recall': early_stop.best_high_recall,
            'best_high_test_map': early_stop.best_high_map
        })

        early_stop.reset()

    # 在所有 split 结束后输出结果
    print("\nFinal Results on Each Split:")
    for result in results:
        print(
            "Split: {split}, Best RMSE: {best_test_rmse:.4f}, Best Recall: {best_test_recall:.2f}%, Best MAP: {best_test_map:.4f}, "
            "Best High RMSE: {best_high_test_rmse:.4f}, Best High Recall: {best_high_test_recall:.2f}%, Best High MAP: {best_high_test_map:.4f}".format(
                **result), flush=True)

    return early_stop.best_rmse, early_stop.best_recall, early_stop.best_map



def training_continual_dual(net,
                       training_epoch,
                       train_loaders,
                       val_loaders,
                       test_loaders,
                       high_test_loaders,
                       road_adj,
                       risk_adj,
                       poi_adj,
                       sum_adj,
                       risk_mask,
                       grid_node_map,
                       trainer,
                       early_stop,
                       device,
                       scaler,
                       trans,
                       bfc,
                       data_type='nyc'):
    global_step = 1
    results = []
    epoch_times = []  # List to store the time for each epoch

    # Initialize the Dual-Memory Experience Replay Buffer
    stm_capacity = 30  # Capacity for Short-Term Memory
    ltm_capacity = 60  # Capacity for Long-Term Memory
    replay_buffer = DualMemoryExperienceReplayBuffer(stm_capacity, ltm_capacity)
    alpha, beta = 0.6, 0.4  # Parameters for prioritization and importance sampling

    for split_idx in range(len(train_loaders)):
        print(f"\nStarting training for Split {split_idx + 1}...\n")

        split_iterator = tqdm(range(1, training_epoch + 1), desc=f"Split {split_idx + 1}", unit="epoch")

        for epoch in split_iterator:
            epoch_start_time = time.time()  # Start timing for the current epoch
            net.train()

            # Experience Replay from STM and LTM
            if replay_buffer.current_capacity()[0] > 0 or replay_buffer.current_capacity()[1] > 0:
                # Decide the batch size and STM ratio
                replay_batch_size = 16
                stm_ratio = 0.75  # Adjust the ratio as needed
                replay_samples, replay_indices, replay_weights = replay_buffer.sample(replay_batch_size, beta=beta, stm_ratio=stm_ratio)

                # Process the sampled experiences
                for experience, (buffer_type, idx), weight in zip(replay_samples, replay_indices, replay_weights):
                    # Unpack experience
                    train_feature, target_time, graph_feature, train_label = experience
                    train_feature = [t.to(device) for t in train_feature]
                    target_time = [t.to(device) for t in target_time]
                    graph_feature = [t.to(device) for t in graph_feature]
                    train_label = [t.to(device) for t in train_label]

                    # Forward pass
                    final_output, classification_output, consistency_loss = net(
                        train_feature, target_time, graph_feature, road_adj, risk_adj, poi_adj, sum_adj, grid_node_map, trans)
                    replay_loss = mask_loss(final_output, classification_output, train_label, risk_mask, bfc, data_type)

                    # Convert weight to tensor
                    weight = torch.tensor(weight, dtype=torch.float32, device=device)

                    # Compute weighted loss
                    loss = replay_loss * weight

                    trainer.zero_grad()
                    loss.backward()
                    trainer.step()

                    # Update priorities in the buffer
                    error = loss.detach().cpu().item()
                    replay_buffer.update_priorities([(buffer_type, idx)], [error])

            # Training on current split data
            epoch_iterator = tqdm(zip(train_loaders[split_idx][0],
                                  train_loaders[split_idx][1],
                                  train_loaders[split_idx][2],
                                  train_loaders[split_idx][3]),
                              desc=f"Epoch {epoch}/{training_epoch}",
                              unit="batch", leave=False)

            for batch_1, batch_2, batch_3, batch_4 in epoch_iterator:
                batch = [batch_1, batch_2, batch_3, batch_4]
                train_feature = []
                target_time = []
                graph_feature = []
                train_label = []
                for i in range(4):
                    t_train_feature, t_target_time, t_graph_feature, t_train_label = batch[i]
                    t_train_feature, t_target_time, t_graph_feature, t_train_label = t_train_feature.to(device), \
                        t_target_time.to(device), t_graph_feature.to(device), t_train_label.to(device)
                    train_feature.append(t_train_feature)
                    target_time.append(t_target_time)
                    graph_feature.append(t_graph_feature)
                    train_label.append(t_train_label)

                # Forward pass
                final_output, classification_output, consistency_loss = net(train_feature, target_time, graph_feature,
                                                                            road_adj, risk_adj, poi_adj, sum_adj, grid_node_map, trans)
                l = mask_loss(final_output, classification_output, train_label, risk_mask, bfc, data_type)

                trainer.zero_grad()
                l.backward()
                trainer.step()
                training_loss = l.cpu().item()

                epoch_iterator.set_postfix({"Training Loss": f"{training_loss:.6f}"})

                # Add experiences to STM after training each batch
                # Move data to CPU to save GPU memory
                # train_feature_cpu = [t.cpu() for t in train_feature]
                # target_time_cpu = [t.cpu() for t in target_time]
                # graph_feature_cpu = [t.cpu() for t in graph_feature]
                # train_label_cpu = [t.cpu() for t in train_label]

                # Calculate individual errors for priority (assuming batch size of 1)
                # error = training_loss  # Using the training loss as error for simplicity
                # replay_buffer.add((
                #     train_feature_cpu,
                #     target_time_cpu,
                #     graph_feature_cpu,
                #     train_label_cpu
                # ), error, to_ltm=False)  # Add to STM

                global_step += 1

            # Validate after each epoch
            val_loss = compute_loss(net, val_loaders[split_idx], risk_mask, road_adj, risk_adj, poi_adj, sum_adj,
                                    grid_node_map, trans, device, bfc, data_type)
            split_iterator.set_postfix({"Validation Loss": f"{val_loss:.6f}"})

            if epoch == 1 or val_loss < early_stop.best_score:
                test_rmse, test_recall, test_map, test_inverse_trans_pre, test_inverse_trans_label = \
                    predict_and_evaluate(net, test_loaders[split_idx], risk_mask, road_adj, risk_adj, poi_adj, sum_adj,
                                         grid_node_map, trans, scaler, device)

                high_test_rmse, high_test_recall, high_test_map, _, _ = \
                    predict_and_evaluate(net, high_test_loaders[split_idx], risk_mask, road_adj, risk_adj, poi_adj, sum_adj,
                                         grid_node_map, trans, scaler, device)

                print(
                    f'Test Results for Split {split_idx + 1}, Epoch {epoch}/{training_epoch}: RMSE={test_rmse:.4f}, Recall={test_recall:.2f}%, MAP={test_map:.4f}')
                print(
                    f'High Test Results: RMSE={high_test_rmse:.4f}, Recall={high_test_recall:.2f}%, MAP={high_test_map:.4f}',
                    flush=True)

            flag = early_stop(val_loss, test_rmse, test_recall, test_map, high_test_rmse, high_test_recall,
                              high_test_map, test_inverse_trans_pre, test_inverse_trans_label)
            if flag:
                torch.save(net, data_type + '_full_model.pth')
                print(f"Model saved in Split {split_idx + 1}, Epoch {epoch}", flush=True)

            if early_stop.early_stop:
                print(f"Early Stopping in global step {global_step}, Split {split_idx + 1}, Epoch {epoch}", flush=True)
                print('Best Test RMSE: {:.4f}, Best Test Recall: {:.2f}%, Best Test MAP: {:.4f}'.format(
                    early_stop.best_rmse, early_stop.best_recall, early_stop.best_map))
                print('Best High Test RMSE: {:.4f}, Best High Test Recall: {:.2f}%, Best High Test MAP: {:.4f}'.format(
                    early_stop.best_high_rmse, early_stop.best_high_recall, early_stop.best_high_map))
                break

            # Record the time taken for the current epoch
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            epoch_times.append(epoch_time)  # Store the epoch time

            # Every 10 epochs, calculate and print average time
            if epoch % 10 == 0:
                average_epoch_time = sum(epoch_times[-10:]) / 10
                print(f"\nAverage time for last 10 epochs: {average_epoch_time:.2f} seconds")

        # After each split, transfer important experiences from STM to LTM
        num_experiences_to_transfer = 15  # Adjust as needed
        replay_buffer.transfer_to_ltm(num_experiences_to_transfer)
        replay_buffer.clear_stm()

        print(f"****STM size: {replay_buffer.current_capacity()[0]}, LTM size: {replay_buffer.current_capacity()[1]}\n****")

        # Record results after each split
        results.append({
            'split': split_idx + 1,
            'best_test_rmse': early_stop.best_rmse,
            'best_test_recall': early_stop.best_recall,
            'best_test_map': early_stop.best_map,
            'best_high_test_rmse': early_stop.best_high_rmse,
            'best_high_test_recall': early_stop.best_high_recall,
            'best_high_test_map': early_stop.best_high_map
        })

        early_stop.reset()

    # Output results after all splits
    print("\nFinal Results on Each Split:")
    for result in results:
        print(
            "Split: {split}, Best RMSE: {best_test_rmse:.4f}, Best Recall: {best_test_recall:.2f}%, Best MAP: {best_test_map:.4f}, "
            "Best High RMSE: {best_high_test_rmse:.4f}, Best High Recall: {best_high_test_recall:.2f}%, Best High MAP: {best_high_test_map:.4f}".format(
                **result), flush=True)

    return early_stop.best_rmse, early_stop.best_recall, early_stop.best_map

def main(config):
    patience, delta, train_rate, valid_rate, recent_prior, week_prior, one_day_period, days_of_week, pre_len, \
        seq_len, training_epoch, batch_size, learning_rate, num_of_transformer_layers, transformer_hidden_size, \
        gcn_num_filter, remote_sensing_image_path, \
        all_data_filename, mask_filename, road_adj_filename, risk_adj_filename, poi_adj_filename, sum_adj_filename, grid_node_filename, \
        north_south_map, west_east_map, trans_filename, bfc_filename, num_of_heads, augment_channel = loadConfig(config)

    bfc = get_trans(bfc_filename)
    nums_of_filter = []
    for _ in range(2):
        nums_of_filter.append(gcn_num_filter)

    scaler = []
    train_data_shape = []
    time_shape = []
    graph_feature_shape = []
    high_test_loader = []
    train_loader = []
    val_loader = []
    test_loader = []
    grid_node_map = []

    splits = [0.3, 0.175, 0.175, 0.175, 0.175]

    for i in range(4):
        grid_node_map.append(get_grid_node_map_maxtrix(grid_node_filename[i]))
        t_scaler, t_train_data_shape, t_time_shape, t_graph_feature_shape, t_train_loader, t_val_loader, t_test_loader, t_high_test_loader = \
            generate_dataloader_continual(
                all_data_filename=all_data_filename[i],
                grid_node_data_filename=grid_node_filename[i],
                batch_size=batch_size,
                splits=splits,
                recent_prior=recent_prior,
                week_prior=week_prior,
                one_day_period=one_day_period,
                days_of_week=days_of_week,
                pre_len=pre_len,
                test=args.test,
                north_south_map=north_south_map[i],
                west_east_map=west_east_map[i]
            )
        scaler.append(t_scaler)
        train_data_shape.append(t_train_data_shape)
        time_shape.append(t_time_shape)
        graph_feature_shape.append(t_graph_feature_shape)
        high_test_loader.append(t_high_test_loader)

        train_loader.append(t_train_loader)
        val_loader.append(t_val_loader)
        test_loader.append(t_test_loader)

    train_loader = [list(row) for row in zip(*train_loader)]
    val_loader = [list(row) for row in zip(*val_loader)]
    test_loader = [list(row) for row in zip(*test_loader)]
    high_test_loader = [list(row) for row in zip(*high_test_loader)]

    trans = []

    for i in trans_filename:
        trans.append(get_trans(i))
    if not args.nors:
        rsdataloader = get_remote_sensing_dataloader('std', 256, remote_sensing_image_path)
        remote_sensing_data = next(iter(rsdataloader)).get('image').to(device)
    else:
        remote_sensing_data = None

    model = MGHSTN(train_data_shape[0][2], num_of_transformer_layers, seq_len, pre_len, transformer_hidden_size,
                   time_shape[0][1], graph_feature_shape[0][2], nums_of_filter, north_south_map, west_east_map,
                   args.nors, remote_sensing_data, num_of_heads, augment_channel)

    model.to(device)
    print(model)

    num_of_parameters = 0
    for name, parameters in model.named_parameters():
        num_of_parameters += np.prod(parameters.shape)
    print("Number of Parameters: {}".format(num_of_parameters), flush=True)

    trainer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stop = EarlyStopping(patience=patience, delta=delta)

    risk_mask = []
    road_adj = []
    risk_adj = []
    poi_adj = []
    sum_adj = []
    for i in range(4):
        risk_mask.append(get_mask(mask_filename[i]))
        road_adj.append(get_adjacent(road_adj_filename[i]))
        risk_adj.append(get_adjacent(risk_adj_filename[i]))
        if poi_adj_filename[0] == "":
            poi_adj.append(None)
        else:
            poi_adj.append(get_adjacent(poi_adj_filename[i]))
        sum_adj.append(get_adjacent(sum_adj_filename[i]))

    for i in range(len(trans)):
        trans[i] = torch.from_numpy(trans[i]).unsqueeze(0).to(device)
    bfc = torch.from_numpy(bfc).unsqueeze(0).to(torch.float32).to(device)

    if args.infer:
        model = torch.load("1" + config['data_type'] + "_full_model.pth")
        test_rmse, test_recall, test_map, test_inverse_trans_pre, test_inverse_trans_label = \
            predict_and_evaluate(model, test_loader, risk_mask, road_adj, risk_adj, poi_adj,
                                 grid_node_map, trans, scaler, device)

        high_test_rmse, high_test_recall, high_test_map, _, _ = \
            predict_and_evaluate(model, high_test_loader, risk_mask, road_adj, risk_adj, poi_adj,
                                 grid_node_map, trans, scaler, device)

        print('test RMSE: %.4f,test Recall: %.2f%%,test MAP: %.4f,'
              'high test RMSE: %.4f,high test Recall: %.2f%%,high test MAP: %.4f'
              % (test_rmse, test_recall, test_map, high_test_rmse, high_test_recall,
                 high_test_map), flush=True)
    else:
        training_continual_dual(
            model,
            training_epoch,
            train_loader,
            val_loader,
            test_loader,
            high_test_loader,
            road_adj,
            risk_adj,
            poi_adj,
            sum_adj,
            risk_mask,
            grid_node_map,
            trainer,
            early_stop,
            device,
            scaler,
            trans,
            bfc,
            data_type=config['data_type'],
        )


if __name__ == "__main__":
    # python train.py --config config/nyc/NYC_Config.json --gpus 0
    main(config)
