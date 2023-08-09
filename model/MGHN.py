import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import ImageEncoder
from torch import Tensor

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


# class SC_Attention(nn.Module):
#     def __init__(self, in_channels):
#         super(SC_Attention, self).__init__()
#
#         self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
#
#         self.softmax = nn.Softmax(dim=-1)
#
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#         self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1,
#                               kernel_size=2, stride=2, padding=0)
#
#     def forward(self, x):
#         # x.shape: (batch_size, in_channels, 20, 20)
#
#         batch_size, in_channels, height, width = x.size()
#
#         # Project the input into the query, key, and value feature spaces
#         proj_query = self.query_conv(x)  # (batch_size, in_channels//8, 20, 20)
#         proj_query = proj_query.view(batch_size, -1, height * width).permute(0, 2,
#                                                                              1)  # (batch_size, 20*20, in_channels//8)
#
#         proj_key = self.key_conv(x)  # (batch_size, in_channels//8, 20, 20)
#         proj_key = proj_key.view(batch_size, -1, height * width)  # (batch_size, in_channels//8, 20*20)
#
#         proj_value = self.value_conv(x)  # (batch_size, in_channels, 20, 20)
#         proj_value = proj_value.view(batch_size, -1, height * width)  # (batch_size, in_channels, 20*20)
#
#         # Compute the dot product of the query and key for each pixel
#         energy = torch.bmm(proj_query, proj_key)  # (batch_size, 20*20, 20*20)
#         attention = self.softmax(energy)  # (batch_size, 20*20, 20*20)
#
#         # Compute the weighted sum of the values
#         proj_out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # (batch_size, in_channels, 20*20)
#         proj_out = proj_out.view(batch_size, -1, height, width)  # (batch_size, in_channels, 20, 20)
#         out = self.gamma * proj_out + x
#
#         # Apply a 2D convolution to reduce the spatial dimensions of the output
#         out = self.conv(out)
#
#         return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term1 = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        div_term2 = torch.exp(torch.arange(0, d_model - 1, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term1)
        if d_model % 2 == 1:
            pe[:, 0, 1::2] = torch.cos(position * div_term2)
        else:
            pe[:, 0, 1::2] = torch.cos(position * div_term1)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class GCN_Layer(nn.Module):
    def __init__(self, num_of_features, num_of_filter):
        """One layer of GCN
        
        Arguments:
            num_of_features {int} -- the dimension of node feature
            num_of_filter {int} -- the number of graph filters
        """
        super(GCN_Layer, self).__init__()
        self.gcn_layer = nn.Sequential(
            nn.Linear(in_features=num_of_features,
                      out_features=num_of_filter),
            nn.ReLU()
        )

    def forward(self, input, adj):
        """
        Arguments:
            input {Tensor} -- signal matrix,shape (batch_size,N,T*D)
            adj {np.array} -- adjacent matrix，shape (N,N)
        Returns:
            {Tensor} -- output,shape (batch_size,N,num_of_filter)
        """
        batch_size, _, _ = input.shape # 224，197，3
        adj = torch.from_numpy(adj).to(input.device)
        adj = adj.repeat(batch_size, 1, 1) # 224，197，197
        input = torch.bmm(adj, input) # 224，197，3
        output = self.gcn_layer(input) # 224，197，64
        return output


# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)
#
#
# class SEBlock(nn.Module):
#     def __init__(self, in_features):
#         super(SEBlock, self).__init__()
#
#         conv_block = [nn.Conv2d(in_features, in_features, 3, 1, 1),
#                       nn.BatchNorm2d(in_features),
#                       nn.ReLU(inplace=True),
#                       nn.Conv2d(in_features, in_features, 3, 1, 1),
#                       nn.BatchNorm2d(in_features),
#                       nn.ReLU(),
#                       ]
#
#         self.se = SELayer(in_features)
#         self.conv_block = nn.Sequential(*conv_block)
#
#     def forward(self, x):
#         out = self.conv_block(x)
#         out = self.se(out)
#         return x + out

class STModule(nn.Module):
    def __init__(self, grid_in_channel, num_of_transformer_layers, seq_len,
                 transformer_hidden_size, num_of_target_time_feature, num_of_heads):
        """[summary]
        
        Arguments:
            grid_in_channel {int} -- the number of grid data feature (batch_size,T,D,W,H),grid_in_channel=D
            num_of_transformer_layers {int} -- the number of GRU layers
            seq_len {int} -- the time length of input
            transformer_hidden_size {int} -- the hidden size of GRU
            num_of_target_time_feature {int} -- the number of target time feature，为24(hour)+7(week)+1(holiday)=32
        """
        super(STModule, self).__init__()
        # self.SE = SEBlock(grid_in_channel)
        self.grid_conv = nn.Sequential(
            nn.Conv2d(in_channels=grid_in_channel, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=grid_in_channel, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # self.grid_gru = nn.GRU(grid_in_channel, transformer_hidden_size, num_of_transformer_layers, batch_first=True)
        # TransformerEncoder layer
        self.positional_encoding = PositionalEncoding(d_model=grid_in_channel, dropout=0.1)  # positional encoding
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=grid_in_channel, nhead=num_of_heads, dim_feedforward=transformer_hidden_size*4),
            num_layers=num_of_transformer_layers)  # Transformer encoder
        self.fc0 = nn.Linear(in_features=grid_in_channel, out_features=transformer_hidden_size)

        self.grid_att_fc1 = nn.Linear(in_features=transformer_hidden_size, out_features=1)
        self.grid_att_fc2 = nn.Linear(in_features=num_of_target_time_feature, out_features=seq_len)
        self.grid_att_bias = nn.Parameter(torch.zeros(1))
        self.grid_att_softmax = nn.Softmax(dim=-1)


        # self.abLinear = nn.Linear(7, 1)

    def forward(self, grid_input, target_time_feature):
        """
        Arguments:
            grid_input {Tensor} -- grid input，shape：(batch_size,seq_len,D,W,H)
            target_time_feature {Tensor} -- the feature of target time，shape：(batch_size,num_target_time_feature)
        Returns:
            {Tensor} -- shape：(batch_size,hidden_size,W,H)
        """
        batch_size, T, D, W, H = grid_input.shape

        grid_input = grid_input.view(-1, D, W, H)
        conv_output = self.grid_conv(grid_input)
        # conv_output = self.SE(grid_input)

        conv_output = conv_output.view(batch_size, -1, D, W, H) \
            .permute(0, 3, 4, 1, 2) \
            .contiguous() \
            .view(-1, T, D)
        # gru_output, _ = self.grid_gru(conv_output)

        x = conv_output.permute(1, 0, 2)  # 把批次大小放在第二个维度上
        x = self.positional_encoding(x)  # 加上位置编码
        tr_output = self.transformer_encoder(x)  # 通过Transformer编码器
        tr_output = tr_output.permute(1, 0, 2)  # 把批次大小放在第一个维度上
        tr_output = self.fc0(tr_output)  # 把维度转换为hidden_size

        grid_target_time = torch.unsqueeze(target_time_feature, 1).repeat(1, W * H, 1).view(batch_size * W * H, -1)
        grid_att_fc1_output = torch.squeeze(self.grid_att_fc1(tr_output))
        grid_att_fc2_output = self.grid_att_fc2(grid_target_time)
        grid_att_score = self.grid_att_softmax(F.relu(grid_att_fc1_output + grid_att_fc2_output + self.grid_att_bias))
        grid_att_score = grid_att_score.view(batch_size * W * H, -1, 1)
        grid_output = torch.sum(tr_output * grid_att_score, dim=1)

        grid_output = grid_output.view(batch_size, W, H, -1).permute(0, 3, 1, 2).contiguous()

        return grid_output

        # for ablation test
        # conv_output = self.abLinear(conv_output.view(batch_size, W, H, D, -1))
        # conv_output = self.fc0(conv_output.view(batch_size, W, H, -1))
        # conv_output = conv_output.permute(0, 3, 1, 2).contiguous()
        #
        # return conv_output


class GPModule(nn.Module):
    def __init__(self, num_of_graph_feature, nums_of_graph_filters, north_south_map, west_east_map):
        """
        Arguments:
            num_of_graph_feature {int} -- the number of graph node feature，(batch_size,seq_len,D,N),num_of_graph_feature=D
            nums_of_graph_filters {list} -- the number of GCN output feature
            seq_len {int} -- the time length of input
            num_of_transformer_layers {int} -- the number of GRU layers
            transformer_hidden_size {int} -- the hidden size of GRU
            num_of_target_time_feature {int} -- the number of target time feature，为24(hour)+7(week)+1(holiday)=32
            north_south_map {int} -- the weight of grid data
            west_east_map {int} -- the height of grid data

        """
        super(GPModule, self).__init__()
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map
        self.road_gcn = nn.ModuleList()
        for idx, num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.road_gcn.append(GCN_Layer(num_of_graph_feature, num_of_filter))
            else:
                self.road_gcn.append(GCN_Layer(nums_of_graph_filters[idx - 1], num_of_filter))

        self.risk_gcn = nn.ModuleList()
        for idx, num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.risk_gcn.append(GCN_Layer(num_of_graph_feature, num_of_filter))
            else:
                self.risk_gcn.append(GCN_Layer(nums_of_graph_filters[idx - 1], num_of_filter))

        self.poi_gcn = nn.ModuleList()
        for idx, num_of_filter in enumerate(nums_of_graph_filters):
            if idx == 0:
                self.poi_gcn.append(GCN_Layer(num_of_graph_feature, num_of_filter))
            else:
                self.poi_gcn.append(GCN_Layer(nums_of_graph_filters[idx - 1], num_of_filter))

    def forward(self, graph_feature, road_adj, risk_adj, poi_adj):

        """
        Arguments:
            graph_feature {Tensor} -- Graph signal matrix，(batch_size,T,D1,N)
            road_adj {np.array} -- road adjacent matrix，shape：(N,N)
            risk_adj {np.array} -- risk adjacent matrix，shape：(N,N)
            poi_adj {np.array} -- poi adjacent matrix，shape：(N,N)
            target_time_feature {Tensor} -- the feature of target time，shape：(batch_size,num_target_time_feature)
            grid_node_map {np.array} -- map graph data to grid data,shape (W*H,N)
        Returns:
            {Tensor} -- shape：(batch_size,pre_len,north_south_map,west_east_map)
        """
        batch_size, T, D1, N = graph_feature.shape

        # shape(batch_size*T,f_N,channel)=(112,243,3)
        road_graph_output = graph_feature.view(-1, D1, N).permute(0, 2, 1).contiguous()
        for gcn_layer in self.road_gcn:
            road_graph_output = gcn_layer(road_graph_output, road_adj)

        risk_graph_output = graph_feature.view(-1, D1, N).permute(0, 2, 1).contiguous()
        for gcn_layer in self.risk_gcn:
            risk_graph_output = gcn_layer(risk_graph_output, risk_adj)

        graph_output = road_graph_output + risk_graph_output

        if poi_adj is not None:
            poi_graph_output = graph_feature.view(-1, D1, N).permute(0, 2, 1).contiguous()
            for gcn_layer in self.poi_gcn:
                poi_graph_output = gcn_layer(poi_graph_output, poi_adj)
            graph_output += poi_graph_output

        graph_output = graph_output.view(batch_size, T, N, -1) \
            .permute(0, 2, 1, 3) \
            .contiguous() \
            .view(batch_size * N, T, -1) \
            .view(batch_size, T, -1, N)
        # print(graph_output.shape)
        # exit(0)
        # graph_output, _ = self.graph_gru(graph_output)

        # torch.Size([32, 7, 64, 197])
        return graph_output


class SGModule(nn.Module):
    def __init__(self, seq_len, num_of_transformer_layers, transformer_hidden_size, num_of_target_time_feature,
                 north_south_map, west_east_map, num_of_heads):
        super(SGModule, self).__init__()
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map
        # self.graph_gru = nn.GRU(64, transformer_hidden_size, num_of_transformer_layers, batch_first=True)
        # 转换后TransformerEncoder层（仅供参考）
        self.positional_encoding = PositionalEncoding(d_model=64, dropout=0.1)  # 位置编码
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=num_of_heads, dim_feedforward=transformer_hidden_size * 4),
            num_layers=num_of_transformer_layers + 1)  # Transformer编码器
        self.fc0 = nn.Linear(in_features=64, out_features=transformer_hidden_size)

        self.graph_att_fc1 = nn.Linear(in_features=transformer_hidden_size, out_features=1)
        self.graph_att_fc2 = nn.Linear(in_features=num_of_target_time_feature, out_features=seq_len)
        self.graph_att_bias = nn.Parameter(torch.zeros(1))
        self.graph_att_softmax = nn.Softmax(dim=-1)


        # self.abLinear = nn.Linear(7, 1)

    def forward(self, graph_output, target_time_feature, grid_node_map):
        batch_size, T, _, N = graph_output.shape

        graph_output = graph_output.view(batch_size, T, N, -1) \
            .permute(0, 2, 1, 3) \
            .contiguous() \
            .view(batch_size * N, T, -1)
        # graph_output, _ = self.graph_gru(graph_output)

        graph_output = graph_output.view(batch_size * N, T, -1) # （32*197, 7, 64）
        # 转换后TransformerEncoder层（仅供参考）
        x = graph_output.permute(1, 0, 2)  # 把批次大小放在第二个维度上
        x = self.positional_encoding(x)  # 加上位置编码
        graph_output = self.transformer_encoder(x)  # 通过Transformer编码器
        graph_output = graph_output.permute(1, 0, 2)  # 把批次大小放在第一个维度上
        graph_output = self.fc0(graph_output)  # 把维度转换为hidden_size

        graph_target_time = torch.unsqueeze(target_time_feature, 1).repeat(1, N, 1).view(batch_size * N, -1)
        graph_att_fc1_output = torch.squeeze(self.graph_att_fc1(graph_output))
        graph_att_fc2_output = self.graph_att_fc2(graph_target_time)
        graph_att_score = self.graph_att_softmax(
            F.relu(graph_att_fc1_output + graph_att_fc2_output + self.graph_att_bias))
        graph_att_score = graph_att_score.view(batch_size * N, -1, 1) # (6304,7,1)
        graph_output = torch.sum(graph_output * graph_att_score, dim=1)
        graph_output = graph_output.view(batch_size, N, -1).contiguous() # (32,197,64)

        # for ablation test
        # graph_output = self.fc0(graph_output)
        # graph_output = self.abLinear(graph_output.view(batch_size, N, T, -1).permute(0, 1, 3, 2).contiguous())
        # graph_output = graph_output.view(batch_size, N, -1).contiguous()


        grid_node_map_tmp = torch.from_numpy(grid_node_map) \
            .to(graph_output.device) \
            .repeat(batch_size, 1, 1)
        graph_output = torch.bmm(grid_node_map_tmp, graph_output) \
            .permute(0, 2, 1) \
            .view(batch_size, -1, self.north_south_map, self.west_east_map)
        return graph_output


class MGHN(nn.Module):
    def __init__(self, grid_in_channel, num_of_transformer_layers, seq_len, pre_len, transformer_hidden_size,
                 num_of_target_time_feature, num_of_graph_feature, nums_of_graph_filters, north_south_map,
                 west_east_map, is_baseline, remote_sensing_data, num_of_heads, augment_channel):
        """[summary]
        
        Arguments:
            grid_in_channel {int} -- the number of grid data feature (batch_size,T,D,W,H),grid_in_channel=D
            num_of_transformer_layers {int} -- the number of GRU layers
            seq_len {int} -- the time length of input
            pre_len {int} -- the time length of prediction
            transformer_hidden_size {int} -- the hidden size of transformer
            num_of_target_time_feature {int} -- the number of target time feature，which is 24(hour)+7(week)+1(holiday)=32
            num_of_graph_feature {int} -- the number of graph node feature，(batch_size,seq_len,D,N),num_of_graph_feature=D
            nums_of_graph_filters {list} -- the number of GCN output feature
            north_south_map {list} -- the weight of grid data
            west_east_map {list} -- the height of grid data
        """
        super(MGHN, self).__init__()
        # self.remote_sensing = remote_sensing
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map
        self.fusion_channel = 16
        self.augment_channel = augment_channel
        self.is_baseline = is_baseline
        # self.fusion_weight = nn.ModuleList([nn.Linear(64, 64) for _ in range(4)])

        if not self.is_baseline:
            self.remote_sensing_data = remote_sensing_data
            self.encoder = ImageEncoder(256 * 256, self.augment_channel)
            # 256,16,16 -> 16
            # self.remote_sensing_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
            # self.remote_sensing_layer = nn.Linear(1 * 16 * 16, self.fusion_channel)
            # self.aff = AFF(self.fusion_channel, 4)

        self.st_module = nn.ModuleList(
            [STModule(grid_in_channel + self.augment_channel, num_of_transformer_layers, seq_len,
                      transformer_hidden_size, num_of_target_time_feature,
                      num_of_heads).cuda(),
             STModule(grid_in_channel, num_of_transformer_layers, seq_len,
                      transformer_hidden_size, num_of_target_time_feature,
                      num_of_heads).cuda(),
             STModule(grid_in_channel, num_of_transformer_layers, seq_len,
                      transformer_hidden_size, num_of_target_time_feature,
                      num_of_heads).cuda(),
             STModule(grid_in_channel, num_of_transformer_layers, seq_len,
                      transformer_hidden_size, num_of_target_time_feature,
                      num_of_heads).cuda()])
        self.gp_module = nn.ModuleList([GPModule(num_of_graph_feature, nums_of_graph_filters,
                                                 north_south_map[0], west_east_map[0]).cuda(),
                                        GPModule(num_of_graph_feature, nums_of_graph_filters,
                                                           north_south_map[1], west_east_map[1]).cuda(),
                                        GPModule(num_of_graph_feature, nums_of_graph_filters,
                                                           north_south_map[2], west_east_map[2]).cuda(),
                                        GPModule(num_of_graph_feature, nums_of_graph_filters,
                                                           north_south_map[3], west_east_map[3]).cuda()])

        self.sg_module = nn.ModuleList([SGModule(seq_len, num_of_transformer_layers,
                                                 transformer_hidden_size,
                                                 num_of_target_time_feature,
                                                 north_south_map[0], west_east_map[0],
                                                 num_of_heads).cuda(),
                                        SGModule(seq_len, num_of_transformer_layers,
                                                            transformer_hidden_size,
                                                            num_of_target_time_feature,
                                                            north_south_map[1], west_east_map[1],
                                                            num_of_heads).cuda(),
                                        SGModule(seq_len, num_of_transformer_layers,
                                                            transformer_hidden_size,
                                                            num_of_target_time_feature,
                                                            north_south_map[2], west_east_map[2],
                                                            num_of_heads).cuda(),
                                        SGModule(seq_len, num_of_transformer_layers,
                                                            transformer_hidden_size,
                                                            num_of_target_time_feature,
                                                            north_south_map[3], west_east_map[3],
                                                            num_of_heads).cuda()])

        self.grid_weight = nn.ModuleList([nn.Conv2d(in_channels=transformer_hidden_size,
                                                    out_channels=self.fusion_channel,
                                                    kernel_size=1).cuda(),
                                          nn.Conv2d(in_channels=transformer_hidden_size,
                                                    out_channels=self.fusion_channel,
                                                    kernel_size=1).cuda(),
                                          nn.Conv2d(in_channels=transformer_hidden_size,
                                                    out_channels=self.fusion_channel,
                                                    kernel_size=1).cuda(),
                                          nn.Conv2d(in_channels=transformer_hidden_size,
                                                    out_channels=self.fusion_channel,
                                                    kernel_size=1).cuda()])

        self.graph_weight = nn.ModuleList([nn.Conv2d(in_channels=transformer_hidden_size,
                                                     out_channels=self.fusion_channel,
                                                     kernel_size=1).cuda(),
                                           nn.Conv2d(in_channels=transformer_hidden_size,
                                                     out_channels=self.fusion_channel,
                                                     kernel_size=1).cuda(),
                                           nn.Conv2d(in_channels=transformer_hidden_size,
                                                     out_channels=self.fusion_channel,
                                                     kernel_size=1).cuda(),
                                           nn.Conv2d(in_channels=transformer_hidden_size,
                                                     out_channels=self.fusion_channel,
                                                     kernel_size=1).cuda()
                                           ])

        self.output_layer = nn.ModuleList([nn.Linear(self.fusion_channel * north_south_map[0] * west_east_map[0],
                                                     pre_len * north_south_map[0] * west_east_map[0]).cuda(),
                                           nn.Linear(self.fusion_channel * north_south_map[1] * west_east_map[1],
                                                     pre_len * north_south_map[1] * west_east_map[1]).cuda(),
                                           nn.Linear(self.fusion_channel * north_south_map[2] * west_east_map[2],
                                                     pre_len * north_south_map[2] * west_east_map[2]).cuda(),
                                           nn.Linear(self.fusion_channel * north_south_map[3] * west_east_map[3],
                                                     pre_len * north_south_map[3] * west_east_map[3]).cuda()])

        # batch,16,20,20
        # self.output_layer = nn.Linear(self.fusion_channel * north_south_map * west_east_map,
        #                               pre_len * north_south_map * west_east_map)

        # attention module transform from fine to coarse
        # self.SC_attention = SC_Attention(1)

    def forward(self, grid_input, target_time_feature, graph_feature,
                road_adj, risk_adj, poi_adj, grid_node_map, trans):
        """
        Arguments:
            grid_input {list of Tensor} -- grid input，shape：(batch_size,T,D,W,H)
            graph_feature {list of Tensor} -- Graph signal matrix，(batch_size,T,D1,N)
            target_time_feature {list of Tensor} -- the feature of target time，shape：(batch_size,num_target_time_feature)
            road_adj {list of np.array} -- road adjacent matrix，shape：(N,N)
            risk_adj {list of np.array} -- risk adjacent matrix，shape：(N,N)
            poi_adj {list of np.array} -- poi adjacent matrix，shape：(N,N)
            grid_node_map {list of np.array} -- map graph data to grid data,shape (W*H,N)
            trans {list of np.array} -- the transformation matrix of graph data，shape：(n_c,n_f)

        Returns:
            {Tensor} -- shape：(batch_size,pre_len,north_south_map,west_east_map)
        """
        batch_size, _, _, _, _ = grid_input[0].shape

        grid_output = []
        graph_output = []
        fusion_output = []
        final_output = []
        classification_output = []

        if not self.is_baseline:
            # # pretrain
            # remote_output = self.remote_sensing_conv(
            #     remote_output.view(256 * 16 * 16, 400).permute(1, 0).view(400, 256, 16, 16))

            # together
            remote_output = self.encoder(self.remote_sensing_data)
            # remote_output = self.remote_sensing_conv(remote_output)
            #
            # remote_output = self.remote_sensing_layer(remote_output.view(400, 1 * 16 * 16))
            # (batch_size, 16, 20, 20)
            remote_output = remote_output.permute(1, 0).view(self.augment_channel, 20, 20).unsqueeze(0).unsqueeze(0) \
                .repeat(batch_size, 7, 1, 1, 1)
            grid_input[0] = torch.cat((grid_input[0], remote_output), dim=2)

            # fusion_output[0] = self.aff(fusion_output[0], remote_output)

        for i in range(4):
            t_grid_output = self.st_module[i](grid_input[i], target_time_feature[i])
            t_graph_output = self.gp_module[i](graph_feature[i], road_adj[i], risk_adj[i], poi_adj[i])
            t_grid_output = self.grid_weight[i](t_grid_output)

            grid_output.append(t_grid_output)
            graph_output.append(t_graph_output)

        # from fine to coarse
        for i in range(4 - 1):
            f_graph_output = graph_output[i]
            c_graph_output = graph_output[i + 1]

            batch_size, T, _, f_N = f_graph_output.shape
            batch_size1, T, _, c_N = c_graph_output.shape

            # coarse to fine
            c_graph_output = c_graph_output.reshape(batch_size1 * T, -1, c_N)
            # cf_out = F.relu(torch.matmul(c_graph_output, trans[i] / 3))
            cf_out = torch.matmul(c_graph_output, trans[i] / 3)

            # cf_out = self.fusion_weight[i](cf_out.reshape(batch_size1 * T, f_N, -1))

            f1_graph_output = f_graph_output + 0.2 * cf_out.reshape(batch_size1, T, -1, f_N)

            # fine to coarse
            f_graph_output = f_graph_output.reshape(batch_size * T, -1, f_N)
            # fc_out = F.relu(torch.matmul(f_graph_output, trans[i].permute(0, 2, 1) / 3))
            fc_out = torch.matmul(f_graph_output, trans[i].permute(0, 2, 1) / 3)

            # fc_out = self.fusion_weight[i+1](fc_out.reshape(batch_size * T, c_N, -1))

            c_graph_output = c_graph_output.reshape(batch_size1, T, -1, c_N)
            c1_graph_output = c_graph_output + 0.8 * fc_out.reshape((batch_size, T, -1, c_N))

            graph_output[i] = f1_graph_output
            graph_output[i + 1] = c1_graph_output

        # # from coarse to fine
        # for i in range(3, 0, -1):
        #     c_graph_output = graph_output[i]
        #     f_graph_output = graph_output[i - 1]
        #
        #     batch_size, T, _, f_N = f_graph_output.shape
        #     batch_size1, T, _, c_N = c_graph_output.shape
        #
        #     # coarse to fine
        #     c_graph_output = c_graph_output.reshape(batch_size1 * T, -1, c_N)
        #     # cf_out = F.relu(torch.matmul(c_graph_output, trans[i] / 3))
        #     cf_out = torch.matmul(c_graph_output, trans[i-1] / 3)
        #     f1_graph_output = f_graph_output + 0.2 * cf_out.reshape(batch_size1, T, -1, f_N)
        #
        #     # fine to coarse
        #     f_graph_output = f_graph_output.reshape(batch_size * T, -1, f_N)
        #     # fc_out = F.relu(torch.matmul(f_graph_output, trans[i].permute(0, 2, 1) / 3))
        #     fc_out = torch.matmul(f_graph_output, trans[i-1].permute(0, 2, 1) / 3)
        #     c_graph_output = c_graph_output.reshape(batch_size1, T, -1, c_N)
        #     c1_graph_output = c_graph_output + 0.8 * fc_out.reshape((batch_size, T, -1, c_N))
        #
        #     graph_output[i] = c1_graph_output
        #     graph_output[i - 1] = f1_graph_output

        for i in range(4):
            graph_output[i] = self.sg_module[i](graph_output[i], target_time_feature[i], grid_node_map[i])
            graph_output[i] = self.graph_weight[i](graph_output[i])
            fusion_output.append(grid_output[i] + graph_output[i])  # 16,20,20

            # for ablation study
            # fusion_output.append(grid_output[i])  # 16,20,20



        for i in range(4):
            fusion_output[i] = fusion_output[i].view(batch_size, -1)
            final_output.append(self.output_layer[i](fusion_output[i])
                                .view(batch_size, -1, self.north_south_map[i], self.west_east_map[i]))


            # classification
            # classification_output = torch.softmax(final_output.view(final_output.shape[0], -1), dim=1)
            # classification_output = torch.sigmoid(final_output.view(final_output.shape[0], -1))
            classification_output.append(torch.relu(final_output[i].view(final_output[i].shape[0], -1)))


        # SC = self.SC_attention(final_output[0])
        #
        # final_output.append(SC)
        # # ranking
        # _, classification_output = torch.sort(final_output.view(final_output.shape[0], -1))
        return final_output, classification_output

# class GpuList:
#     def __init__(self):
#         self.data = torch.empty(0, device='cuda')
#         self.size = 0
#
#     def push_back(self, value):
#         self.data = torch.cat([self.data, value.unsqueeze(0)])
#         self.size += 1
#
#     def insert(self, index, value):
#         if index < 0 or index > self.size:
#             raise IndexError("Index out of range")
#
#         new_data = torch.empty(self.size + 1, device='cuda')
#         new_data[:index] = self.data[:index]
#         new_data[index] = value.unsqueeze(0)
#         new_data[index + 1:] = self.data[index:]
#
#         self.data = new_data
#         self.size += 1
#     def __getitem__(self, index):
#         return self.data[index].squeeze(0)
#
#     def __len__(self):
#         return self.size