import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool as gmp
from torch_scatter import scatter_sum, scatter_mean

# 工具函数: 创建多层全连接网络
def create_mlp(input_dim, layers, dropout_rate, output_dim, batch_norm=False):
    """创建多层感知器，包括LeakyReLU激活和可选的批量归一化."""
    mlp_layers = []
    for i, layer_dim in enumerate(layers):
        if i > 0:  # 第一层之后添加Dropout
            mlp_layers.append(nn.Dropout(dropout_rate))
        mlp_layers.append(nn.Linear(input_dim if i == 0 else layers[i-1], layer_dim))
        if batch_norm:
            mlp_layers.append(nn.BatchNorm1d(layer_dim))  # 添加BatchNorm
        mlp_layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
    mlp_layers.append(nn.Linear(layers[-1], output_dim))
    return nn.Sequential(*mlp_layers)

# 初始化权重的函数
def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

# 药物特征提取模块
class DrugFEM(nn.Module):
    def __init__(self, num_features, gat_dims, output_dim, dropout, heads=1):
        """初始化药物特征提取模块."""
        super(DrugFEM, self).__init__()
        self.conv_layers = nn.ModuleList()
        for i, dim in enumerate(gat_dims):
            self.conv_layers.append(
                GATConv(num_features if i == 0 else gat_dims[i - 1], dim, heads=heads, dropout=dropout)
            )
        self.aagam = AAGAM(gat_dims[-1])  # 自适应注意力机制图聚合模块
        self.fc_g = create_mlp(output_dim, [output_dim // 2], dropout, output_dim, batch_norm=True)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x, edge_index, batch):
        for conv in self.conv_layers:
            x = self.leaky_relu(conv(x, edge_index))
        x = self.aagam(x, batch)  # 应用自适应注意力机制图聚合模块
        x = self.fc_g(x)
        return x

# 主模型 MFSynDCP
class MFSynDCP(nn.Module):
    def __init__(self, n_output=2, gat_dims=(32, 64, 128),
                 num_features_xd=78, num_features_xt=954, output_dim=128,
                 dropout=0.1, num_mfic_layers=2, heads=1):
        super(MFSynDCP, self).__init__()

        # 初始化两个药物特征提取模块
        self.drug1_fem = DrugFEM(num_features_xd, gat_dims, output_dim, dropout, heads)
        self.drug2_fem = DrugFEM(num_features_xd, gat_dims, output_dim, dropout, heads)

        self.dropout = nn.Dropout(dropout)

        # 细胞系特征提取模块
        self.cell_fem = create_mlp(num_features_xt, [output_dim * 2, output_dim], dropout, output_dim, batch_norm=True)

        # 多源特征交互学习控制器
        self.mfic = MFIC(num_mfic_layers, 3 * output_dim, dropout)

        # 协同效应预测模块
        self.synergy_pm = nn.Sequential(
            nn.Linear(3 * output_dim, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(128, n_output)
        )

        # 权重初始化
        self.apply(weights_init)

    def forward(self, data1, data2):
        x1 = self.drug1_fem(data1.x, data1.edge_index, data1.batch)
        x2 = self.drug2_fem(data2.x, data2.edge_index, data2.batch)

        cell_vector = F.normalize(data1.cell, p=2, dim=1)  # L2归一化
        cell_vector = self.cell_fem(cell_vector)

        xc = torch.cat((x1, x2, cell_vector), 1)
        xc = self.mfic(xc)  # 应用MFIC

        out = self.synergy_pm(xc)
        return out

# 自适应注意力机制图聚合模块
class AAGAM(nn.Module):
    def __init__(self, in_dim):
        super(AAGAM, self).__init__()
        self.attention_weights = nn.Linear(in_dim, 1)  # 计算注意力分数

    def forward(self, x, batch):
        attn_score = self.attention_weights(x)  # 计算每个节点的注意力分数
        attn_score = F.softmax(attn_score, dim=0)  # softmax归一化

        x = attn_score * x  # 加权节点特征
        x = scatter_sum(x, batch, dim=0)  # 聚合到图级别

        return x

# 多源特征交互学习控制器
class MFIC(nn.Module):
    def __init__(self, num_layers, input_size, dropout_rate):
        """初始化多源特征交互学习控制器."""
        super(MFIC, self).__init__()
        self.num_layers = num_layers
        self.non_linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        for i in range(self.num_layers):
            gate = torch.sigmoid(self.gate[i](x))
            non_linear = self.leaky_relu(self.non_linear[i](x))
            linear = self.linear[i](x)
            x = gate * non_linear + (1 - gate) * linear
            x = self.dropout(x)
        return x
