import csv
from itertools import islice

import pandas as pd
import numpy as np
from rdkit import Chem
import networkx as nx
from utils_test import *

# 获取细胞特征
def get_cell_feature(cellId, cell_features):
    for row in islice(cell_features, 0, None):
        if row[0] == cellId:
            return row[1: ]

# 生成原子特征
def atom_features(atom):
    # 返回原子的特征向量，包括原子类型、度数、氢原子数、隐式价和是否为芳香族的特征
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +      # 原子类型
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +     # 原子的度数
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +       # 原子的氢原子数
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +  # 原子的隐式价
                    [atom.GetIsAromatic()])         # 原子是否为芳香族

# one-hot编码函数
def one_of_k_encoding(x, allowable_set):
    # 对输入x进行one-hot编码，只有在allowable_set中的元素会被编码
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

# 对未知元素进行one-hot编码
def one_of_k_encoding_unk(x, allowable_set):
    """对不在allowable_set中的输入x编码为最后一个元素."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

# 将SMILES字符串转换为图
def smile_to_graph(smile):
    # 使用RDKit将SMILES字符串转换为分子，并提取原子和化学键信息
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index

# 创建数据集
def creat_data(datafile, cellfile):
    # 读取细胞特征文件
    file2 = cellfile
    cell_features = []
    with open(file2) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            cell_features.append(row)
    cell_features = np.array(cell_features)
    print('cell_features', cell_features)

    # 读取药物分子的SMILES字符串
    compound_iso_smiles = []
    df = pd.read_csv('data/smiles.csv')
    compound_iso_smiles += list(df['smile'])
    compound_iso_smiles = set(compound_iso_smiles)
    smile_graph = {}
    print('compound_iso_smiles', compound_iso_smiles)
    for smile in compound_iso_smiles:
        print('smiles', smile)
        g = smile_to_graph(smile)
        smile_graph[smile] = g

    datasets = datafile
    # 将数据转换为PyTorch Geometric格式
    processed_data_file_train = 'data/processed/' + datasets + '_train.pt'

    if ((not os.path.isfile(processed_data_file_train))):
        df = pd.read_csv('data/' + datasets + '.csv')
        # 读取药物、细胞、标签等数据
        drug1, drug2, cell, label = list(df['drug1']), list(df['drug2']), list(df['cell']), list(df['label'])
        drug1, drug2, cell, label = np.asarray(drug1), np.asarray(drug2), np.asarray(cell), np.asarray(label)

        print('开始创建数据')
        TestbedDataset(root='data', dataset=datafile + '_drug1', xd=drug1, xt=cell, xt_featrue=cell_features, y=label,smile_graph=smile_graph)
        TestbedDataset(root='data', dataset=datafile + '_drug2', xd=drug2, xt=cell, xt_featrue=cell_features, y=label,smile_graph=smile_graph)
        print('创建数据成功')


if __name__ == "__main__":
    # 数据集和细胞特征文件路径
    cellfile = 'data/cell_features.csv'
    da = ['labels']
    for datafile in da:
        creat_data(datafile, cellfile)
