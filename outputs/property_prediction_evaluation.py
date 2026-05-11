#!/usr/bin/env python3
"""
分子属性预测评估脚本
使用预训练的分类器计算生成分子的属性MAE
"""

import os
import sys
import glob
import torch
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

# 添加项目路径
sys.path.append(os.path.abspath('.'))

from qm9.property_prediction.main_qm9_prop import get_model
from qm9 import dataset, utils
from configs.datasets_config import get_dataset_info


class MoleculeDataset(torch.utils.data.Dataset):
    """自定义分子数据集"""
    
    def __init__(self, molecules: List[Tuple[np.ndarray, np.ndarray]], dataset_info: dict):
        self.molecules = molecules
        self.dataset_info = dataset_info
        
    def __len__(self):
        return len(self.molecules)
    
    def __getitem__(self, idx):
        pos, atom_types = self.molecules[idx]
        n_atoms = len(pos)
        
        # 创建one-hot编码
        n_atom_types = len(self.dataset_info['atom_decoder'])
        one_hot = np.zeros((n_atoms, n_atom_types))
        for i, atom_type in enumerate(atom_types):
            one_hot[i, atom_type] = 1.0
        
        # 创建mask
        atom_mask = np.ones((n_atoms, 1))
        
        # 创建edge_mask (简化版本)
        edge_mask = np.ones((n_atoms * n_atoms, 1))
        
        return {
            'positions': torch.FloatTensor(pos),
            'one_hot': torch.FloatTensor(one_hot),
            'atom_mask': torch.FloatTensor(atom_mask),
            'edge_mask': torch.FloatTensor(edge_mask),
            'charges': torch.zeros(n_atoms, 1)  # 假设电荷为0
        }


def collate_fn(batch):
    """自定义collate函数"""
    # 找到最大原子数
    max_atoms = max([item['positions'].size(0) for item in batch])
    batch_size = len(batch)
    
    # 初始化批次张量
    positions = torch.zeros(batch_size, max_atoms, 3)
    one_hot = torch.zeros(batch_size, max_atoms, batch[0]['one_hot'].size(1))
    atom_mask = torch.zeros(batch_size, max_atoms, 1)
    edge_mask = torch.zeros(batch_size, max_atoms * max_atoms, 1)
    charges = torch.zeros(batch_size, max_atoms, 1)
    
    for i, item in enumerate(batch):
        n_atoms = item['positions'].size(0)
        positions[i, :n_atoms] = item['positions']
        one_hot[i, :n_atoms] = item['one_hot']
        atom_mask[i, :n_atoms] = item['atom_mask']
        charges[i, :n_atoms] = item['charges']
        
        # 简化的edge_mask处理
        edge_mask[i, :n_atoms*n_atoms] = 1.0
    
    return {
        'positions': positions,
        'one_hot': one_hot,
        'atom_mask': atom_mask,
        'edge_mask': edge_mask,
        'charges': charges
    }


def read_molecules_from_dir(molecules_dir: str, dataset_info: dict) -> List[Tuple[np.ndarray, np.ndarray]]:
    """从目录读取所有分子"""
    molecules_dir = Path(molecules_dir)
    molecule_files = sorted(glob.glob(str(molecules_dir / "*.txt")))
    molecules = []
    
    for file_path in molecule_files:
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            n_atoms = int(lines[0].strip())
            positions = []
            atom_types = []
            
            for i in range(2, 2 + n_atoms):
                parts = lines[i].strip().split()
                atom_symbol = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                
                positions.append([x, y, z])
                if atom_symbol in dataset_info['atom_decoder']:
                    atom_idx = dataset_info['atom_decoder'].index(atom_symbol)
                else:
                    atom_idx = 0  # H
                atom_types.append(atom_idx)
            
            molecules.append((np.array(positions), np.array(atom_types)))
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    return molecules


def load_classifier(property_name: str, classifiers_base_path: str, device: torch.device):
    """加载属性分类器"""
    classifier_path = f"{classifiers_base_path}_{property_name}"
    
    try:
        # 加载参数
        with open(os.path.join(classifier_path, 'args.pickle'), 'rb') as f:
            args_classifier = pickle.load(f)
        
        args_classifier.device = device
        args_classifier.model_name = 'egnn'
        
        # 创建模型
        classifier = get_model(args_classifier)
        
        # 加载权重
        checkpoint_path = os.path.join(classifier_path, f'{property_name}_best_checkpoint.npy')
        if os.path.exists(checkpoint_path):
            classifier_state_dict = torch.load(checkpoint_path, map_location=device)
            classifier.load_state_dict(classifier_state_dict)
            classifier.to(device)
            classifier.eval()
            return classifier
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
            return None
            
    except Exception as e:
        print(f"Error loading classifier for {property_name}: {e}")
        return None


def predict_property(classifier, dataloader, device):
    """使用分类器预测属性"""
    predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch_size, n_nodes, _ = batch['positions'].size()
            
            # 准备输入
            atom_positions = batch['positions'].view(batch_size * n_nodes, -1).to(device)
            atom_mask = batch['atom_mask'].view(batch_size * n_nodes, -1).to(device)
            edge_mask = batch['edge_mask'].to(device)
            nodes = batch['one_hot'].to(device)
            nodes = nodes.view(batch_size * n_nodes, -1)
            
            # 创建邻接矩阵
            from qm9.property_prediction import prop_utils
            edges = prop_utils.get_adj_matrix(n_nodes, batch_size, device)
            
            # 预测
            pred = classifier(h0=nodes, x=atom_positions, edges=edges, 
                            edge_attr=None, node_mask=atom_mask, 
                            edge_mask=edge_mask, n_nodes=n_nodes)
            
            predictions.extend(pred.cpu().numpy())
    
    return np.array(predictions)


def main():
    parser = argparse.ArgumentParser(description='Property Prediction Evaluation')
    parser.add_argument('--molecules_dir', type=str, required=True,
                       help='Directory containing molecule txt files')
    parser.add_argument('--classifiers_path', type=str, 
                       default='qm9/property_prediction/outputs/exp_class',
                       help='Base path for property classifiers')
    parser.add_argument('--properties', nargs='+', 
                       default=['alpha', 'gap', 'homo', 'lumo', 'mu', 'Cv'],
                       help='Properties to predict')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--output_file', type=str, default='property_predictions.txt',
                       help='Output file')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    dataset_info = get_dataset_info('qm9', remove_h=False)
    
    print("=== Property Prediction Evaluation ===")
    print(f"Device: {device}")
    print(f"Properties: {args.properties}")
    
    # 读取分子
    print("Reading molecules...")
    molecules = read_molecules_from_dir(args.molecules_dir, dataset_info)
    print(f"Loaded {len(molecules)} molecules")
    
    if not molecules:
        print("No molecules found!")
        return
    
    # 创建数据集和数据加载器
    mol_dataset = MoleculeDataset(molecules, dataset_info)
    dataloader = torch.utils.data.DataLoader(
        mol_dataset, batch_size=args.batch_size, 
        shuffle=False, collate_fn=collate_fn
    )
    
    # 预测每个属性
    results = {}
    for prop in args.properties:
        print(f"\nPredicting {prop}...")
        classifier = load_classifier(prop, args.classifiers_path, device)
        
        if classifier is not None:
            try:
                predictions = predict_property(classifier, dataloader, device)
                results[prop] = predictions
                print(f"Predicted {len(predictions)} values for {prop}")
                print(f"Mean: {np.mean(predictions):.4f}, Std: {np.std(predictions):.4f}")
            except Exception as e:
                print(f"Error predicting {prop}: {e}")
        else:
            print(f"Skipping {prop} (classifier not available)")
    
    # 保存结果
    with open(args.output_file, 'w') as f:
        f.write("Property Prediction Results\n")
        f.write("=" * 30 + "\n")
        f.write(f"Total molecules: {len(molecules)}\n\n")
        
        for prop, preds in results.items():
            f.write(f"{prop}:\n")
            f.write(f"  Mean: {np.mean(preds):.4f}\n")
            f.write(f"  Std:  {np.std(preds):.4f}\n")
            f.write(f"  Min:  {np.min(preds):.4f}\n")
            f.write(f"  Max:  {np.max(preds):.4f}\n\n")
    
    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()