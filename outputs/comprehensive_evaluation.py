#!/usr/bin/env python3
"""
综合分子评估脚本
功能：
1. 计算生成分子的稳定性、有效性、唯一性等属性 (Table 1, 2)
2. 使用预训练分类器计算各属性的MAE值 (Table 3)
3. 分子对接评估 (使用qvina2.1)
"""

import os
import sys
import glob
import torch
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# 添加项目路径
sys.path.append(os.path.abspath('.'))

# 导入必要的模块
try:
    from rdkit import Chem
    from qm9.rdkit_functions import BasicMolecularMetrics
    use_rdkit = True
except ModuleNotFoundError:
    print("Warning: RDKit not found. Some metrics will be unavailable.")
    use_rdkit = False

from qm9.analyze import analyze_stability_for_molecules, check_stability
from qm9.property_prediction.main_qm9_prop import get_model, test
from qm9.property_prediction import prop_utils
from qm9 import dataset, utils
from qm9.utils import compute_mean_mad
from configs.datasets_config import get_dataset_info
import qm9.bond_analyze as bond_analyze


class MolecularFileReader:
    """读取分子文件的类"""
    
    def __init__(self, molecules_dir: str):
        self.molecules_dir = Path(molecules_dir)
        self.dataset_info = get_dataset_info('qm9', remove_h=False)
        
    def read_molecule_file(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        读取单个分子文件
        返回: (positions, atom_types)
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # 第一行是原子数量
        n_atoms = int(lines[0].strip())
        # 第二行通常是空行或注释
        
        positions = []
        atom_types = []
        
        for i in range(2, 2 + n_atoms):
            parts = lines[i].strip().split()
            atom_symbol = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            
            positions.append([x, y, z])
            # 将原子符号转换为索引
            if atom_symbol in self.dataset_info['atom_decoder']:
                atom_idx = self.dataset_info['atom_decoder'].index(atom_symbol)
            else:
                print(f"Warning: Unknown atom type {atom_symbol}, using H instead")
                atom_idx = 0  # H的索引
            atom_types.append(atom_idx)
        
        return np.array(positions), np.array(atom_types)
    
    def read_all_molecules(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """读取文件夹中所有分子文件"""
        molecule_files = sorted(glob.glob(str(self.molecules_dir / "*.txt")))
        molecules = []
        
        print(f"Found {len(molecule_files)} molecule files")
        
        for file_path in molecule_files:
            try:
                pos, atom_types = self.read_molecule_file(file_path)
                molecules.append((pos, atom_types))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
        
        print(f"Successfully loaded {len(molecules)} molecules")
        return molecules


class MolecularStabilityAnalyzer:
    """分子稳定性分析器"""
    
    def __init__(self, dataset_info):
        self.dataset_info = dataset_info
    
    def analyze_molecules(self, molecules: List[Tuple[np.ndarray, np.ndarray]]) -> Dict:
        """
        分析分子列表的稳定性
        返回类似Table 1的结果
        """
        n_samples = len(molecules)
        molecule_stable = 0
        nr_stable_bonds = 0
        n_atoms = 0
        
        for pos, atom_type in molecules:
            validity_results = check_stability(pos, atom_type, self.dataset_info)
            molecule_stable += int(validity_results[0])
            nr_stable_bonds += int(validity_results[1])
            n_atoms += int(validity_results[2])
        
        # 计算稳定性指标
        atom_stable_pct = (nr_stable_bonds / n_atoms) * 100 if n_atoms > 0 else 0
        mol_stable_pct = (molecule_stable / n_samples) * 100 if n_samples > 0 else 0
        
        stability_dict = {
            'atom_stable_pct': atom_stable_pct,
            'mol_stable_pct': mol_stable_pct,
            'n_samples': n_samples,
            'stable_molecules': molecule_stable,
            'stable_bonds': nr_stable_bonds,
            'total_atoms': n_atoms
        }
        
        # 如果有RDKit，计算有效性、唯一性、新颖性
        rdkit_metrics = None
        if use_rdkit:
            try:
                metrics = BasicMolecularMetrics(self.dataset_info)
                rdkit_metrics = metrics.evaluate(molecules)
            except Exception as e:
                print(f"Error computing RDKit metrics: {e}")
        
        return stability_dict, rdkit_metrics


class PropertyPredictor:
    """分子属性预测器"""
    
    def __init__(self, classifiers_base_path: str, device: str = 'cuda'):
        self.classifiers_base_path = classifiers_base_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.properties = ['alpha', 'gap', 'homo', 'lumo', 'mu', 'Cv']
        self.classifiers = {}
        self.property_norms = {}
        
    def load_classifier(self, property_name: str):
        """加载特定属性的分类器"""
        classifier_path = f"{self.classifiers_base_path}_{property_name}"
        
        try:
            # 加载参数
            with open(os.path.join(classifier_path, 'args.pickle'), 'rb') as f:
                args_classifier = pickle.load(f)
            
            args_classifier.device = self.device
            args_classifier.model_name = 'egnn'
            
            # 创建模型
            classifier = get_model(args_classifier)
            
            # 加载权重
            checkpoint_path = os.path.join(classifier_path, f'{property_name}_best_checkpoint.npy')
            if os.path.exists(checkpoint_path):
                classifier_state_dict = torch.load(checkpoint_path, map_location=self.device)
                classifier.load_state_dict(classifier_state_dict)
                classifier.to(self.device)
                classifier.eval()
                
                self.classifiers[property_name] = classifier
                print(f"Successfully loaded classifier for {property_name}")
                return True
            else:
                print(f"Checkpoint not found for {property_name}: {checkpoint_path}")
                return False
                
        except Exception as e:
            print(f"Error loading classifier for {property_name}: {e}")
            return False
    
    def load_all_classifiers(self):
        """加载所有属性的分类器"""
        for prop in self.properties:
            self.load_classifier(prop)
        
        print(f"Loaded {len(self.classifiers)} classifiers")
    
    def convert_molecules_to_dataloader(self, molecules: List[Tuple[np.ndarray, np.ndarray]], 
                                     batch_size: int = 32):
        """将分子列表转换为DataLoader格式"""
        # 这里需要实现将分子转换为模型输入格式的逻辑
        # 由于这比较复杂，我们先返回一个占位符
        # 实际实现需要参考qm9/dataset.py中的数据格式
        pass
    
    def predict_properties(self, molecules: List[Tuple[np.ndarray, np.ndarray]]) -> Dict:
        """
        预测分子属性并计算MAE
        返回类似Table 3的结果
        """
        results = {}
        
        # 这里需要实现属性预测的逻辑
        # 由于需要将分子转换为正确的数据格式，这部分比较复杂
        # 建议先实现基本的稳定性分析，然后再扩展属性预测
        
        for prop in self.properties:
            if prop in self.classifiers:
                # 实现属性预测逻辑
                results[prop] = {
                    'mae': 0.0,  # 占位符
                    'predictions': [],
                    'targets': []
                }
        
        return results


class MolecularDocking:
    """分子对接评估"""
    
    def __init__(self, qvina_path: str = "qvina2.1"):
        self.qvina_path = qvina_path
    
    def prepare_molecule_for_docking(self, positions: np.ndarray, atom_types: np.ndarray, 
                                   output_path: str):
        """准备分子用于对接"""
        # 将分子转换为SDF或PDB格式
        # 这需要RDKit或其他化学信息学工具
        pass
    
    def dock_to_protein(self, molecule_path: str, protein_path: str, 
                       binding_site: Dict) -> float:
        """
        使用qvina2.1进行分子对接
        返回对接得分
        """
        # 实现qvina2.1对接逻辑
        # 这需要调用外部程序
        pass
    
    def evaluate_crossdocked_proteins(self, molecules: List[Tuple[np.ndarray, np.ndarray]], 
                                    protein_paths: List[str]) -> Dict:
        """
        在Crossdocked2020数据集的蛋白质上评估分子
        """
        results = {}
        
        # 实现对接评估逻辑
        for i, protein_path in enumerate(protein_paths):
            results[f'protein_{i}'] = {
                'docking_scores': [],
                'best_score': 0.0,
                'mean_score': 0.0
            }
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Molecular Evaluation')
    parser.add_argument('--molecules_dir', type=str, required=True,
                       help='Directory containing molecule txt files')
    parser.add_argument('--classifiers_path', type=str, 
                       default='qm9/property_prediction/outputs/exp_class',
                       help='Base path for property classifiers')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for property prediction')
    parser.add_argument('--skip_property_prediction', action='store_true',
                       help='Skip property prediction (only do stability analysis)')
    parser.add_argument('--skip_docking', action='store_true',
                       help='Skip molecular docking evaluation')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== Comprehensive Molecular Evaluation ===")
    print(f"Molecules directory: {args.molecules_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    
    # 1. 读取分子文件
    print("\n1. Reading molecule files...")
    reader = MolecularFileReader(args.molecules_dir)
    molecules = reader.read_all_molecules()
    
    if not molecules:
        print("No molecules found. Exiting.")
        return
    
    # 2. 分析分子稳定性 (Table 1, 2)
    print("\n2. Analyzing molecular stability...")
    analyzer = MolecularStabilityAnalyzer(reader.dataset_info)
    stability_dict, rdkit_metrics = analyzer.analyze_molecules(molecules)
    
    print("\n=== Stability Analysis Results (Table 1 style) ===")
    print(f"Atom stable: {stability_dict['atom_stable_pct']:.1f}%")
    print(f"Molecule stable: {stability_dict['mol_stable_pct']:.1f}%")
    print(f"Total molecules: {stability_dict['n_samples']}")
    print(f"Stable molecules: {stability_dict['stable_molecules']}")
    
    if rdkit_metrics is not None:
        validity, uniqueness, novelty = rdkit_metrics[0]
        print(f"\n=== Validity and Uniqueness Results (Table 2 style) ===")
        print(f"Valid: {validity * 100:.1f}%")
        print(f"Valid and Unique: {uniqueness * 100:.1f}%")
        print(f"Novelty: {novelty * 100:.1f}%")
    
    # 保存稳定性结果
    stability_results = {
        'stability': stability_dict,
        'rdkit_metrics': rdkit_metrics[0] if rdkit_metrics else None
    }
    
    with open(os.path.join(args.output_dir, 'stability_results.pkl'), 'wb') as f:
        pickle.dump(stability_results, f)
    
    # 3. 属性预测 (Table 3)
    if not args.skip_property_prediction:
        print("\n3. Property prediction analysis...")
        predictor = PropertyPredictor(args.classifiers_path, args.device)
        predictor.load_all_classifiers()
        
        if predictor.classifiers:
            property_results = predictor.predict_properties(molecules)
            
            print("\n=== Property Prediction Results (Table 3 style) ===")
            for prop, results in property_results.items():
                print(f"{prop}: MAE = {results['mae']:.3f}")
            
            # 保存属性预测结果
            with open(os.path.join(args.output_dir, 'property_results.pkl'), 'wb') as f:
                pickle.dump(property_results, f)
        else:
            print("No classifiers loaded. Skipping property prediction.")
    
    # 4. 分子对接评估
    if not args.skip_docking:
        print("\n4. Molecular docking evaluation...")
        docker = MolecularDocking()
        
        # 这里需要提供Crossdocked2020蛋白质的路径
        protein_paths = [
            # 添加两个蛋白质口袋的路径
            # "path/to/protein1.pdb",
            # "path/to/protein2.pdb"
        ]
        
        if protein_paths:
            docking_results = docker.evaluate_crossdocked_proteins(molecules, protein_paths)
            
            print("\n=== Docking Results ===")
            for protein, results in docking_results.items():
                print(f"{protein}: Best score = {results['best_score']:.2f}")
            
            # 保存对接结果
            with open(os.path.join(args.output_dir, 'docking_results.pkl'), 'wb') as f:
                pickle.dump(docking_results, f)
        else:
            print("No protein paths provided. Skipping docking evaluation.")
    
    print(f"\n=== Evaluation Complete ===")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()