#!/usr/bin/env python3
"""
简化版分子评估脚本
专注于稳定性分析和基本指标计算
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

# 导入必要的模块
try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from rdkit import DataStructs
    from qm9.rdkit_functions import BasicMolecularMetrics,build_molecule
    use_rdkit = True
except ModuleNotFoundError:
    print("Warning: RDKit not found. Some metrics will be unavailable.")
    use_rdkit = False

from qm9.analyze import check_stability
from configs.datasets_config import get_dataset_info


def read_molecule_file(file_path: str, dataset_info: dict) -> Tuple[np.ndarray, np.ndarray]:
    """读取单个分子文件"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 第一行是原子数量
    n_atoms = int(lines[0].strip())
    
    positions = []
    atom_types = []
    
    for i in range(1, 1 + n_atoms):
        parts = lines[i].strip().split()
        atom_symbol = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        
        positions.append([x, y, z])
        # 将原子符号转换为索引
        if atom_symbol in dataset_info['atom_decoder']:
            atom_idx = dataset_info['atom_decoder'].index(atom_symbol)
        else:
            print(f"Warning: Unknown atom type {atom_symbol}, using H instead")
            atom_idx = 0  # H的索引
        atom_types.append(atom_idx)
    
    return np.array(positions), np.array(atom_types)



def calculate_diversity(molecules: List[Tuple[np.ndarray, np.ndarray]], dataset_info: dict) -> float:
    """
    计算分子多样性
    Diversity是所有生成分子对之间分子指纹相似性的平均值
    """
    if not use_rdkit or len(molecules) < 2:
        return 0.0
    
    # 将分子转换为RDKit分子对象
    rdkit_mols = []
   
    for pos, atom_types in molecules:
        pos = torch.tensor(pos, dtype=torch.float32)
        atom_types = torch.tensor(atom_types, dtype=torch.long)

        graph = (pos, atom_types)
        mol = build_molecule(*graph, dataset_info)
        rdkit_mols.append(mol)

    if len(rdkit_mols) < 2:
        return 0.0
    
    # 计算分子指纹
    fingerprints = []
    for mol in rdkit_mols:
        try:
            # Ensure RDKit mol has proper valence/implicit H information
            try:
                mol.UpdatePropertyCache(strict=False)
            except Exception:
                # non-fatal, continue to attempt further fixes
                pass

            try:
                # Calculate implicit valences and sanitize; if this fails, skip the molecule
                # Chem.rdmolops.CalcImplicitValences(mol)
                Chem.SanitizeMol(mol)
            except Exception as e:
                print(f"Skipping molecule due to sanitization error: {e}")
                continue

            # 使用Morgan指纹（ECFP）
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fingerprints.append(fp)
        except Exception as e:
            print(f"Error computing fingerprint for molecule: {e}")
            continue
    
    if len(fingerprints) < 2:
        return 0.0
    
    # 计算所有分子对之间的相似性
    similarities = []
    for i in range(len(fingerprints)):
        for j in range(i + 1, len(fingerprints)):
            similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            similarities.append(similarity)
    
    # Diversity = 1 - 平均相似性
    if similarities:
        avg_similarity = np.mean(similarities)
        # diversity = 1.0 - avg_similarity
        diversity = avg_similarity
        return diversity
    else:
        return 0.0

def analyze_molecules_stability(molecules_dir: str):
    """分析分子稳定性"""
    dataset_info = get_dataset_info('qm9', remove_h=False)
    molecules_dir = Path(molecules_dir)
    
    # 读取所有分子文件
    molecule_files = sorted(glob.glob(str(molecules_dir / "*.txt")))
    print(f"Found {len(molecule_files)} molecule files")
    
    molecules = []
    n_samples = 0
    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0
    
    for file_path in molecule_files:
        try:
            pos, atom_types = read_molecule_file(file_path, dataset_info)
            molecules.append((pos, atom_types))
            
            # 检查稳定性
            validity_results = check_stability(pos, atom_types, dataset_info)
            n_samples += 1
            molecule_stable += int(validity_results[0])
            nr_stable_bonds += int(validity_results[1])
            n_atoms += int(validity_results[2])
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # 计算多样性
    diversity = 0.0
    if use_rdkit and molecules:
       
        diversity = calculate_diversity(molecules, dataset_info)
        print(f"Molecular Diversity: {diversity:.3f}")
     
    
    # 计算指标
    atom_stable_pct = (nr_stable_bonds / n_atoms) * 100 if n_atoms > 0 else 0
    mol_stable_pct = (molecule_stable / n_samples) * 100 if n_samples > 0 else 0
    
    print("\n=== Molecular Stability Analysis ===")
    print(f"Total molecules analyzed: {n_samples}")
    print(f"Stable molecules: {molecule_stable} ({mol_stable_pct:.1f}%)")
    print(f"Stable atoms: {nr_stable_bonds}/{n_atoms} ({atom_stable_pct:.1f}%)")
    print(f"Molecular Diversity: {diversity:.3f}")
    
    # RDKit分析
    if use_rdkit and molecules:
        all_pos = [torch.tensor(pos, dtype=torch.float32) for pos, _ in molecules]
        all_atom_types = [torch.tensor(atom_types, dtype=torch.long) for _, atom_types in molecules]
        molecules = list(zip(all_pos, all_atom_types))
        try:
            metrics = BasicMolecularMetrics(dataset_info)
            rdkit_metrics = metrics.evaluate(molecules)
            
            if rdkit_metrics and rdkit_metrics[0]:
                validity, uniqueness, novelty = rdkit_metrics[0]
                print(f"\n=== RDKit Metrics ===")
                print(f"Validity: {validity * 100:.1f}%")
                print(f"Uniqueness: {uniqueness * 100:.1f}%")
                print(f"Novelty: {novelty * 100:.1f}%")
        except Exception as e:
            print(f"Error computing RDKit metrics: {e}")
    
    return {
        'n_samples': n_samples,
        'molecule_stable': molecule_stable,
        'mol_stable_pct': mol_stable_pct,
        'atom_stable_pct': atom_stable_pct,
        'nr_stable_bonds': nr_stable_bonds,
        'n_atoms': n_atoms,
        'diversity': diversity,
        'validity': validity if use_rdkit else None,
        'uniqueness': uniqueness if use_rdkit else None,
        'novelty': novelty if use_rdkit else None
    }


def main():
    parser = argparse.ArgumentParser(description='Simple Molecular Evaluation')
    parser.add_argument('--molecules_dir', type=str, default='outputs/log/mols/11071245_debug_Cv_mu',
                       help='Directory containing molecule txt files')
    parser.add_argument('--output_file', type=str, default='molecular_analysis_results.txt',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    print("=== Simple Molecular Evaluation ===")
    print(f"Analyzing molecules in: {args.molecules_dir}")
    
    # 分析分子
    results = analyze_molecules_stability(args.molecules_dir)
    
    # 保存结果
    output_file = f'{args.molecules_dir}/{args.output_file}'
    with open(output_file, 'w') as f:
        f.write("Molecular Analysis Results\n")
        f.write("=" * 30 + "\n")
        f.write(f"Total molecules: {results['n_samples']}\n")
        f.write(f"Stable molecules: {results['molecule_stable']} ({results['mol_stable_pct']:.1f}%)\n")
        f.write(f"Atom stability: {results['atom_stable_pct']:.1f}%\n")
        f.write(f"Stable bonds: {results['nr_stable_bonds']}/{results['n_atoms']}\n")
        f.write(f"Molecular Diversity: {results['diversity']:.3f}\n")
        if use_rdkit:
            f.write(f"Validity: {results['validity'] * 100:.1f}%\n")
            f.write(f"Uniqueness: {results['uniqueness'] * 100:.1f}%\n")
            f.write(f"Novelty: {results['novelty'] * 100:.1f}%\n")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()