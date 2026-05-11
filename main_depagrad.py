import argparse
from datetime import datetime
from os.path import join
from tkinter.constants import FALSE
import torch
from torch import nn
import torch.nn.functional as F
from qm9.property_prediction import prop_utils
import pickle
import numpy as np
import os
from qm9.models import get_latent_diffusion
from configs.datasets_config import get_dataset_info
from qm9 import dataset
from qm9.utils import compute_mean_mad
from qm9.property_prediction import main_qm9_prop
from EGMEvolver_single import Fragment_mask
from qm9.analyze import analyze_stability_for_molecules
from equivariant_diffusion import utils as diffusion_utils
import copy
import random
import pandas as pd
import time
try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from rdkit import DataStructs
    rdkit_available = True
except ModuleNotFoundError:
    rdkit_available = False

loss_l1 = nn.L1Loss()
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
max_n_atoms = 29

def compute_atom_uniqueness_from_population(population):
        signatures = []
        for ind in population:
            one_hot = ind['one_hot']
            atom_mask = ind.get('atom_mask', None)
            sig = atom_type_signature_from_one_hot(one_hot, atom_mask)
            signatures.append(sig)
        unique_sigs = list(set(signatures))
        N = len(population)
        ratio = (len(unique_sigs) / N) if N > 0 else 0.0
        return ratio

def atom_type_signature_from_one_hot(one_hot, atom_mask=None):
    if isinstance(one_hot, torch.Tensor):
        x = one_hot.detach().cpu()
        if atom_mask is not None:
            m = atom_mask
            if isinstance(m, torch.Tensor):
                m = m.detach().cpu()
            else:
                m = torch.tensor(m)
            if m.ndim == 2:
                m = m.squeeze(-1)
            x = x[m > 0]
        else:
            x = x[x.sum(dim=1) > 0]
        counts = x.sum(dim=0).to(torch.int64).tolist()
    else:
        x = np.asarray(one_hot)
        if atom_mask is not None:
            m = np.asarray(atom_mask)
            if m.ndim == 2:
                m = np.squeeze(m, axis=-1)
            x = x[m > 0]
        else:
            x = x[x.sum(axis=1) > 0]
        counts = x.sum(axis=0).astype(int).tolist()
    return tuple(counts)


class evol_grad():
    def __init__(self, args):
        self.args = args
        self.population_size = args.population_size
        self.device = args.device
        self.classifier = None
        self.means = None
        self.mads = None
        self.dataset_info = None
        self.model = None
        self.T = 1000  # 总扩散步数
        self.fitness_scores = None
        if isinstance(args.property, list):
            # 直接使用列表
            self.property_names = args.property
        else:
            # 默认处理
            self.property_names = [args.property]

    def add_noise_to_molecule(self, z_x_mu, z_h_mu, node_mask, t_add):
        """
        在潜在空间向分子表示添加t_add步的噪声以实现结构松弛
        直接使用编码器输出的z_x_mu和z_h_mu进行批量加噪
        
        Args:
            z_x_mu: 位置潜在表示均值 [batch_size, n_nodes, 3]
            z_h_mu: 特征潜在表示均值 [batch_size, n_nodes, feature_dim]
            node_mask: 节点掩码 [batch_size, n_nodes, 1]
            t_add: 添加噪声的步数
        
        Returns:
            noisy_z_xh: 添加噪声后的潜在表示 [batch_size, n_nodes, latent_dim]
        """
        batch_size = z_x_mu.size(0)
        n_nodes = z_x_mu.size(1)
        
        # 确保位置部分满足零均值约束
        diffusion_utils.assert_mean_zero_with_mask(z_x_mu, node_mask)
        
        # 将特征部分包装成字典格式（与forward方法一致）
        # z_h_dict = {'categorical': torch.zeros(0).to(z_h_mu), 'integer': z_h_mu}
        
        # 重新组合z_x_mu和z_h_mu为z_xh格式
        z_xh = torch.cat([z_x_mu, z_h_mu], dim=2)
        
        # 创建时间步张量，归一化到[0,1]
        t = torch.full((batch_size, 1), t_add / self.T, device=self.device, dtype=torch.float)
        
        # 获取噪声调度参数
        gamma_t = self.model.inflate_batch_array(self.model.gamma(t), z_x_mu)
        alpha_t = self.model.alpha(gamma_t, z_x_mu)
        sigma_t = self.model.sigma(gamma_t, z_x_mu)
        
        # 使用模型的噪声采样方法生成正确的噪声
        eps = self.model.sample_combined_position_feature_noise(
            n_samples=batch_size, n_nodes=n_nodes, node_mask=node_mask
        )
        
        # 按照compute_loss中的公式添加噪声: z_t = alpha_t * z_xh + sigma_t * eps
        noisy_z_xh = alpha_t * z_xh + sigma_t * eps
        
        return noisy_z_xh
    
    def denoise_with_diffusion(self, noisy_samples, node_mask, edge_mask, context=None, 
                              start_t=None):
        """
        使用预训练的扩散模型进行去噪
        参考 en_diffusion.py 中的 sample 函数实现
        
        Args:
            noisy_samples: 噪声样本 [batch_size, n_nodes, n_features]
            node_mask: 节点掩码
            edge_mask: 边掩码
            context: 条件信息
            start_t: 开始去噪的时间步（如果为None，从T开始完整去噪）
        
        Returns:
            x, h: 去噪后的位置和特征
        """
        if start_t is None:
            start_t = self.model.T  # 从最大时间步开始
        
        batch_size = noisy_samples.size(0)
        z = noisy_samples
        
        # 确保位置部分满足零重心约束
        diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.model.n_dims], node_mask)
        
        # 反向扩散过程：从 start_t 到 0
        # 注意：这里使用标准化的时间步 (s/T, t/T)
        for s in reversed(range(0, start_t)):
            s_array = torch.full((batch_size, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            
            # 标准化时间步到 [0, 1] 区间
            s_normalized = s_array / self.model.T
            t_normalized = t_array / self.model.T
            
            # 使用扩散模型的标准采样步骤
            z = self.model.sample_p_zs_given_zt(
                s_normalized, t_normalized, z, node_mask, edge_mask, context, 
                fix_noise=False
            )
            # 检查中间结果
        # 最终从 z_0 采样得到 x, h
        x, h = self.model.sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=False)
        if torch.isnan(h['categorical']).any():
            print("Warning: h['categorical'] contains NaN after sample_p_xh_given_z0")
            h['categorical'] = torch.nan_to_num(h['categorical'], nan=0.0)
        if torch.isnan(h['integer']).any():
            print("Warning: h['integer'] contains NaN after sample_p_xh_given_z0")
            h['integer'] = torch.nan_to_num(h['integer'], nan=0.0)
        
        # 确保位置满足零重心约束
        diffusion_utils.assert_mean_zero_with_mask(x, node_mask)
        
        # 检查重心漂移并修正
        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f'Warning: COG drift with error {max_cog:.3f}. Projecting positions.')
            x = diffusion_utils.remove_mean_with_mask(x, node_mask)
        
        return x, h

    def cal_child(self, parent, child_list, off, role='offspring'):
        w_encode, losses, new_position, new_one_hot = self.compute_fitness(off, self.device, self.classifiers, self.property_names, self.means, self.mads, role=role)
        off['positions'] = new_position
        off['one_hot'] = new_one_hot
        losses = [loss.item() for loss in losses]
        off['fitness'] = losses
        stability_dict, rdkit_metrics = self.analyze_fitness(off, use_rdkit=False)
        off['mol_sta'] = stability_dict['mol_stable']
        off['atm_sta'] = stability_dict['atm_stable']
        off['w'] = w_encode
        if stability_dict['mol_stable'] >0.8 and stability_dict['atm_stable'] >0.8:
        # if stability_dict['atm_stable'] >0.8:
            # 检查是否重复，不重复才添加
            if not self.is_molecule_duplicate([parent]+child_list, off):
                child_list.append(off)
        return child_list,losses,stability_dict

    def encode_population(self, population):
        """
        使用VAE encoder将分子编码到潜在空间
        
        Args:
            population: 分子population
        
        Returns:
            latent_population: 潜在空间表示的population，格式与egd_optimize中的current_population兼容
        """
        print("将population编码到潜在空间...")
        latent_population = []
        # node_mask_list = []
        # edge_mask_list = []
        if self.model.vae is not None and len(population) > 0:
            # 批量处理所有分子
            with torch.no_grad():
                # 获取最大原子数
                max_atoms = max_n_atoms
                batch_size = len(population)
                
                # 初始化批量张量
                batch_positions = torch.zeros(batch_size, max_atoms, 3, device=self.device)
                batch_one_hot = torch.zeros(batch_size, max_atoms, population[0]['one_hot'].shape[-1], device=self.device)
                batch_node_mask = torch.zeros(batch_size, max_atoms,1, device=self.device)
                batch_edge_mask = torch.zeros(batch_size, max_atoms*max_atoms,1, device=self.device)
                batch_n_atoms = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
                # 填充批量数据
                for i, molecule in enumerate(population):
                    n_atoms = molecule['n_atoms']
                    positions = molecule['positions']  # [n_atoms, 3]
                    one_hot = molecule['one_hot']      #  [n_atoms, 5]
                    atom_mask = molecule['atom_mask']  # [n_atoms,1]
                    edge_mask = molecule['edge_mask']  # [n_atoms*n_atoms,1]
                    # 填充到批量张量中
                    batch_positions[i, :n_atoms] = positions
                    batch_one_hot[i, :n_atoms] = one_hot
                    batch_node_mask[i, :n_atoms] = atom_mask
                    batch_edge_mask[i, :n_atoms*n_atoms] = edge_mask
                    batch_n_atoms[i] = n_atoms
                
                # 准备h字典格式
                h = {
                    'categorical': batch_one_hot,  # [batch_size, max_atoms, 5]
                    'integer': batch_node_mask  # [batch_size, max_atoms, 1]
                }
                
                # 使用VAE的encode方法进行批量编码
                z_x_mu, z_x_sigma, z_h_mu, z_h_sigma = self.model.vae.encode(
                    x=batch_positions,  # [batch_size, max_atoms, 3]
                    h=h,               # {'categorical': [batch_size, max_atoms, 5], 'integer': [batch_size, max_atoms, 1]}
                    node_mask=batch_node_mask,  # [batch_size, max_atoms, 1]
                    edge_mask=batch_edge_mask,        # [batch_size * max_atoms * max_atoms, 1]
                    context=None
                )
                
                # 组合潜在表示
                # batch_latent = torch.cat([z_x_mu, z_h_mu], dim=2)  # [batch_size, max_atoms, 3+2]
                noisy_z_xh = self.add_noise_to_molecule(z_x_mu, z_h_mu, node_mask=batch_node_mask, t_add=self.args.t_add)

                for i,individual in enumerate(population):
                    individual['noises'] = noisy_z_xh[i:i+1, :batch_n_atoms[i], :].squeeze(0)  #z_xh
        else:
            # 如果没有VAE，直接使用原始数据
            print("no VAE encoder and decoder ...")
        
        return population
   
    def initial_data_process(self, data, index):
        """
        修改为支持多属性的数据处理
        
        Args:
            property_names: 属性名称列表
            target_mean_values: 目标均值列表
        """
        # 原子类型映射 (charges -> atom types)
        charge_to_type = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}  # 'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        n_atom_types = len(charge_to_type)
        population = []
        
        for i, idx in enumerate(index):
            # 获取分子数据
            n_atoms = int(data['num_atoms'][idx])
            charges = data['charges'][idx]  # 原子电荷数组
            positions = data['positions'][idx]  # 原子坐标数组

            # 转换charges为atom types
            atom_types = []
            for charge in charges:
                atom_type = charge_to_type.get(int(charge), 1)  # 默认为C
                atom_types.append(atom_type)
            
            # 转换为tensor
            positions_tensor = torch.tensor(positions, dtype=torch.float32, device=device)
            atom_types_tensor = torch.tensor(atom_types, dtype=torch.long, device=device)
            
            # 创建one-hot编码
            one_hot = torch.zeros(n_atoms, n_atom_types, device=device)
            one_hot.scatter_(1, atom_types_tensor.unsqueeze(1), 1)
            
            # 创建掩码
            atom_mask = torch.ones(n_atoms, device=device)
            
            # 创建边掩码
            edge_mask = atom_mask.unsqueeze(0) * atom_mask.unsqueeze(1)  # [n_atoms, n_atoms]
            diag_mask = ~torch.eye(n_atoms, dtype=torch.bool, device=device)  # 移除对角线（自环）
            edge_mask *= diag_mask
            edge_mask = edge_mask.view(n_atoms * n_atoms, 1)  # [n_atoms*n_atoms, 1]

            molecule_data = {
                'positions': positions_tensor,  # [n_atoms, 3]
                'one_hot': one_hot,            # [n_atoms, 5]
                'atom_types': atom_types_tensor,  # [n_atoms] 
                'atom_mask': atom_mask.unsqueeze(-1),        # [n_atoms,1] 
                'edge_mask': edge_mask,                     # [n_atoms*n_atoms, 1] - 边掩码
                'n_atoms': n_atoms,
                'original_index': int(idx)  # 记录原始索引
            }   
            population.append(molecule_data)
            
        return population
    
    def calculate_multi_objective_distance(self, fitness_scores, method='euclidean'):
        """
        计算多目标距离
        
        Args:
            fitness_scores: 各个目标的差距矩阵，形状为 [N, M]，N是个体数量，M是目标数量
            method: 距离计算方法
                - 'euclidean': 欧几里得距离 √(a² + b² + ...)
                - 'manhattan': 曼哈顿距离 |a| + |b| + ...
                - 'mean': 平均距离 (|a| + |b| + ...) / n
                - 'max': 最大距离 max(|a|, |b|, ...)
        
        Returns:
            distance: 计算得到的距离，形状为 [N, 1]
        """

        
        # 确保输入是numpy数组
        if not isinstance(fitness_scores, np.ndarray):
            fitness_scores = np.array(fitness_scores)
        # 如果属性值包含 homo,lumo,gap, 对应目标应乘以1000再计算
        property_names = self.property_names
        if 'homo' in property_names:
            fitness_scores[:, property_names.index('homo')] *= 10*self.p_cos
        if 'lumo' in property_names:
            fitness_scores[:, property_names.index('lumo')] *= 10*self.p_cos
        if 'gap' in property_names:
            fitness_scores[:, property_names.index('gap')] *= 10*self.p_cos

        # 如果是1维数组，转换为2维
        if fitness_scores.ndim == 1:
            fitness_scores = fitness_scores.reshape(1, -1)
        
        if method == 'euclidean':
            # 欧几里得距离：√(a² + b² + ...)，沿axis=1计算
            distances = np.sqrt(np.sum(fitness_scores ** 2, axis=1))
        elif method == 'manhattan':
            # 曼哈顿距离：|a| + |b| + ...，沿axis=1计算
            distances = np.sum(np.abs(fitness_scores), axis=1)
        elif method == 'mean':
            # 平均距离：(|a| + |b| + ...) / n，沿axis=1计算
            distances = np.mean(np.abs(fitness_scores), axis=1)
        elif method == 'max':
            # 最大距离：max(|a|, |b|, ...)，沿axis=1计算
            distances = np.max(np.abs(fitness_scores), axis=1)
        else:
            raise ValueError(f"Unknown distance method: {method}")
        
        if len(fitness_scores) == 1:
            distances = distances.item()
        else:
            distances = distances.tolist()
        # 返回形状为 [N, 1] 的数组
        return distances


    def initial_population_from_top_file_40000(self, npz_file_path, property_names, target_means, 
                                            population_size, device, evol_iterations=10):
        """
        修改为支持多属性的初始化
        """
        # 确保property_names和target_means是列表
        if isinstance(property_names, str):
            property_names = [property_names]
        if not isinstance(target_means, list):
            target_means = [target_means]
          
        # 检查文件是否存在
        if not os.path.exists(npz_file_path):
            raise FileNotFoundError(f"NPZ文件不存在: {npz_file_path}")
        
        # 加载npz文件
        data = np.load(npz_file_path, allow_pickle=True)
        max_n_atoms = max(data['num_atoms'])
        # 检查属性是否存在
        for prop_name in property_names:
            if prop_name not in data:
                available_props = [key for key in data.keys() if key not in ['num_atoms', 'charges', 'positions']]
                raise ValueError(f"属性 {prop_name} 不存在于文件中。可用属性: {available_props}")
        
        total_molecules = len(data['num_atoms'])
     
        csv_path = f"outputs/qm9_40000_stats/{args.exp_name}_best_index.csv"
        df = pd.read_csv(csv_path)
        available_indices = df['index'].values
        candidate_size = min(16, len(available_indices))  # 确保不超过可用索引数量
        candidate_indices = np.random.choice(available_indices, size=candidate_size, replace=False)
        population = self.initial_data_process(data, candidate_indices)

        # 计算候选分子与目标值的总体差距
        target_mean_values = []
        for target_mean in target_means:
            if hasattr(target_mean, 'item'):
                target_mean_values.append(target_mean.item())
            elif hasattr(target_mean, 'cpu'):
                target_mean_values.append(target_mean.cpu().numpy())
            else:
                target_mean_values.append(float(target_mean))
        
        _, population = self.calculate_fitness_grad(population, device, self.classifiers, property_names, target_mean_values)

        print(f"从 {total_molecules} 个分子中随机选择了 {len(candidate_indices)} 个候选分子")

        return population


    def initial_population_from_random_top_eucl_40000(self, npz_file_path, property_names, target_means, 
                                            population_size, device, evol_iterations=10):
        """
        修改为支持多属性的初始化
        """
        # 确保property_names和target_means是列表
        if isinstance(property_names, str):
            property_names = [property_names]
        if not isinstance(target_means, list):
            target_means = [target_means]
        
        candidate_size = population_size * 10
        print(f"从 {npz_file_path} 随机选择 {candidate_size} 个候选分子，然后选择其中最接近目标值的 {population_size} 个分子...")
        print(f"目标属性: {property_names}")
        
        # 检查文件是否存在
        if not os.path.exists(npz_file_path):
            raise FileNotFoundError(f"NPZ文件不存在: {npz_file_path}")
        
        # 加载npz文件
        data = np.load(npz_file_path, allow_pickle=True)
        max_n_atoms = max(data['num_atoms'])
        # 检查属性是否存在
        for prop_name in property_names:
            if prop_name not in data:
                available_props = [key for key in data.keys() if key not in ['num_atoms', 'charges', 'positions']]
                raise ValueError(f"属性 {prop_name} 不存在于文件中。可用属性: {available_props}")
        
        total_molecules = len(data['num_atoms'])
        candidate_size = min(candidate_size, total_molecules)
        
        # 随机选择候选分子索引
        # np.random.seed(42)
        candidate_indices = np.random.choice(total_molecules, size=candidate_size, replace=False)
        population = self.initial_data_process(data, candidate_indices)

        # 计算候选分子与目标值的总体差距
        target_mean_values = []
        for target_mean in target_means:
            if hasattr(target_mean, 'item'):
                target_mean_values.append(target_mean.item())
            elif hasattr(target_mean, 'cpu'):
                target_mean_values.append(target_mean.cpu().numpy())
            else:
                target_mean_values.append(float(target_mean))
        
        fitness_scores, population = self.calculate_fitness_grad(population, device, self.classifiers, property_names, target_mean_values)
        distance_scores = self.calculate_multi_objective_distance(fitness_scores)
        # 选择距离最小的个体
        sorted_indices = np.argsort(distance_scores)[:population_size]
        current_population = [population[i] for i in sorted_indices]

        print(f"从 {total_molecules} 个分子中随机选择了 {len(candidate_indices)} 个候选分子")
        print(f"选择其中最接近目标值的 {population_size} 个分子")
        best_indices = [pop['original_index'] for pop in current_population]
        best_fitness_scores = [pop['fitness'] for pop in current_population]
        
        # 保存到CSV文件

        csv_file = f"outputs/qm9_40000_stats/{args.exp_name}_best_index.csv"
        
        # 读取现有数据
        if os.path.exists(csv_file):
            data = pd.read_csv(csv_file)
            existing_indices = set(data['index'].tolist())
        else:
            data = pd.DataFrame(columns=['index', 'fitness_score'])
            existing_indices = set()
        
        # 只添加新的索引
        new_data = []
        for idx, score in zip(best_indices, best_fitness_scores):
            if idx not in existing_indices:
                new_data.append({'index': idx, 'fitness_score': score})
        
        if new_data:
            new_df = pd.DataFrame(new_data)
            data = pd.concat([data, new_df], ignore_index=True)
            data.to_csv(csv_file, index=False)
            print(f"已保存 {len(new_data)} 个新结果到 {csv_file}")
        else:
            print("没有新的结果需要保存")
        
        return current_population

    def initial_population_from_random_top_40000(self, npz_file_path, property_names, target_means, 
                                            population_size, device, evol_iterations=10):
        """
        修改为支持多属性的初始化
        """
        # 确保property_names和target_means是列表
        if isinstance(property_names, str):
            property_names = [property_names]
        if not isinstance(target_means, list):
            target_means = [target_means]
        
        candidate_size = population_size * 10
        print(f"从 {npz_file_path} 随机选择 {candidate_size} 个候选分子，然后选择其中最接近目标值的 {population_size} 个分子...")
        print(f"目标属性: {property_names}")
        
        # 检查文件是否存在
        if not os.path.exists(npz_file_path):
            raise FileNotFoundError(f"NPZ文件不存在: {npz_file_path}")
        
        # 加载npz文件
        data = np.load(npz_file_path, allow_pickle=True)
        max_n_atoms = max(data['num_atoms'])
        # 检查属性是否存在
        for prop_name in property_names:
            if prop_name not in data:
                available_props = [key for key in data.keys() if key not in ['num_atoms', 'charges', 'positions']]
                raise ValueError(f"属性 {prop_name} 不存在于文件中。可用属性: {available_props}")
        
        total_molecules = len(data['num_atoms'])
        candidate_size = min(candidate_size, total_molecules)
        
        # 随机选择候选分子索引
        # np.random.seed(42)
        candidate_indices = np.random.choice(total_molecules, size=candidate_size, replace=False)
        population = self.initial_data_process(data, candidate_indices)

        # 计算候选分子与目标值的总体差距
        target_mean_values = []
        for target_mean in target_means:
            if hasattr(target_mean, 'item'):
                target_mean_values.append(target_mean.item())
            elif hasattr(target_mean, 'cpu'):
                target_mean_values.append(target_mean.cpu().numpy())
            else:
                target_mean_values.append(float(target_mean))
        
        fitness_scores, population = self.calculate_fitness_grad(population, device, self.classifiers, property_names, target_mean_values)
        current_population = self.environmental_selection(population, fitness_scores, self.population_size)

        
        print(f"从 {total_molecules} 个分子中随机选择了 {len(candidate_indices)} 个候选分子")
        print(f"选择其中最接近目标值的 {population_size} 个分子")
        best_indices = [pop['original_index'] for pop in current_population]
        best_fitness_scores = [pop['fitness'] for pop in current_population]
        
        # 保存到CSV文件

        csv_file = f"outputs/qm9_40000_stats/{args.exp_name}_best_index.csv"
        
        # 读取现有数据
        if os.path.exists(csv_file):
            data = pd.read_csv(csv_file)
            existing_indices = set(data['index'].tolist())
        else:
            data = pd.DataFrame(columns=['index', 'fitness_score'])
            existing_indices = set()
        
        # 只添加新的索引
        new_data = []
        for idx, score in zip(best_indices, best_fitness_scores):
            if idx not in existing_indices:
                new_data.append({'index': idx, 'fitness_score': score})
        
        if new_data:
            new_df = pd.DataFrame(new_data)
            data = pd.concat([data, new_df], ignore_index=True)
            data.to_csv(csv_file, index=False)
            print(f"已保存 {len(new_data)} 个新结果到 {csv_file}")
        else:
            print("没有新的结果需要保存")
        
        return current_population

    def initial_population_from_random_40000(self, npz_file_path, property_names, target_means, 
                                            population_size, device, evol_iterations=10):
        """
        修改为支持多属性的初始化
        """
        # 确保property_names和target_means是列表
        if isinstance(property_names, str):
            property_names = [property_names]
        if not isinstance(target_means, list):
            target_means = [target_means]
        
        # candidate_size = population_size * evol_iterations
        print(f"从 {npz_file_path} 随机选择 {population_size} 个候选分子...")
        print(f"目标属性: {property_names}")
        
        # 检查文件是否存在
        if not os.path.exists(npz_file_path):
            raise FileNotFoundError(f"NPZ文件不存在: {npz_file_path}")
        
        # 加载npz文件
        data = np.load(npz_file_path, allow_pickle=True)
        max_n_atoms = max(data['num_atoms'])
        # 检查属性是否存在
        for prop_name in property_names:
            if prop_name not in data:
                available_props = [key for key in data.keys() if key not in ['num_atoms', 'charges', 'positions']]
                raise ValueError(f"属性 {prop_name} 不存在于文件中。可用属性: {available_props}")
        
        total_molecules = len(data['num_atoms'])
        candidate_size = min(population_size, total_molecules)
        
        # 随机选择候选分子索引
        rng = np.random.RandomState(42)
        candidate_indices = rng.choice(total_molecules, size=candidate_size, replace=False)
        population = self.initial_data_process(data, candidate_indices)

        # 计算候选分子与目标值的总体差距
        target_mean_values = []
        for target_mean in target_means:
            if hasattr(target_mean, 'item'):
                target_mean_values.append(target_mean.item())
            elif hasattr(target_mean, 'cpu'):
                target_mean_values.append(target_mean.cpu().numpy())
            else:
                target_mean_values.append(float(target_mean))
        
        _, population = self.calculate_fitness_grad(population, device, self.classifiers, property_names, target_mean_values)
        
        print(f"从 {total_molecules} 个分子中随机选择了 {len(candidate_indices)} 个候选分子")
        
        return population

    def environmental_selection(self, population, fitness_scores, target_size):
        """
        基于NSGA-II的环境选择：使用快速非支配排序和拥挤距离
        
        Args:
            population: 当前种群
            fitness_scores: 适应度分数，形状为 [N, M]，N是个体数量，M是目标数量
            target_size: 目标种群大小
        
        Returns:
            selected_population: 选中的种群
        """
        # import numpy as np
        
        # 将fitness_scores转换为numpy数组，确保是最小化问题
        if isinstance(fitness_scores[0], list):
            # 如果fitness_scores是损失列表的列表
            objectives = np.array(fitness_scores)
        else:
            objectives = np.array(fitness_scores)
        
        n_individuals, n_objectives = objectives.shape
        
        # 快速非支配排序
        fronts = self.fast_non_dominated_sort(objectives)
        
        # 计算拥挤距离
        # crowding_distances = self.calculate_crowding_distance(objectives, fronts)
        euclidean_distances = self.calculate_multi_objective_distance(objectives)
        # 环境选择
        selected_indices = []
        front_idx = 0
        
        while len(selected_indices) < target_size and front_idx < len(fronts):
            current_front = fronts[front_idx]
            
            if len(selected_indices) + len(current_front) <= target_size:
                # 整个前沿都可以加入
                selected_indices.extend(current_front)
            else:
                # 需要从当前前沿中选择部分个体
                remaining_slots = target_size - len(selected_indices)
                
                # 按拥挤距离降序排序
                front_crowding = [(i, euclidean_distances[i]) for i in current_front]
                front_crowding.sort(key=lambda x: x[1])
                
                # 选择拥挤距离最大的个体
                selected_from_front = [i for i, _ in front_crowding[:remaining_slots]]
                selected_indices.extend(selected_from_front)
            
            front_idx += 1
        
        # 构建选中的种群
        selected_population = [population[i] for i in selected_indices]
        
        return selected_population

    def environmental_children_selection(self, population, fitness_scores, parent_fitness, target_size=1):
        """
        基于NSGA-II的环境选择：使用快速非支配排序和拥挤距离
        
        Args:
            population: 当前种群
            fitness_scores: 适应度分数，形状为 [N, M]，N是个体数量，M是目标数量
            target_size: 目标种群大小
        
        Returns:
            selected_population: 选中的种群
        """
        # import numpy as np
        selected_population = []
        # 将fitness_scores转换为numpy数组，确保是最小化问题
        if isinstance(fitness_scores[0], list):
            # 如果fitness_scores是损失列表的列表
            objectives = np.array(fitness_scores)
        else:
            objectives = np.array(fitness_scores)
        
        n_individuals, n_objectives = objectives.shape
        dominate_idx = []
        for i in range(0, n_individuals):
            if self.dominates(objectives[i, :], parent_fitness):
                selected_population.append(population[i])
                dominate_idx.append(i)
                
        distance = self.calculate_multi_objective_distance(objectives)
        if len(selected_population) < target_size:
            sorted_indices = np.argsort(distance)[:target_size - len(selected_population)]
            selected_population.extend([population[i] for i in sorted_indices])
        elif len(selected_population) > target_size:
            distance_dom = [distance[i] for i in dominate_idx]
            sorted_indices = np.argsort(distance_dom)[:target_size].item()
            selected_population = [population[dominate_idx[sorted_indices]]]

        return selected_population

    def fast_non_dominated_sort(self, objectives):
        """
        快速非支配排序
        
        Args:
            objectives: 目标值矩阵，形状为 [n_individuals, n_objectives]
        
        Returns:
            fronts: 前沿列表，每个前沿包含个体索引
        """
        n_individuals = len(objectives)
        
        # 初始化
        domination_count = [0] * n_individuals  # 支配当前个体的个体数量
        dominated_solutions = [[] for _ in range(n_individuals)]  # 当前个体支配的个体列表
        fronts = [[]]  # 前沿列表
        
        # 计算支配关系
        for i in range(n_individuals):
            for j in range(n_individuals):
                if i != j:
                    if self.dominates(objectives[i], objectives[j]):
                        dominated_solutions[i].append(j)
                    elif self.dominates(objectives[j], objectives[i]):
                        domination_count[i] += 1
            
            # 如果没有个体支配当前个体，则属于第一前沿
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        # 构建后续前沿
        front_idx = 0
        while front_idx < len(fronts) and len(fronts[front_idx]) > 0:
            next_front = []
            for i in fronts[front_idx]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            
            if len(next_front) > 0:
                fronts.append(next_front)
            front_idx += 1
        
        return fronts[:-1] if len(fronts[-1]) == 0 else fronts

    def dominates(self, obj1, obj2):
        """
        判断obj1是否支配obj2（最小化问题）
        
        Args:
            obj1, obj2: 目标值向量
        
        Returns:
            bool: obj1是否支配obj2
        """
        # 对于最小化问题：obj1支配obj2当且仅当
        # obj1在所有目标上都不劣于obj2，且至少在一个目标上严格优于obj2
        at_least_one_better = False
        for i in range(len(obj1)):
            if obj1[i] > obj2[i]:  # obj1在目标i上劣于obj2
                return False
            elif obj1[i] < obj2[i]:  # obj1在目标i上优于obj2
                at_least_one_better = True
        
        return at_least_one_better

    def calculate_crowding_distance(self, objectives, fronts):
        """
        计算拥挤距离
        
        Args:
            objectives: 目标值矩阵
            fronts: 前沿列表
        
        Returns:
            crowding_distances: 每个个体的拥挤距离
        """
        n_individuals, n_objectives = objectives.shape
        crowding_distances = [0.0] * n_individuals
        
        for front in fronts:
            if len(front) <= 2:
                # 前沿中个体数量<=2时，设置为无穷大
                for i in front:
                    crowding_distances[i] = float('inf')
                continue
            
            # 对每个目标计算拥挤距离
            for obj_idx in range(n_objectives):
                # 按当前目标排序
                front_objectives = [(i, objectives[i][obj_idx]) for i in front]
                front_objectives.sort(key=lambda x: x[1])
                
                # 边界个体设置为无穷大
                crowding_distances[front_objectives[0][0]] = float('inf')
                crowding_distances[front_objectives[-1][0]] = float('inf')
                
                # 计算目标范围
                obj_range = front_objectives[-1][1] - front_objectives[0][1]
                if obj_range == 0:
                    continue
                
                # 计算中间个体的拥挤距离
                for i in range(1, len(front_objectives) - 1):
                    individual_idx = front_objectives[i][0]
                    distance = (front_objectives[i + 1][1] - front_objectives[i - 1][1]) / obj_range
                    crowding_distances[individual_idx] += distance
        
        return crowding_distances

    def get_classifiers(self, base_dir_path='', property_names=None, device='cuda'):
        """
        修改为支持加载多个独立的分类器
        """
        classifiers = {}
        
        for prop_name in property_names:
            # 为每个属性构建独立的分类器路径
            classifier_dir = f'{base_dir_path}{prop_name}'
            
            with open(join(classifier_dir, 'args.pickle'), 'rb') as f:
                args_classifier = pickle.load(f)
            args_classifier.device = device
            args_classifier.model_name = 'egnn'
            
            # 加载对应属性的分类器
            classifier_file = f'{prop_name}_best_checkpoint.npy'
            
            classifier = main_qm9_prop.get_model(args_classifier)
            classifier_state_dict = torch.load(join(classifier_dir, classifier_file), map_location=torch.device('cpu'))
            classifier.load_state_dict(classifier_state_dict)
            
            classifiers[prop_name] = classifier.to(device)
            print(f"成功加载 {prop_name} 分类器从路径: {classifier_dir}")
        
        return classifiers

    def get_args_gen(self, dir_path):
        with open(join(dir_path, 'args.pickle'), 'rb') as f:
            args_gen = pickle.load(f)
        args_gen.dataset = 'qm9_second_half'
        assert args_gen.dataset == 'qm9_second_half'
        # assert args_gen.dataset == 'qm9'

        # Add missing args!
        if not hasattr(args_gen, 'normalization_factor'):
            args_gen.normalization_factor = 1
        if not hasattr(args_gen, 'aggregation_method'):
            args_gen.aggregation_method = 'sum'
        return args_gen

    def get_generator(self, dir_path, dataloaders, device, args_gen, property_norms):
        dataset_info = get_dataset_info(args_gen.dataset, args_gen.remove_h)
        model, nodes_dist, prop_dist = get_latent_diffusion(args_gen, device, dataset_info, dataloaders['train'])
        fn = 'generative_model_ema.npy' if args_gen.ema_decay > 0 else 'generative_model.npy'
        model_state_dict = torch.load(join(dir_path, fn), map_location='cpu')
        model.load_state_dict(model_state_dict)

        # The following function be computes the normalization parameters using the 'valid' partition

        if prop_dist is not None:
            prop_dist.set_normalizer(property_norms)
        return model.to(device), nodes_dist, prop_dist, dataset_info

    def get_dataloader(self, args_gen):
        dataloaders, charge_scale = dataset.retrieve_dataloaders(args_gen)
        return dataloaders

    def compute_fitness(self, population, device, classifier, property_names, means,mads, role='parent', t_add=0):
        """
        修改为支持多目标适应度计算，使用多个独立的分类器
        支持2个或3个目标的动态处理
        """
        batch_size = 1
        if t_add == 0:
            t_add = self.args.t_add
        n_atoms = population['n_atoms']
        edges = prop_utils.get_adj_matrix(n_atoms, batch_size, device)
        node_mask = population['atom_mask']
        edge_mask = population['edge_mask']
        torch.cuda.empty_cache()
        # 当存在 encoding 时：将其设为叶子并在此处解码；避免对解码得到的 x/h 做 detach
        if 'noises' in population and population['noises'] is not None:
            encoding = population['noises'].detach().requires_grad_(True)
            z = encoding.unsqueeze(0)  # [1, max_atoms, z_dim]（与 decode 期望一致）
            batch_node_mask = node_mask.unsqueeze(0)
            batch_edge_mask = edge_mask.unsqueeze(0)
            z_x, z_h = self.denoise_with_diffusion(
                noisy_samples=z, 
                node_mask=batch_node_mask, 
                edge_mask=batch_edge_mask, 
                context=None,
                start_t= t_add  # 从 t_add 开始去噪
            )
        
            z_xh = torch.cat([z_x, z_h['categorical'], z_h['integer']], dim=2)
            x_dec, h_dec = self.model.vae.decode(z_xh, batch_node_mask, batch_edge_mask, context=None)
            # 取回当前个体的有效原子长度
            position = x_dec[0, :n_atoms, :]                 # [n_atoms, 3]
            one_hot = h_dec['categorical'][0, :n_atoms, :]   # [n_atoms, 5]
        else:
            # 没有潜变量时，保持原来的行为（用于只优化 x/h 的场景）
            one_hot = population['one_hot']
            position = population['positions']
            encoding = None

        # 多目标损失
        losses = []
        if isinstance(classifier, dict):
            classifier_items = list(classifier.items())
        else:
            classifier_items = [('single', classifier)]

        # 先计算所有损失，不立即求梯度
        for i, (prop_name, clf_item) in enumerate(zip(property_names, classifier_items)):
            if isinstance(clf_item, tuple):
                _, clf = clf_item
            else:
                clf = clf_item

            pred = clf(
                h0=one_hot,
                x=position,
                edges=edges,
                edge_attr=None,
                node_mask=node_mask,
                edge_mask=edge_mask,
                n_nodes=n_atoms
            )

            mean = means[i].to(device)
            mad = mads[i].to(device)
            label = mean.detach().clone()
            loss = loss_l1(mad * pred + mean, label)
            losses.append(loss)
            print(f'{role} {prop_name} loss: {loss.item():.6f}')

        # 统一计算梯度，确保retain_graph正确设置
        gradients_encoding = []
        if encoding is not None:
            for i, loss in enumerate(losses):
                retain = (i < len(losses) - 1)  # 只有最后一个设为False
                grad_encoding = torch.autograd.grad(loss, encoding, retain_graph=retain)[0]
                gradients_encoding.append(grad_encoding)
        
        return gradients_encoding, losses, position.detach(), one_hot.detach()

    def calculate_fitness_grad(self, population, device, classifier, property_names, means, roles='parents'):

        fitness_scores = []
        for individual in population:
            if 'fitness' not in individual:
                w, losses, _, _ = self.compute_fitness(individual, device, classifier, property_names, self.means, self.mads, role=roles)
                losses = [loss.item() for loss in losses]
                individual['fitness'] = losses
                # individual['w'] = w
            if 'atom_types' not in individual:
                individual['atom_types'] = individual['one_hot'].argmax(dim=-1)
            fitness_scores.append(losses)  
        return fitness_scores, population

    def gen_off_w(self, w):
        de_F = self.args.de_F
        # w_pos = w
        g1, g2 = w[0], w[1]  # [N,3], [N,3]

        eps = 1e-8
        same_sign = torch.sign(g1) == torch.sign(g2)                # [N,3]
        zero_involved = (g1.abs() <= eps) | (g2.abs() <= eps)       # [N,3]
        mask_same_or_zero = same_sign | zero_involved               # [N,3]

        # 逐坐标保留绝对值更小（同号或含零）
        smaller_abs = torch.where(g1.abs() <= g2.abs(), g1, g2)     # [N,3]
        # 在不满足掩码的位置：DE 差分整合
        off1 = torch.where(
            mask_same_or_zero,
            smaller_abs,                           # 逐坐标保留绝对值更小的值
            g1 + de_F * (g1 - g2)                    # 差分更新
        )  # [N,3]

        off2 = torch.where(
            mask_same_or_zero,
            smaller_abs,                           # 逐坐标保留绝对值更小的值
            g2 + de_F * (g2 - g1)                    # 差分更新
        )  # [N,3]
        
            # 每原子向量级冲突检测（点积<0为冲突）
        dot = (g1 * g2).sum(dim=-1)                                 # [N]
        n2 = (g2 * g2).sum(dim=-1) + eps                            # [N]
        conflict = dot < 0                                          # [N]

        # 投影消冲突：去掉 g1 在 g2 方向上的分量（仅对冲突原子）
        proj_coeff = torch.where(conflict, dot / n2, torch.zeros_like(dot))  # [N]
        g1_proj = g1 - proj_coeff.unsqueeze(-1) * g2                # [N,3]
        # 同理也可对 g2 去掉在 g1 上的分量
        n1 = (g1 * g1).sum(dim=-1) + eps
        proj_coeff2 = torch.where(conflict, dot / n1, torch.zeros_like(dot))
        g2_proj = g2 - proj_coeff2.unsqueeze(-1) * g1               # [N,3]

        # 合成综合梯度（冲突原子用投影后的平均，不冲突用逐坐标规则）
        g_comb_conflict = 0.5 * (g1_proj + g2_proj)                  # [N,3]
        off3 = torch.where(
            mask_same_or_zero,
            smaller_abs,                                            # 同号或含零：逐坐标保留绝对值更小
            g_comb_conflict                                         # 异号：用投影平均，避免互相伤害
        )
        return off1, off2, off3

    def gen_off_2w(self, w):    
        de_F = self.args.de_F
        # F=1
        # l = w[0].shape[0]
        # F = torch.rand(w[0].shape[0], 1, device=self.device) 
        # w_pos = w
        g1, g2 = w[0], w[1]  # [N,3], [N,3]

        eps = 1e-8
        same_sign = torch.sign(g1) == torch.sign(g2)                # [N,3]
        zero_involved = (g1.abs() <= eps) | (g2.abs() <= eps)       # [N,3]
        mask_same_or_zero = same_sign | zero_involved               # [N,3]

        # 逐坐标保留绝对值更小（同号或含零）
        smaller_abs = torch.where(g1.abs() <= g2.abs(), g1, g2)     # [N,3]
        # 在不满足掩码的位置：DE 差分整合
        
            # 每原子向量级冲突检测（点积<0为冲突）
        dot = (g1 * g2).sum(dim=-1)                                 # [N]
        n2 = (g2 * g2).sum(dim=-1) + eps                            # [N]
        conflict = dot < 0                                          # [N]

        # 投影消冲突：去掉 g1 在 g2 方向上的分量（仅对冲突原子）
        proj_coeff = torch.where(conflict, dot / n2, torch.zeros_like(dot))  # [N]
        g1_proj = g1 - proj_coeff.unsqueeze(-1) * g2                # [N,3]
        # 同理也可对 g2 去掉在 g1 上的分量
        n1 = (g1 * g1).sum(dim=-1) + eps
        proj_coeff2 = torch.where(conflict, dot / n1, torch.zeros_like(dot))
        g2_proj = g2 - proj_coeff2.unsqueeze(-1) * g1               # [N,3]

        # 合成综合梯度（冲突原子用投影后的平均，不冲突用逐坐标规则）
        g_comb_conflict = 0.5 * (g1_proj + g2_proj)                  # [N,3]
        off3 = torch.where(
            mask_same_or_zero,
            smaller_abs,                                            # 同号或含零：逐坐标保留绝对值更小
            g_comb_conflict                                         # 异号：用投影平均，避免互相伤害
        )

        off1 = torch.where(
            mask_same_or_zero,
            smaller_abs,                           # 逐坐标保留绝对值更小的值
            g1 + de_F * random.random() * g2_proj                   # 差分更新
        )  # [N,3]

        off2 = torch.where(
            mask_same_or_zero,
            smaller_abs,                           # 逐坐标保留绝对值更小的值
            g2 + de_F * random.random() * g1_proj                   # 差分更新
        )  # [N,3]
        
        # off4 = torch.where(
        #     mask_same_or_zero,
        #     smaller_abs,                           # 逐坐标保留绝对值更小的值
        #     g1 + F * g1_proj                   # 差分更新
        # )  # [N,3]
        
        # off5 = torch.where(
        #     mask_same_or_zero,
        #     smaller_abs,                           # 逐坐标保留绝对值更小的值
        #     g2 + F * g2_proj                   # 差分更新
        # )  # [N,3]

        return off1, off2, off3

    def gen_off_3w(self, w):
        de_F = self.args.de_F
        # F=1
        g1, g2, g3 = w[0], w[1], w[2]  # [N,3], [N,3], [N,3]

        eps = 1e-8
        # 三者同号的准确判断（逐坐标）
        signs_equal = (torch.sign(g1) == torch.sign(g2)) & (torch.sign(g2) == torch.sign(g3))  # [N,3]
        # 任意一个为近零则保守处理
        zero_involved = (g1.abs() <= eps) | (g2.abs() <= eps) | (g3.abs() <= eps)               # [N,3]
        mask_same_or_zero = signs_equal | zero_involved                                         # [N,3]

        # 逐坐标保留三者中的最小绝对值（更稳健）
        smaller_12 = torch.where(g1.abs() <= g2.abs(), g1, g2)                                  # [N,3]
        smaller_abs = torch.where(smaller_12.abs() <= g3.abs(), smaller_12, g3)                 # [N,3]

        # PCGrad式冲突消解：对负点积的对进行投影
        def project_conflicts(gi, gj):
            dot = (gi * gj).sum(dim=-1)                                                         # [N]
            denom = (gj * gj).sum(dim=-1) + eps                                                 # [N]
            coeff = torch.where(dot < 0, dot / denom, torch.zeros_like(dot))                    # [N]
            return gi - coeff.unsqueeze(-1) * gj                                                # [N,3]

        # 对每个梯度依次去除与其他梯度的冲突分量
        g1_proj = project_conflicts(g1, g2)
        g1_proj = project_conflicts(g1_proj, g3)

        g2_proj = project_conflicts(g2, g1)
        g2_proj = project_conflicts(g2_proj, g3)

        g3_proj = project_conflicts(g3, g1)
        g3_proj = project_conflicts(g3_proj, g2)

        # 冲突区的综合梯度：投影后的三者平均
        g_pc = (g1_proj + g2_proj + g3_proj) / 3.0                                              # [N,3]

        # 三个子代的生成：
        # - 一致/含零：保守取 smaller_abs
        # - 冲突：使用PCGrad综合或DE式差分整合
        off1 = torch.where(
            mask_same_or_zero,
            smaller_abs,
            g1 + de_F * random.random()* ((g2_proj + g3_proj) / 2.0)                                                # 差分整合推进目标1
        )  # [N,3]

        off2 = torch.where(
            mask_same_or_zero,
            smaller_abs,
            g2 + de_F * random.random() * ((g1_proj + g3_proj) / 2.0)                                                # 差分整合推进目标2
        )  # [N,3]

        off3 = torch.where(
            mask_same_or_zero,
            smaller_abs,
            g3 + de_F * random.random() * ((g1_proj + g2_proj) / 2.0)                                                                                # 冲突区用综合不伤害任何目标
        )  # [N,3]

        off4 = torch.where(
            mask_same_or_zero,
            smaller_abs,
            g_pc                                              # 差分整合推进目标3
        )  # [N,3]
        return off1, off2, off3, off4

    def analyze_fitness(self, current_population, use_rdkit=False):
        molecules = {'one_hot': [], 'x': [], 'node_mask': []}
        if not isinstance(current_population, list):
            current_population = [current_population]

        for ind in current_population:
            one_hot = ind['one_hot'].detach().cpu()
            positions = ind['positions'].detach().cpu()
            atom_mask = ind['atom_mask'].detach().cpu()
        
            # 如果张量不是max_n_atoms长度，需要进行填充
            if one_hot.size(0) < max_n_atoms:
                # 填充到max_n_atoms长度
                pad_size = max_n_atoms - one_hot.size(0)
                one_hot = torch.nn.functional.pad(one_hot, (0, 0, 0, pad_size), value=0)
                positions = torch.nn.functional.pad(positions, (0, 0, 0, pad_size), value=0)
                atom_mask = torch.nn.functional.pad(atom_mask, (0, 0, 0, pad_size), value=0)
            elif one_hot.size(0) > max_n_atoms:
                # 截断到max_n_atoms长度
                one_hot = one_hot[:max_n_atoms]
                positions = positions[:max_n_atoms]
                atom_mask = atom_mask[:max_n_atoms]
            
            # 添加batch维度 [1, max_nodes, features]
            molecules['one_hot'].append(one_hot.unsqueeze(0))
            molecules['x'].append(positions.unsqueeze(0))
            molecules['node_mask'].append(atom_mask.unsqueeze(0))

        # 使用torch.cat拼接所有张量，与eval_analyze.py保持一致
        molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
        stability_dict, rdkit_metrics = analyze_stability_for_molecules(molecules, self.dataset_info, use_rdkit=use_rdkit)
        
        print(stability_dict)
        return stability_dict, rdkit_metrics

    def compute_diversity(self, population):
        """
        计算种群分子的平均 Tanimoto 相似度（Morgan 指纹）。
        返回值：
            - float: 平均 Tanimoto 相似度（数值越小多样性越高）
            - None: 若 RDKit 不可用或无法构建分子
        """
        if not 'dataset_info' in self.__dict__ or self.dataset_info is None:
            return None
        if not rdkit_available:
            return None
        # 统一为列表处理
        inds = population if isinstance(population, list) else [population]
        fps = []
        for ind in inds:
            try:
                # positions
                # pos = ind['positions']
                # pos = pos.detach().cpu().numpy() if hasattr(pos, 'detach') else np.asarray(pos)
                # # atom types from one_hot
                # one_hot = ind['one_hot']
                # if hasattr(one_hot, 'detach'):
                #     atom_types = torch.argmax(one_hot, dim=-1).detach().cpu().numpy()
                # else:
                #     atom_types = np.argmax(one_hot, axis=-1)
                # # 构建 RDKit 分子
                # mol = build_molecule(atom_types, pos, self.dataset_info)
                # if mol is None:
                #     continue
                mol = Chem.MolFromSmiles(ind)
                if mol is None:
                    continue
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                fps.append(fp)
            except Exception:
                continue
        n = len(fps)
        if n <= 1:
            return None if n == 0 else 0.0
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
        if len(sims) == 0:
            return None
        return float(np.mean(sims))

    def center_noise(self, noisy_z_xh, batch_node_mask):
        noise_z_x = noisy_z_xh[:, :3]
        noise_center = diffusion_utils.remove_mean_with_mask(
            noise_z_x.unsqueeze(0),  # 添加batch维度 [1, off_n_node, 3]
            batch_node_mask.unsqueeze(0)   # 添加batch维度 [1, off_n_node, 1]
        ).squeeze(0)
        noisy_z_xh[:, :3] = noise_center
        return noisy_z_xh

    def is_molecule_duplicate(self, existing_population, new_molecule, tolerance=1e-6):
        """
        Check if new molecules duplicate molecules in an existing population 
        Args: 
            existing_population: List of existing populations 
            new_molecule: New molecules to be checked 
            tolerance: Tolerance for coordinate comparison 
            Returns: bool: True means duplicates, False means no duplicates
        """
        if not existing_population:
            return False
        
        
        existing_signatures = set()
        for mol in existing_population:
            pos = mol['positions'].detach().cpu().numpy() if hasattr(mol['positions'], 'detach') else mol['positions']
            atoms = mol['atom_types'].detach().cpu().numpy() if hasattr(mol['atom_types'], 'detach') else mol['atom_types']

            
            pos_quantized = np.round(pos / tolerance).astype(int)
            signature = (tuple(atoms), tuple(pos_quantized.flatten()))
            existing_signatures.add(signature)
        
        
        new_pos = new_molecule['positions'].detach().cpu().numpy() if hasattr(new_molecule['positions'], 'detach') else new_molecule['positions']
        new_atoms = new_molecule['atom_types'].detach().cpu().numpy() if hasattr(new_molecule['atom_types'], 'detach') else new_molecule['atom_types']
        
        # Create a new signature
        new_pos_quantized = np.round(new_pos / tolerance).astype(int)
        new_signature = (tuple(new_atoms), tuple(new_pos_quantized.flatten()))
        
        # Check for duplicates
        return new_signature in existing_signatures

    def save_molecule_to_file(self, positions: np.ndarray, atom_types: np.ndarray, 
                         output_path: str, molecule_idx: int):
        """
        将分子保存为指定格式的文件
        Args:
            positions: atom positions (n_atoms, 3)
            atom_types: atom types
            output_path: output path
            molecule_idx: molecule index
            property_value: property value
            property_name:  property name
        """
        os.makedirs(output_path, exist_ok=True)
        
        filename = os.path.join(output_path, f"molecule_{molecule_idx:03d}.txt")
        
        # 
        atom_decoder = self.dataset_info['atom_decoder']  # ['H', 'C', 'N', 'O', 'F']
        
        with open(filename, 'w') as f:
            #
            n_atoms = len(positions)
            f.write(f"{n_atoms}\n")
            
        
            for i in range(n_atoms):
                atom_type_idx = int(atom_types[i])
                atom_symbol = atom_decoder[atom_type_idx] if atom_type_idx < len(atom_decoder) else f'X{atom_type_idx}'
                x, y, z = positions[i]
                f.write(f"{atom_symbol} {x:.9f} {y:.9f} {z:.9f}\n")
    
    def sample(self, fitness_score, sample_size):
        """
     
        """
        # 
        objective_scores = self.calculate_multi_objective_distance(fitness_score)
        objective_scores = np.array(objective_scores)
        # sampling probability
        sample_prob = objective_scores / objective_scores.sum()
        # sampling
        # sample_indices = np.random.choice(len(sample_prob), size=sample_size, p=sample_prob)
        sample_indices = np.random.choice(len(sample_prob), size=sample_size, replace=False)
        return sample_indices

    def get_mask(self,w1,w2,loss1,loss2, min_len):
        num_obj = len(w1)
        w1_consistency = []
        w2_consistency = []
        mask_list = []

        for i in range(num_obj):
            w1_obj_normalized=(F.normalize(w1[i], dim=-1, eps=1e-8))
            w2_obj_normalized=(F.normalize(w2[i], dim=-1, eps=1e-8))

            w1_obj_consistency=(1.0 / (torch.var(w1_obj_normalized, dim=-1, keepdim=True) + 1e-8))
            w2_obj_consistency=(1.0 / (torch.var(w2_obj_normalized, dim=-1, keepdim=True) + 1e-8))

            quality1_obj=(torch.exp(-loss1[i]))
            quality2_obj=(torch.exp(-loss2[i]))
            
            w1_consistency =(w1_obj_consistency * quality1_obj)
            w2_consistency =(w2_obj_consistency * quality2_obj)
    
            
            mask=(w1_consistency[:min_len] < w2_consistency[:min_len])
            mask_list.append(mask.squeeze(-1))

        # mask = mask_list[0]
        # for i in range(1, num_obj):
        #     mask &= mask_list[i]
        # mask = mask.squeeze(-1)
        return mask_list

    def run_evolution(self):
        # Get classifier
    
        print(f"目标属性列表: {self.property_names}")
    
    
        self.classifiers = self.get_classifiers(self.args.classifiers_path, device=self.device, property_names=self.property_names)
        # Get generator and dataloader used to train the generator and evalute the classifier
        args_gen = self.get_args_gen(self.args.generators_path)
        
        # Careful with this -->
        if not hasattr(args_gen, 'diffusion_noise_precision'):
            args_gen.normalization_factor = 1e-4
        if not hasattr(args_gen, 'normalization_factor'):
            args_gen.normalization_factor = 1
        if not hasattr(args_gen, 'aggregation_method'):
            args_gen.aggregation_method = 'sum'
        args_gen.conditioning = properties
        dataloaders = self.get_dataloader(args_gen)
        property_norms = compute_mean_mad(dataloaders, args_gen.conditioning, args_gen.dataset)
        self.dataset_info = get_dataset_info(args_gen.dataset, args_gen.remove_h)
        self.model, nodes_dist, prop_dist, _ = self.get_generator(self.args.generators_path, dataloaders,
                                                    self.device, args_gen, property_norms)
        means = []
        mads = []
        for prop in self.property_names:
            means.append(property_norms[prop]['mean'])
            mads.append(property_norms[prop]['mad'])
        self.means = means
        self.mads = mads
        target_tensor = [mean.detach() for mean in means]  

        # 1. Initialize population - Select a molecule from qm9_40000.npz
        print('-'*30)
        print(f"init population size: {self.population_size}")
        population = self.initial_population_from_random_40000(
            npz_file_path=self.args.population_dir,
            property_names=self.property_names,
            target_means=target_tensor,
            population_size=self.population_size,
            device=self.device,
            evol_iterations=self.args.n_generations    
        )
        csv_file = f"outputs/qm9_40000_stats/{args.exp_name}_best_index.csv"
        # 
        if os.path.exists(csv_file):
            best_population = self.initial_population_from_top_file_40000(
                npz_file_path=self.args.population_dir,
                property_names=self.property_names,
                target_means=target_tensor,
                population_size=self.population_size,
                device=self.device,
                evol_iterations=self.args.n_generations    
            )
        else:
            best_population = self.initial_population_from_random_top_eucl_40000(
                npz_file_path=self.args.population_dir,
                property_names=self.property_names,
                target_means=target_tensor,
                population_size=self.population_size,
                device=self.device,
                evol_iterations=self.args.n_generations    
            )
        # population = Fragment_mask(best_population, self.dataset_info, max_n_atoms)
        # best_population = Fragment_mask(best_population, self.dataset_info, max_n_atoms)
        # torch.cuda.empty_cache()
        population = self.encode_population(population)
        best_population = self.encode_population(best_population)

        parent_fitness = []
        for individual in population:
            stability_dict, rdkit_metrics = self.analyze_fitness(individual, use_rdkit=False)
            mol_sta = stability_dict['mol_stable']
            atm_sta = stability_dict['atm_stable']
            # {'mol_stable': 1.0, 'atm_stable': 1.0}
            individual['mol_sta'] = mol_sta
            individual['atm_sta'] = atm_sta
            parent_fitness.append(individual['fitness'])
        self.fitness_scores = parent_fitness
        mean_parent_fitness = np.mean(parent_fitness,axis=0)
        best_fitness_scores = []
        for individual in best_population:
            best_fitness_scores.append(individual['fitness'])
        print('------------finish init population-------------')
        # print('-------------start to cal Scaffold -----------------')
        # initial_population = Fragment_mask(initial_population, self.dataset_info, max_n_atoms)
        # print('------------finish cal Scaffold -----------------')
        n_generation = args.n_generations
        de_F = self.args.de_F
        timestamp = datetime.now().strftime("%m%d%H%M")
        log_filename = f"outputs/log/{self.args.exp_name}_lr10diver_{self.population_size}_{self.args.t_add}_{timestamp}.txt"
        with open(log_filename, 'a') as f:
            f.write(f'mean parent fitness: {mean_parent_fitness}\n')
            start_time = time.time()
            for generation in range(n_generation):
                print(f'------------- generation {generation} -----------------')
                f.write(f'generation:{generation} / {args.n_generations} \n')
                offspring_population = []
                self.p_cos = np.cos(np.pi/2*(generation/(n_generation+1)))
                medium_level = np.percentile(self.fitness_scores, 50, axis=0)
                if 'gap' in self.property_names:
                    medium_level[self.property_names.index('gap')] *= 10*self.p_cos
                if 'lumo' in self.property_names:
                    medium_level[self.property_names.index('lumo')] *= 10*self.p_cos
                if 'homo' in self.property_names:
                    medium_level[self.property_names.index('homo')] *= 10*self.p_cos
                # best_level = min(self.fitness_scores)
                for i, individual in enumerate(population):
                    old_data = individual['noises']
                    old_fitness = individual['fitness'].copy()
                    if 'gap' in self.property_names:
                        old_fitness[self.property_names.index('gap')] *= 10*self.p_cos
                    if 'lumo' in self.property_names:
                        old_fitness[self.property_names.index('lumo')] *= 10*self.p_cos
                    if 'homo' in self.property_names:
                        old_fitness[self.property_names.index('homo')] *= 10*self.p_cos

                    result = np.sum( np.array(old_fitness)) > np.sum(medium_level)
                    # result = np.any( np.array(old_fitness) > np.array(medium_level))
                    if result:
                        selected_children = None
                        if 'w' in individual:
                            w = individual['w']
                            loss1 = individual['fitness']
                        else:
                            w, loss1, _, _ = self.compute_fitness(individual, self.device, self.classifiers, self.property_names, means, mads, role='offspring')
                        child_list = []
                        # choose = np.array(individual["fitness"]) > medium_level
                            
                        if len(loss1)==2:
                            off_w1, off_w2, off_w3 = self.gen_off_2w(w)
                        else:
                            off_w1, off_w2, off_w3, off_w4 = self.gen_off_3w(w)     
                
                        off1 = copy.deepcopy(individual)
                        off2 = copy.deepcopy(individual)
                        off3 = copy.deepcopy(individual)
        
                        f.write(f'parent{i}: {individual["fitness"]}\n')
                        print(f'parent{i}: {individual["fitness"]}')

                        a  = np.max (np.array(individual["fitness"]))
                        lr =  10**self.p_cos *a*(1- generation / (n_generation+1))
                        # lr = max(0.5 * (1- generation / (n_generation+1) ), 0.05)
                        off1['noises'] = old_data - lr * off_w1
                        off2['noises'] = old_data - lr * off_w2
                        off3['noises'] = old_data - lr * off_w3

                        off1['noises'] = self.center_noise(off1['noises'], off1['atom_mask'])
                        off2['noises'] = self.center_noise(off2['noises'], off2['atom_mask'])
                        off3['noises'] = self.center_noise(off3['noises'], off3['atom_mask'])
                
                        for j, off in enumerate([off1, off2, off3]):
                            child_list, losses, stability_dict = self.cal_child(individual, child_list, off)
                            f.write(f'child of parent{i} {losses}, {stability_dict}\n')
                            print(f'child{j}: {losses}')
                        f.flush()
                        
                        off_fitness = [ind['fitness'] for ind in child_list]
                        if len(off_fitness) > 0:
                            distance = self.calculate_multi_objective_distance([individual['fitness']]+ off_fitness)
                            for j, dist in enumerate(distance):
                                if j==0:
                                    min_dis = distance[0]
                                    continue
                                else:
                                    if dist < min_dis:
                                        selected_children = child_list[j-1]
                                        min_dis = dist
                        # result = False
                        if result:
                        # if selected_children is None or result:
                            evol_child = []
                            if selected_children is not None:
                                evol_child.append(selected_children)
                            # old_data_noise = old_data.clone()
                            # z_x_mu, z_h_mu = old_data_noise[:, :3],old_data_noise[:, 3:]
                            
                            # old_data_noise = self.add_noise_to_molecule(z_x_mu=z_x_mu, z_h_mu=z_h_mu, node_mask=individual['atom_mask'], t_add=100)
                            new_indics = self.sample(best_fitness_scores, 1)
                            new_individual = best_population[new_indics[0]]
                            new_data = new_individual['noises']
                            # new_w, loss2, _, _ = self.compute_fitness(new_individual, self.device, self.classifiers, self.property_names, means, mads, role='offspring')
                            min_len = min(individual['n_atoms'], new_individual['n_atoms'])
                            flag = False
                            if individual['n_atoms'] == min_len:
                                off_noise = new_data.clone()
                                off_de = copy.deepcopy(new_individual)
                                max_len = new_individual['n_atoms']
                                flag = True
                            else:
                                off_noise = old_data.clone()
                                off_de = copy.deepcopy(individual)
                                max_len = individual['n_atoms']
                            random_values = np.random.random(max_len)
                            mask = (random_values >= 0.5).astype(int)
                            # mask_list = self.get_mask(w, new_w, loss1, loss2, min_len)
                            off_noise[:min_len] = old_data[:min_len]
                            # for mask in mask_list:    
                            #     r=random.random()
                            #     off_noise[:min_len][~mask] = old_data[:min_len][~mask]
                            #     off_noise[:min_len][mask] = off_noise[:min_len][mask]+de_F*r*(new_data[:min_len,:][mask]-off_noise[:min_len,:][mask])
                            r=random.random()
                            # r=1
                            # if flag:
                            #     off_noise[:min_len][mask] = off_noise[:min_len][mask]+de_F*r*(old_data[:min_len,:][mask]-off_noise[:min_len,:][mask])
                            # else:
                            off_noise[:min_len][mask] = off_noise[:min_len][mask]+de_F*r*(new_data[:min_len,:][mask]-off_noise[:min_len,:][mask])
                        
                            off_noise = self.center_noise(off_noise, off_de['atom_mask'])
                            off_de['noises'] = off_noise
                            evol_child, losses, stability_dict = self.cal_child(individual, evol_child, off_de)
                            f.write(f'child of parent{i} {losses}, {stability_dict}\n')
                            print(f'child_evol: {losses}')
                            off_fitness = [ind['fitness'] for ind in evol_child]
                            if len(off_fitness) > 0:
                                distance = self.calculate_multi_objective_distance([individual['fitness']]+ off_fitness)
                                for j, dist in enumerate(distance):
                                    if j==0:
                                        min_dis = distance[0]
                                        continue
                                    else:
                                        if dist < min_dis:
                                            selected_children = evol_child[j-1]
                                            min_dis = dist
                            
                        if selected_children is not None:
                            offspring_population.append(selected_children)
                        else:
                            offspring_population.append(individual)
                    else:
                        offspring_population.append(individual)
                        f.write(f'parent{i}: {individual["fitness"]}\n')
                        print(f'parent{i}: {individual["fitness"]}')
                population = offspring_population
                final_fitness_scores = [ind['fitness'] for ind in population]
                final_mol_sta = [ind['mol_sta'] for ind in population]
                final_atm_sta = [ind['atm_sta'] for ind in population]
                self.fitness_scores = final_fitness_scores
                mean_fitness = np.mean(final_fitness_scores, axis=0)
                mean_mol_sta = np.mean(final_mol_sta, axis=0)
                mean_atm_sta = np.mean(final_atm_sta, axis=0)
                f.write(f'gen {generation}: mean fitness: {mean_fitness}, mean mol_sta: {mean_mol_sta}, mean atm_sta: {mean_atm_sta}\n')
                print(f'gen {generation}: mean fitness: {mean_fitness}, mean mol_sta: {mean_mol_sta}, mean atm_sta: {mean_atm_sta}\n')
               
                if (generation +1) % self.args.log_interval == 0:
                    stability_dict, rdkit_metrics = self.analyze_fitness(population, use_rdkit=True)
                    self.smiles = rdkit_metrics[1]
                    validity, uniqueness, novelty = rdkit_metrics[0]
                    f.write(f'final_valid_unique_novel, {[validity, uniqueness, novelty]}\n')
                    print(f'final_valid_unique_novel, {[validity, uniqueness, novelty]}\n')
                    f.flush()
            end_time = time.time()        
            diversity = 1-self.compute_diversity(self.smiles)
            atom_uniqueness = compute_atom_uniqueness_from_population(population)
            f.write(f'gen {generation}: atom_uniqueness: {atom_uniqueness:.4f}\n')
            print(f'gen {generation}: atom_uniqueness: {atom_uniqueness:.4f}')
            
            if diversity is not None:
                f.write(f'gen {generation}: diversity (avg tanimoto): {diversity:.4f}\n')
                print(f'gen {generation}: diversity (avg tanimoto): {diversity:.4f}')
            else:
                f.write(f'gen {generation}: diversity (avg tanimoto): N/A\n')
                print(f'gen {generation}: diversity (avg tanimoto): N/A')
            f.flush()
            
            elapsed_minutes = (end_time - start_time) / 60.0
            f.write(f'generation {generation} time (min): {elapsed_minutes:.2f}\n')
            print(f'generation {generation} time (min): {elapsed_minutes:.2f}\n')
            output_path = f'outputs/log/mols/{timestamp}_{self.args.exp_name}/'
            for i, ind in enumerate(population):
                current_atom_types = torch.argmax(ind['one_hot'].detach().cpu(), dim=-1).numpy()
                self.save_molecule_to_file(
                    positions=ind['positions'].detach().cpu().numpy(),
                    atom_types=current_atom_types,
                    output_path=output_path,
                    molecule_idx=i
                )
            
if __name__ == "__main__":
    properties = ['alpha', 'homo', 'lumo', 'gap', 'mu', 'Cv']
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='debug_')
    parser.add_argument('--generators_path', type=str, default='outputs/debug_10/qm9_latent2')
    parser.add_argument('--classifiers_path', type=str, default='qm9/property_prediction/models/exp_class_')
    parser.add_argument('--property', type=str, default=['alpha', 'mu'],
                        help="'alpha', 'homo', 'lumo', 'gap', 'mu', 'Cv'")
    parser.add_argument('--task', type=str, default='egd',
                        help='egd | egm | edm | qm9_second_half | naive')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA')
    parser.add_argument('--debug_break', type=eval, default=False)
    
    # EGD特定参数
    parser.add_argument('--t_add', type=int, default=100)
    parser.add_argument('--target_value', type=float, default=None)
    # parser.add_argument('--crossover_rate', type=float, default=0.8,
    #                     help='交叉率')
    # parser.add_argument('--mutation_rate', type=float, default=0.2,
    #                     help='变异率')
    parser.add_argument('--de_F', type=float, default=0.5,
                        help='Scale Factor')                    
    parser.add_argument('--n_generations', type=int, default=30,
                        help='evolutional generation')
    parser.add_argument('--population_size', type=int, default=16,
                        help='population size')
    parser.add_argument('--population_dir', type=str, 
                        default='/home/smj/workspace/GeoLDM-main/outputs/qm9_40000.npz', 
                        help='Initial population data file path')
    parser.add_argument('--dp', type=eval, default=True, help='True | False')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:1" if args.cuda else "cpu")
    args.device = device
    

    # 处理属性参数以构建路径和实验名称
    
    
    print(f"使用设备: {device}")
    # print(f"任务类型: {args.task}")
    print(f"parameters - t_add: {args.t_add}, population size: {args.population_size}, evolutionary generations: {args.n_generations}")
    '''
   , ['gap','mu'], ['alpha', 'mu'],
     ['homo','lumo'],['lumo','gap'],
     ['homo','gap'] ,['Cv','mu'],['lumo','mu']
    ['homo','lumo','mu'],['homo', 'lumo', 'gap'],
    ['homo','mu','alpha'],['alpha','mu','Cv']
    '''
    for property_name in [ ['homo','gap'] ]:
        args.property = property_name
        if isinstance(args.property, list):
            property_str = '_'.join(args.property)  
        else:
            property_str = str(args.property)
        print(f"property name: {args.property}")
        # 更新实验名称
        args.exp_name = f'{args.exp_name}{property_str}'
        evolver = evol_grad(args)
        evolver.run_evolution()
        args.exp_name = 'debug_'

