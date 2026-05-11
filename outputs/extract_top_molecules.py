import numpy as np
import os
from typing import Dict, List, Tuple

topN = 10
def load_qm9_data(file_path: str) -> Dict:
    """加载QM9数据文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    data = np.load(file_path, allow_pickle=True)
    return data

def get_drug_optimal_direction(property_name: str) -> bool:
    """
    获取药物分子设计中各属性的最优排序方向
    
    Args:
        property_name: 属性名称
        
    Returns:
        bool: True表示升序(值越小越好)，False表示降序(值越大越好)
    """
    # 基于药物分子设计的最优化方向
    drug_optimal_directions = {
        'gap': False,      # HOMO-LUMO gap: 越大越好 (更高稳定性)
        'homo': True,      # HOMO能量: 越低越好 (更好的供电子能力)
        'lumo': False,     # LUMO能量: 越高越好 (更好的受电子能力)
        'alpha': False,    # 极化率: 适度增加更好 (增强分子间相互作用)
        'mu': False,       # 偶极矩: 适度增加更好 (改善溶解性和相互作用)
        'Cv': False        # 热容: 适中偏高更好 (热力学稳定性)
    }
    
    return drug_optimal_directions.get(property_name, True)

def get_property_description(property_name: str) -> str:
    """获取属性的药物化学意义描述"""
    descriptions = {
        'gap': 'HOMO-LUMO能隙 - 分子稳定性指标，较大值表示更高的动力学稳定性和较低的反应性',
        'homo': 'HOMO能量 - 最高占据分子轨道能量，较低值有利于供电子相互作用',
        'lumo': 'LUMO能量 - 最低未占据分子轨道能量，较高值有利于受电子相互作用',
        'alpha': '极化率 - 分子对外电场的响应能力，适度增加有利于分子间相互作用',
        'mu': '偶极矩 - 分子极性指标，影响溶解性和生物大分子结合',
        'Cv': '定容热容 - 分子热力学性质，反映在生理温度下的稳定性'
    }
    return descriptions.get(property_name, f'{property_name} - 分子属性')

def get_top_molecules_by_property(data: Dict, property_name: str, top_k: int = 100, 
                                 use_drug_optimal: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据指定属性获取排名前k的分子
    
    Args:
        data: QM9数据字典
        property_name: 属性名称 ('alpha', 'gap', 'homo', 'lumo', 'mu', 'Cv')
        top_k: 取前k个分子
        use_drug_optimal: 是否使用药物分子最优化方向
    
    Returns:
        indices: 排序后的分子索引
        values: 对应的属性值
    """
    if property_name not in data:
        raise ValueError(f"属性 '{property_name}' 不存在于数据中")
    
    property_values = data[property_name]
    
    # 确定排序方向
    if use_drug_optimal:
        ascending = get_drug_optimal_direction(property_name)
    else:
        ascending = True  # 默认升序
    
    # 排序获取索引
    if ascending:
        sorted_indices = np.argsort(property_values)
    else:
        sorted_indices = np.argsort(property_values)[::-1]
    
    # 取前k个
    top_indices = sorted_indices[:top_k]
    top_values = property_values[top_indices]
    
    return top_indices, top_values

def save_molecule_to_file(positions: np.ndarray, charges: np.ndarray, atom_decoder: List[str], 
                         output_path: str, molecule_idx: int, property_value: float = None, 
                         property_name: str = None):
    """
    将分子保存为指定格式的文件
    
    Args:
        positions: 原子坐标 (n_atoms, 3)
        charges: 原子序数 (n_atoms,) - 注意：这里是原子序数，不是编码索引
        atom_decoder: 原子类型解码器
        output_path: 输出目录路径
        molecule_idx: 分子索引
        property_value: 属性值
        property_name: 属性名称
    """
    os.makedirs(output_path, exist_ok=True)
    
    filename = os.path.join(output_path, f"molecule_{molecule_idx:03d}.txt")
    
    # 原子序数到原子类型的映射
    atomic_number_to_symbol = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
    
    with open(filename, 'w') as f:
        # 写入原子数量
        n_atoms = len(positions)
        f.write(f"{n_atoms}\n")
        
        # 写入属性信息作为注释
        if property_value is not None and property_name is not None:
            f.write(f"# {property_name}: {property_value:.6f}\n")
        else:
            f.write("\n")
        
        # 写入每个原子的信息
        for i in range(n_atoms):
            atomic_number = int(charges[i])
            atom_type = atomic_number_to_symbol.get(atomic_number, f'X{atomic_number}')
            x, y, z = positions[i]
            f.write(f"{atom_type} {x:.9f} {y:.9f} {z:.9f}\n")

def extract_drug_optimal_molecules(train_file_path: str, property_name: str, 
                                 output_dir: str, top_k: int = 100):
    """
    从QM9训练数据中提取指定属性在药物分子设计中最优的前k个分子
    
    Args:
        train_file_path: train.npz文件路径
        property_name: 属性名称 ('alpha', 'gap', 'homo', 'lumo', 'mu', 'Cv')
        output_dir: 输出目录
        top_k: 提取前k个分子
    """
    # QM9原子类型解码器
    atom_decoder = ['H', 'C', 'N', 'O', 'F']
    
    print(f"正在加载数据文件: {train_file_path}")
    data = load_qm9_data(train_file_path)
    
    print(f"数据文件包含的键: {list(data.keys())}")
    
    # 获取属性描述
    prop_description = get_property_description(property_name)
    optimal_direction = get_drug_optimal_direction(property_name)
    direction_text = "越小越好" if optimal_direction else "越大越好"
    
    print(f"\n属性说明: {prop_description}")
    print(f"药物分子最优方向: {direction_text}")
    
    # 获取药物分子最优的前k个分子
    print(f"正在获取{property_name}属性在药物分子设计中最优的前{top_k}个分子...")
    top_indices, top_values = get_top_molecules_by_property(data, property_name, top_k, use_drug_optimal=True)
    
    print(f"找到{len(top_indices)}个分子")
    print(f"{property_name}值范围: {top_values.min():.6f} 到 {top_values.max():.6f}")
    
    # 创建输出目录
    property_output_dir = os.path.join(output_dir, f"drug_optimal_{property_name}_top_{top_k}")
    os.makedirs(property_output_dir, exist_ok=True)
    
    # 保存每个分子
    positions = data['positions']
    charges = data['charges']
    num_atoms = data['num_atoms']
    
    print(f"正在保存分子到: {property_output_dir}")
    
    for i, mol_idx in enumerate(top_indices):
        # 获取分子的原子数量
        n_atoms = num_atoms[mol_idx]
        
        # 直接从positions和charges数组中获取对应分子的数据
        # positions和charges的形状是 (n_molecules, max_atoms, 3) 和 (n_molecules, max_atoms)
        mol_positions = positions[mol_idx][:n_atoms]  # 只取实际原子数量的坐标
        mol_charges = charges[mol_idx][:n_atoms]      # 只取实际原子数量的电荷
        
        save_molecule_to_file(mol_positions, mol_charges, atom_decoder, 
                            property_output_dir, i, top_values[i], property_name)
        
        if (i + 1) % 10 == 0:
            print(f"已保存 {i + 1}/{len(top_indices)} 个分子")
    
    # 保存详细的属性值信息
    info_file = os.path.join(property_output_dir, "drug_optimal_info.txt")
    with open(info_file, 'w') as f:
        f.write("=== 药物分子设计最优化分子筛选结果 ===\n\n")
        f.write(f"属性名称: {property_name}\n")
        f.write(f"属性描述: {prop_description}\n")
        f.write(f"药物分子最优方向: {direction_text}\n")
        f.write(f"筛选分子数量: {len(top_indices)}\n")
        f.write(f"属性值范围: {top_values.min():.6f} 到 {top_values.max():.6f}\n\n")
        
        f.write("分子列表 (按药物分子最优性排序):\n")
        f.write("-" * 60 + "\n")
        for i, (mol_idx, value) in enumerate(zip(top_indices, top_values)):
            f.write(f"排名 {i+1:3d}: molecule_{i:03d}.txt (原始索引: {mol_idx:5d}) -> {value:.6f}\n")
    
    print(f"完成！已保存{len(top_indices)}个药物分子最优化分子到 {property_output_dir}")
    print(f"详细信息已保存到 {info_file}")

def extract_all_drug_optimal_molecules():
    """提取所有属性在药物分子设计中最优的前20个分子"""
    train_file = "/home/smj/workspace/GeoLDM-main/qm9/temp/qm9/train.npz"
    output_base_dir = "/home/smj/workspace/GeoLDM-main/outputs/drug_optimal_molecules"
    
    # QM9的主要属性
    properties = ['alpha', 'gap', 'homo', 'lumo', 'mu', 'Cv']
    
    print("=== QM9数据集药物分子最优化筛选 ===\n")
    
    for prop in properties:
        try:
            print(f"\n{'='*60}")
            print(f"处理属性: {prop}")
            print(f"{'='*60}")
            
            extract_drug_optimal_molecules(
                train_file, prop, output_base_dir, top_k=topN
            )
            
        except Exception as e:
            print(f"处理属性 {prop} 时出错: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("所有属性处理完成！")
    print(f"结果保存在: {output_base_dir}")
    print(f"每个属性提取前{topN}个分子")
    print(f"{'='*60}")

def extract_top_molecules_by_property(train_file_path: str, property_name: str, 
                                    output_dir: str, top_k: int = topN, 
                                    ascending: bool = True):
    """
    从QM9训练数据中提取指定属性排名前k的分子并保存 (保留原有功能)
    
    Args:
        train_file_path: train.npz文件路径
        property_name: 属性名称 ('alpha', 'gap', 'homo', 'lumo', 'mu', 'Cv')
        output_dir: 输出目录
        top_k: 提取前k个分子
        ascending: True表示从小到大排序，False表示从大到小排序
    """
    # 注意：这里不再需要atom_decoder，因为我们直接使用原子序数映射
    
    print(f"正在加载数据文件: {train_file_path}")
    data = load_qm9_data(train_file_path)
    
    print(f"数据文件包含的键: {list(data.keys())}")
    
    # 获取排名前k的分子
    print(f"正在获取{property_name}属性排名前{top_k}的分子...")
    
    if property_name not in data:
        raise ValueError(f"属性 '{property_name}' 不存在于数据中")
    
    property_values = data[property_name]
    
    # 排序获取索引
    if ascending:
        sorted_indices = np.argsort(property_values)
    else:
        sorted_indices = np.argsort(property_values)[::-1]
    
    # 取前k个
    top_indices = sorted_indices[:top_k]
    top_values = property_values[top_indices]
    
    print(f"找到{len(top_indices)}个分子")
    print(f"{property_name}值范围: {top_values.min():.6f} 到 {top_values.max():.6f}")
    
    # 创建输出目录
    property_output_dir = os.path.join(output_dir, f"top_{top_k}_{property_name}_{'asc' if ascending else 'desc'}")
    os.makedirs(property_output_dir, exist_ok=True)
    
    # 保存每个分子
    positions = data['positions']
    charges = data['charges']
    num_atoms = data['num_atoms']
    
    print(f"正在保存分子到: {property_output_dir}")
    
    for i, mol_idx in enumerate(top_indices):
        # 获取分子的原子数量
        n_atoms = num_atoms[mol_idx]
        
        # 修正：直接从positions和charges数组中获取对应分子的数据
        mol_positions = positions[mol_idx][:n_atoms]  # 只取实际原子数量的坐标
        mol_charges = charges[mol_idx][:n_atoms]      # 只取实际原子数量的电荷
        
        save_molecule_to_file(mol_positions, mol_charges, atom_decoder, 
                            property_output_dir, i, top_values[i], property_name)
        
        if (i + 1) % 10 == 0:
            print(f"已保存 {i + 1}/{len(top_indices)} 个分子")
    
    # 保存属性值信息
    info_file = os.path.join(property_output_dir, "property_info.txt")
    with open(info_file, 'w') as f:
        f.write(f"Property: {property_name}\n")
        f.write(f"Number of molecules: {len(top_indices)}\n")
        f.write(f"Sorting order: {'ascending' if ascending else 'descending'}\n")
        f.write(f"Value range: {top_values.min():.6f} to {top_values.max():.6f}\n\n")
        f.write("Molecule Index -> Property Value:\n")
        for i, (mol_idx, value) in enumerate(zip(top_indices, top_values)):
            f.write(f"molecule_{i:03d}.txt (original_idx: {mol_idx}) -> {value:.6f}\n")
    
    print(f"完成！已保存{len(top_indices)}个分子到 {property_output_dir}")
    print(f"属性信息已保存到 {info_file}")

def main():
    """主函数 - 默认执行药物分子最优化筛选"""
    extract_all_drug_optimal_molecules()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "drug_optimal":
            # 药物分子最优化筛选
            if len(sys.argv) > 2:
                property_name = sys.argv[2]
                train_file = "/home/smj/workspace/GeoLDM-main/qm9/temp/qm9/train.npz"
                output_dir = "/home/smj/workspace/GeoLDM-main/outputs/drug_optimal_molecules"
                extract_drug_optimal_molecules(train_file, property_name, output_dir, 100)
            else:
                extract_all_drug_optimal_molecules()
                
        elif command == "custom":
            # 自定义排序方向
            if len(sys.argv) > 3:
                property_name = sys.argv[2]
                ascending = sys.argv[3].lower() == 'true'
                
                train_file = "/home/smj/workspace/GeoLDM-main/qm9/temp/qm9/train.npz"
                output_dir = "/home/smj/workspace/GeoLDM-main/outputs/top_molecules"
                
                extract_top_molecules_by_property(train_file, property_name, output_dir, 100, ascending)
            else:
                print("用法: python extract_top_molecules.py custom <property_name> <true/false>")
        else:
            print("可用命令:")
            print("  drug_optimal [property_name] - 药物分子最优化筛选")
            print("  custom <property_name> <true/false> - 自定义排序方向")
    else:
        main()