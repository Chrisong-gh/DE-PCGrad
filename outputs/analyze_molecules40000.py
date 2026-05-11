#!/usr/bin/env python3
"""
分析分子属性分布的独立脚本
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_molecule_properties(npz_file_path, properties=None, save_plots=True, output_dir=None):
    """
    读取npz文件中的分子数据，计算所有属性的最大值、最小值和分布
    
    Args:
        npz_file_path (str): npz文件路径
        properties (list): 要分析的属性列表，默认为None时使用所有可用属性
        save_plots (bool): 是否保存分布图
        output_dir (str): 图片保存目录，默认为None时保存到npz文件同目录
    
    Returns:
        dict: 包含所有属性统计信息的字典
    """
    print(f"正在读取分子数据: {npz_file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(npz_file_path):
        raise FileNotFoundError(f"文件不存在: {npz_file_path}")
    
    # 加载数据
    data = np.load(npz_file_path, allow_pickle=True)
    
    # 如果没有指定属性，使用默认属性列表
    if properties is None:
        properties = ['alpha', 'homo', 'lumo', 'gap', 'mu', 'Cv']
    
    # 检查哪些属性在数据中可用
    available_properties = []
    for prop in properties:
        if prop in data:
            available_properties.append(prop)
        else:
            print(f"警告: 属性 '{prop}' 在数据中不存在")
    
    if not available_properties:
        raise ValueError("没有找到任何可用的属性数据")
    
    print(f"找到 {len(data['num_atoms'])} 个分子")
    print(f"可用属性: {available_properties}")
    
    # 计算统计信息
    stats = {}
    
    for prop in available_properties:
        prop_values = data[prop]
        
        stats[prop] = {
            'min': float(np.min(prop_values)),
            'max': float(np.max(prop_values)),
            'mean': float(np.mean(prop_values)),
            'std': float(np.std(prop_values)),
            'median': float(np.median(prop_values)),
            'q25': float(np.percentile(prop_values, 25)),
            'q75': float(np.percentile(prop_values, 75)),
            'count': len(prop_values)
        }
    
    # 打印统计信息
    print("\n=== 属性统计信息 ===")
    for prop in available_properties:
        s = stats[prop]
        print(f"\n属性: {prop}")
        print(f"  数量: {s['count']}")
        print(f"  最小值: {s['min']:.6f}")
        print(f"  最大值: {s['max']:.6f}")
        print(f"  均值: {s['mean']:.6f}")
        print(f"  标准差: {s['std']:.6f}")
        print(f"  中位数: {s['median']:.6f}")
        print(f"  25%分位数: {s['q25']:.6f}")
        print(f"  75%分位数: {s['q75']:.6f}")
        print(f"  范围: [{s['min']:.6f}, {s['max']:.6f}]")
    
    # 绘制分布图
    if save_plots:
        if output_dir is None:
            output_dir = os.path.dirname(npz_file_path)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建子图
        n_props = len(available_properties)
        n_cols = 3
        n_rows = (n_props + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, prop in enumerate(available_properties):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            prop_values = data[prop]
            
            # 绘制直方图
            ax.hist(prop_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(f'{prop} 分布')
            ax.set_xlabel(f'{prop} 值')
            ax.set_ylabel('频次')
            ax.grid(True, alpha=0.3)
            
            # 添加统计线
            ax.axvline(stats[prop]['mean'], color='red', linestyle='--', 
                      label=f'均值: {stats[prop]["mean"]:.4f}')
            ax.axvline(stats[prop]['median'], color='green', linestyle='--', 
                      label=f'中位数: {stats[prop]["median"]:.4f}')
            ax.legend()
        
        # 隐藏多余的子图
        for i in range(n_props, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = os.path.join(output_dir, 'property_distributions.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n分布图已保存到: {plot_path}")
        
        # 显示图片
        plt.show()
    
    # 保存统计信息到文件
    if output_dir is None:
        output_dir = os.path.dirname(npz_file_path)
    
    stats_file = os.path.join(output_dir, 'property_statistics.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=== 分子属性统计信息 ===\n")
        f.write(f"数据文件: {npz_file_path}\n")
        f.write(f"分子总数: {len(data['num_atoms'])}\n")
        f.write(f"分析属性: {available_properties}\n\n")
        
        for prop in available_properties:
            s = stats[prop]
            f.write(f"属性: {prop}\n")
            f.write(f"  数量: {s['count']}\n")
            f.write(f"  最小值: {s['min']:.6f}\n")
            f.write(f"  最大值: {s['max']:.6f}\n")
            f.write(f"  均值: {s['mean']:.6f}\n")
            f.write(f"  标准差: {s['std']:.6f}\n")
            f.write(f"  中位数: {s['median']:.6f}\n")
            f.write(f"  25%分位数: {s['q25']:.6f}\n")
            f.write(f"  75%分位数: {s['q75']:.6f}\n")
            f.write(f"  范围: [{s['min']:.6f}, {s['max']:.6f}]\n\n")
    
    print(f"统计信息已保存到: {stats_file}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='分析分子属性分布')
    parser.add_argument('--input_file', type=str, 
                       default='outputs/qm9_40000.npz',
                       help='要分析的npz文件路径')
    parser.add_argument('--output_dir', type=str, default='outputs/qm9_40000_stats',
                       help='输出目录，默认为输入文件同目录')
    parser.add_argument('--properties', nargs='+', 
                       default=['alpha', 'homo', 'lumo', 'gap', 'mu', 'Cv'],
                       help='要分析的属性列表')
    parser.add_argument('--no_plots', action='store_true', default=False,
                       help='不生成分布图')
    
    args = parser.parse_args()
    
    # 分析分子属性
    stats = analyze_molecule_properties(
        npz_file_path=args.input_file,
        properties=args.properties,
        save_plots=not args.no_plots,
        output_dir=args.output_dir
    )
    
    print("\n=== 分析完成 ===")
    return stats


if __name__ == "__main__":
    main()