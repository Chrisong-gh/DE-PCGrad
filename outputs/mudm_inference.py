import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from geoldm.models import EGNN, Decoder, Encoder  # 假设从GEOLDM导入模型
from geoldm.utils import load_pretrained_model  # GEOLDM的模型加载工具
import numpy as np
from sklearn.metrics import pairwise_distances

# -------------------------- 1. 加载预训练组件 --------------------------
class MUDM:
    def __init__(self, geoldm_ckpt_path, prop_predictor_ckpt_path, device='cuda'):
        self.device = device
        
        # 1.1 加载GEOLDM预训练模型（冻结参数）
        self.denoise_net = load_pretrained_model(geoldm_ckpt_path, 'denoiser').to(device).eval()  # 去噪网络
        self.encoder = load_pretrained_model(geoldm_ckpt_path, 'encoder').to(device).eval()      # 编码器
        self.decoder = load_pretrained_model(geoldm_ckpt_path, 'decoder').to(device).eval()      # 解码器
        
        # 1.2 加载预训练性质预测器（如偶极矩预测器，GNN结构）
        self.prop_predictor = torch.load(prop_predictor_ckpt_path).to(device).eval()
        
        # 1.3 扩散参数（与GEOLDM保持一致）
        self.T = 1000  # 总扩散步数
        self.beta = torch.linspace(1e-4, 0.02, self.T, device=device)  # 噪声系数
        self.alpha = 1 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)  # 累积乘积

    # -------------------------- 2. 进化操作（潜在空间） --------------------------
    def crossover(self, z1, z2, prob=0.8):
        """单点交叉：交换两个潜在变量的部分维度"""
        if torch.rand(1) < prob:
            split_dim = torch.randint(1, z1.shape[1]-1, (1,))[0]  # 随机分割点
            z_cross = torch.cat([z1[:, :split_dim], z2[:, split_dim:]], dim=1)
            return z_cross
        return z1  # 不交叉时返回原变量

    def mutate(self, z, sigma=0.1, prob=0.5):
        """高斯变异：为潜在变量添加噪声"""
        if torch.rand(1) < prob:
            noise = torch.randn_like(z) * sigma
            return z + noise
        return z  # 不变异时返回原变量

    def select(self, population, fitness, top_k=0.5):
        """选择高适应度个体（基于SPEA2简化版）"""
        k = int(len(population) * top_k)
        sorted_idx = torch.argsort(fitness)  # 升序（损失越小越优）
        return [population[i] for i in sorted_idx[:k]]

    # -------------------------- 3. 条件梯度计算（MUDM核心） --------------------------
    def compute_cond_grad(self, z_t, target_y):
        """计算性质损失对潜在变量z_t的梯度（条件梯度）"""
        z_t.requires_grad_(True)
        
        # 解码潜在变量到原子空间（获取分子结构）
        with torch.no_grad():
            x, h = self.decoder(z_t)  # x: [batch, N, 3]坐标; h: [batch, N, d]特征
        
        # 预测分子性质并计算损失
        pred_y = self.prop_predictor(x, h)  # 性质预测器输入原子坐标和特征
        loss = F.mse_loss(pred_y, target_y.repeat(z_t.shape[0], 1))  # 与目标性质的MSE
        
        # 反向传播求梯度（d(loss)/dz_t）
        loss.backward()
        cond_grad = z_t.grad.detach()  # 条件梯度 = 损失对z_t的梯度
        z_t.requires_grad_(False)
        
        return cond_grad

    # -------------------------- 4. 反向采样主流程 --------------------------
    def sample(self, target_y, population_size=20, semantic_start_t=400):
        """
        生成符合目标性质的分子
        target_y: 目标性质（如偶极矩=3.0D）
        population_size: 进化种群规模
        semantic_start_t: 语义阶段起始步数（t≤此值时启用引导）
        """
        # 初始化种群（潜在空间噪声变量，t=T）
        z_dim = 128  # GEOLDM的latent维度（需与预训练模型一致）
        population = [torch.randn(1, z_dim, device=self.device) for _ in range(population_size)]
        
        # 反向采样循环（从t=T到t=0）
        for t in range(self.T, 0, -1):
            current_beta = self.beta[t-1]
            current_alpha = self.alpha[t-1]
            current_alpha_cumprod = self.alpha_cumprod[t-1]
            dt = 1.0 / self.T  # 离散化时间步长
            
            # 生成子种群（进化操作）
            offspring = []
            for z in population:
                # 随机选择配偶进行交叉
                partner = population[torch.randint(0, len(population), (1,))[0]]
                z_cross = self.crossover(z, partner)
                # 变异
                z_mutated = self.mutate(z_cross, sigma=0.1 * (t / self.T))  # 变异强度随t衰减
                offspring.append(z_mutated)
            
            # 去噪+引导
            new_population = []
            fitness = []
            for z_t in offspring:
                # 时间步嵌入（GEOLDM的时间编码）
                t_emb = torch.tensor([t], device=self.device).repeat(z_t.shape[0], 1)
                
                # 步骤1：GEOLDM基础去噪
                with torch.no_grad():
                    epsilon_hat = self.denoise_net(z_t, t_emb)  # 预测噪声
                    # 计算基础去噪后的z_{t-1}
                    z0_hat = (z_t - torch.sqrt(1 - current_alpha_cumprod) * epsilon_hat) / torch.sqrt(current_alpha_cumprod)
                    z_denoised = torch.sqrt(current_alpha) * z0_hat + torch.sqrt(1 - current_alpha) * epsilon_hat
                    # 加扩散项（随机噪声）
                    if t > 1:
                        z_denoised += torch.sqrt(current_beta) * torch.randn_like(z_t)
                
                # 步骤2：MUDM条件梯度引导（仅语义阶段）
                if t <= semantic_start_t:
                    cond_grad = self.compute_cond_grad(z_denoised, target_y)
                    z_guided = z_denoised + current_beta * cond_grad * dt  # 叠加条件梯度
                else:
                    z_guided = z_denoised  # 混沌阶段不引导
                
                new_population.append(z_guided)
                
                # 计算适应度（性质损失，用于选择）
                with torch.no_grad():
                    x, h = self.decoder(z_guided)
                    pred_y = self.prop_predictor(x, h)
                    fit = F.mse_loss(pred_y, target_y).item()
                    fitness.append(fit)
            
            # 选择高适应度个体进入下一代
            population = self.select(new_population, fitness)
        
        # 最终解码最优个体
        best_z = population[0]  # 选择适应度最高的
        with torch.no_grad():
            x, h = self.decoder(best_z)  # 原子坐标x和特征h
        return x[0], h[0]  # 返回单个分子

# -------------------------- 5. 使用示例 --------------------------
if __name__ == "__main__":
    # 配置路径（需替换为实际下载的模型路径）
    GEOLDM_CKPT = "./outputs/pretrained/qm9_pretrained/denoiser.ckpt"
    PROP_PREDICTOR_CKPT = "./pretrained/property_predictor_dipole.pt"  # 预训练的偶极矩预测器
    
    # 初始化MUDM
    mudm = MUDM(GEOLDM_CKPT, PROP_PREDICTOR_CKPT)
    
    # 目标性质：偶极矩=3.0 Debye
    target_dipole = torch.tensor([3.0], device='cuda')
    
    # 生成分子
    molecule_coords, molecule_features = mudm.sample(target_dipole)
    
    # 输出结果（原子坐标和特征）
    print("生成的分子原子坐标：\n", molecule_coords.cpu().numpy())
    print("生成的分子原子特征：\n", molecule_features.cpu().numpy())
