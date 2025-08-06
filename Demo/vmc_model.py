"""
VMC (Variational Memory Compression) 模型架构
包含 V (Variational), M (Memory), C (Controller) 三个核心组件
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from config import VMCConfig

class VariationalEncoder(nn.Module):
    """V - 变分编码器组件"""
    
    def __init__(self, input_dim, latent_dim, hidden_dims):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 构建编码器网络
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # 变分参数层
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入数据 [batch_size, input_dim]
            
        Returns:
            z: 重参数化采样 [batch_size, latent_dim]
            mu: 均值 [batch_size, latent_dim]
            logvar: 对数方差 [batch_size, latent_dim]
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # 重参数化技巧
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar
    
    def reparameterize(self, mu, logvar):
        """重参数化采样"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class MemoryModule(nn.Module):
    """M - 记忆模块（混合高斯分布）"""
    
    def __init__(self, memory_size, memory_dim, num_gaussians):
        super().__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_gaussians = num_gaussians
        
        # 记忆槽参数
        self.memory_keys = nn.Parameter(torch.randn(memory_size, memory_dim))
        self.memory_values = nn.Parameter(torch.randn(memory_size, memory_dim))
        
        # 混合高斯分布参数
        self.gaussian_means = nn.Parameter(torch.randn(num_gaussians, memory_dim))
        self.gaussian_logvars = nn.Parameter(torch.zeros(num_gaussians, memory_dim))
        self.gaussian_weights = nn.Parameter(torch.ones(num_gaussians))
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=memory_dim,
            num_heads=4,
            batch_first=True
        )
        
    def forward(self, query):
        """
        记忆检索和更新
        
        Args:
            query: 查询向量 [batch_size, memory_dim]
            
        Returns:
            retrieved_memory: 检索到的记忆 [batch_size, memory_dim]
            attention_weights: 注意力权重 [batch_size, memory_size]
            mixture_probs: 混合概率 [batch_size, num_gaussians]
        """
        batch_size = query.size(0)
        
        # 扩展记忆键以匹配批次大小
        keys = self.memory_keys.unsqueeze(0).expand(batch_size, -1, -1)
        values = self.memory_values.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 注意力机制检索记忆
        query_expanded = query.unsqueeze(1)  # [batch_size, 1, memory_dim]
        retrieved_memory, attention_weights = self.attention(
            query_expanded, keys, values
        )
        retrieved_memory = retrieved_memory.squeeze(1)  # [batch_size, memory_dim]
        attention_weights = attention_weights.squeeze(1)  # [batch_size, memory_size]
        
        # 计算混合高斯分布概率
        mixture_probs = self.compute_mixture_probabilities(query)
        
        return retrieved_memory, attention_weights, mixture_probs
    
    def compute_mixture_probabilities(self, query):
        """计算混合高斯分布概率"""
        batch_size = query.size(0)
        
        # 计算每个高斯分量的概率
        log_probs = []
        weights = F.softmax(self.gaussian_weights, dim=0)
        
        for i in range(self.num_gaussians):
            mean = self.gaussian_means[i]
            logvar = self.gaussian_logvars[i]
            
            # 计算高斯概率密度
            diff = query - mean.unsqueeze(0)
            var = torch.exp(logvar).unsqueeze(0)
            log_prob = -0.5 * torch.sum((diff ** 2) / var + logvar.unsqueeze(0), dim=1)
            log_prob = log_prob + torch.log(weights[i])
            log_probs.append(log_prob)
        
        log_probs = torch.stack(log_probs, dim=1)  # [batch_size, num_gaussians]
        probs = F.softmax(log_probs, dim=1)
        
        return probs
    
    def update_memory(self, queries, values, learning_rate=0.01):
        """软更新记忆"""
        with torch.no_grad():
            # 计算注意力权重
            batch_size = queries.size(0)
            keys = self.memory_keys.unsqueeze(0).expand(batch_size, -1, -1)
            
            # 计算相似度
            similarities = torch.bmm(
                queries.unsqueeze(1), 
                keys.transpose(1, 2)
            ).squeeze(1)  # [batch_size, memory_size]
            
            attention_weights = F.softmax(similarities, dim=1)
            
            # 软更新记忆值
            for i in range(self.memory_size):
                weight = attention_weights[:, i].mean()
                if weight > 0.1:  # 只更新被显著激活的记忆槽
                    update = values.mean(0) * learning_rate * weight
                    self.memory_values.data[i] += update

class Controller(nn.Module):
    """C - 控制器组件"""
    
    def __init__(self, variational_dim, memory_dim, hidden_dim, output_dim):
        super().__init__()
        self.variational_dim = variational_dim
        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 融合网络
        self.fusion_layer = nn.Sequential(
            nn.Linear(variational_dim + memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(variational_dim + memory_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, variational_code, memory_code):
        """
        控制器前向传播
        
        Args:
            variational_code: 变分编码 [batch_size, variational_dim]
            memory_code: 记忆编码 [batch_size, memory_dim]
            
        Returns:
            output: 最终输出 [batch_size, output_dim]
            gate_weight: 门控权重 [batch_size, 1]
        """
        # 拼接变分编码和记忆编码
        combined = torch.cat([variational_code, memory_code], dim=1)
        
        # 门控机制
        gate_weight = self.gate(combined)
        
        # 融合处理
        fused = self.fusion_layer(combined)
        
        # 门控输出
        gated_output = fused * gate_weight
        
        # 最终输出
        output = self.output_layer(gated_output)
        
        return output, gate_weight

class VMC(nn.Module):
    """完整的VMC模型"""
    
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = VMCConfig()
        
        self.config = config
        
        # 初始化三个核心组件
        self.variational_encoder = VariationalEncoder(
            input_dim=config.input_dim,
            latent_dim=config.variational_dim,
            hidden_dims=config.encoder_hidden_dims
        )
        
        self.memory_module = MemoryModule(
            memory_size=config.memory_size,
            memory_dim=config.memory_dim,
            num_gaussians=config.num_gaussians
        )
        
        self.controller = Controller(
            variational_dim=config.variational_dim,
            memory_dim=config.memory_dim,
            hidden_dim=config.controller_hidden_dim,
            output_dim=config.controller_output_dim
        )
        
        # 投影层：将变分编码投影到记忆空间
        self.var_to_memory = nn.Linear(config.variational_dim, config.memory_dim)
        
    def forward(self, x):
        """
        VMC前向传播
        
        Args:
            x: 输入数据 [batch_size, input_dim]
            
        Returns:
            output: 分类输出 [batch_size, output_dim]
            components: 各组件的输出和中间结果
        """
        # V - 变分编码
        var_code, var_mu, var_logvar = self.variational_encoder(x)
        
        # 将变分编码投影到记忆空间
        memory_query = self.var_to_memory(var_code)
        
        # M - 记忆检索
        retrieved_memory, attention_weights, mixture_probs = self.memory_module(memory_query)
        
        # C - 控制器处理
        output, gate_weight = self.controller(var_code, retrieved_memory)
        
        # 返回所有组件的结果用于可视化
        components = {
            'variational': {
                'code': var_code,
                'mu': var_mu,
                'logvar': var_logvar
            },
            'memory': {
                'retrieved': retrieved_memory,
                'attention_weights': attention_weights,
                'mixture_probs': mixture_probs,
                'query': memory_query
            },
            'controller': {
                'output': output,
                'gate_weight': gate_weight
            }
        }
        
        return output, components
    
    def compute_loss(self, x, y, output, components):
        """计算VMC总损失"""
        # 分类损失
        classification_loss = F.cross_entropy(output, y)
        
        # KL散度损失（变分部分）
        var_mu = components['variational']['mu']
        var_logvar = components['variational']['logvar']
        kl_loss = -0.5 * torch.sum(1 + var_logvar - var_mu.pow(2) - var_logvar.exp())
        kl_loss = kl_loss / x.size(0)  # 平均到batch
        
        # 记忆正则化损失
        memory_loss = self.compute_memory_regularization()
        
        # 总损失
        total_loss = (classification_loss + 
                     self.config.kl_beta * kl_loss + 
                     self.config.memory_beta * memory_loss)
        
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'kl_loss': kl_loss,
            'memory_loss': memory_loss
        }
    
    def compute_memory_regularization(self):
        """计算记忆正则化损失"""
        # 记忆槽多样性损失
        memory_keys = self.memory_module.memory_keys
        memory_values = self.memory_module.memory_values
        
        # 计算记忆键之间的相似性，鼓励多样性
        key_sim = torch.mm(memory_keys, memory_keys.t())
        key_sim = key_sim - torch.eye(key_sim.size(0)).to(key_sim.device)
        diversity_loss = torch.sum(key_sim ** 2)
        
        # 混合高斯分布的正则化
        gaussian_weights = F.softmax(self.memory_module.gaussian_weights, dim=0)
        entropy_loss = -torch.sum(gaussian_weights * torch.log(gaussian_weights + 1e-8))
        
        return diversity_loss * 0.01 - entropy_loss * 0.1

def create_vmc_model(config=None):
    """创建VMC模型的工厂函数"""
    if config is None:
        config = VMCConfig()
    
    model = VMC(config).to(config.device)
    
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🏗️  VMC模型创建完成")
    print(f"   总参数数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    print(f"   变分编码维度: {config.variational_dim}")
    print(f"   记忆槽数量: {config.memory_size}")
    print(f"   记忆维度: {config.memory_dim}")
    print(f"   混合高斯组件数: {config.num_gaussians}")
    
    return model

def test_vmc_model():
    """测试VMC模型"""
    print("🧪 测试VMC模型...")
    
    config = VMCConfig()
    model = create_vmc_model(config)
    
    # 创建测试输入
    batch_size = 4
    x = torch.randn(batch_size, config.input_dim).to(config.device)
    y = torch.randint(0, config.controller_output_dim, (batch_size,)).to(config.device)
    
    # 前向传播
    output, components = model(x)
    
    # 计算损失
    losses = model.compute_loss(x, y, output, components)
    
    print(f"✅ 模型测试通过")
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {output.shape}")
    print(f"   总损失: {losses['total_loss'].item():.4f}")
    print(f"   分类损失: {losses['classification_loss'].item():.4f}")
    print(f"   KL损失: {losses['kl_loss'].item():.4f}")
    print(f"   记忆损失: {losses['memory_loss'].item():.4f}")

if __name__ == "__main__":
    test_vmc_model()