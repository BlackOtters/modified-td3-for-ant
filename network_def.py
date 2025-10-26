import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QNetwork(nn.Module):#价值网络
    def __init__(self, observation_space, action_space):##输入环境的状态空间和动作空间
        super().__init__()
        # 通过空间定义直接初始化（不依赖env对象）
        obs_dim = np.prod(observation_space.shape)#prod()将多维形状转换为相应数量的整数，得到总变量数
        act_dim = np.prod(action_space.shape)
        
        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)#全连接神经网络
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):#定义前向传播过程
        x = torch.cat([x, a], 1)#输入状态和动作
        x = F.relu(self.fc1(x))#激活函数
        x = F.relu(self.fc2(x))#激活函数
        x = self.fc3(x)#无激活函数，Q值可以是任意实数范围
        return x


class Actor(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        obs_dim = np.prod(observation_space.shape)
        act_dim = np.prod(action_space.shape)
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, act_dim)#输出动作
        # action rescaling#动作空间规范化,也就是计算两个不参与梯度更新的动作变换参数,通过注册不参与梯度更新
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (action_space.high - action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )#动作缩放比例
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (action_space.high + action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )#动作取值范围的中心点相对原点的偏差

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias#返回大小正常的原动作

class AntFeatureExtractor(nn.Module):
    def __init__(self, obs_dim=27):
        super().__init__()
        # 分模态处理Ant的27维观测
        self.torso_net = nn.Sequential(
            nn.Linear(7, 32),  # 躯干位置+旋转(7D)
            nn.LayerNorm(32),
            nn.GELU()
        )
        self.joint_net = nn.Sequential(
            nn.Linear(16, 64),  # 关节角度+速度(8+8=16D)
            nn.LayerNorm(64),
            nn.GELU()
        )
        self.contact_net = nn.Sequential(
            nn.Linear(4, 16),   # 接触力(4D)
            nn.SiLU()
        )
        # 融合层（输出256维与原有网络兼容）
        self.fusion = nn.Sequential(
            nn.Linear(32+64+16, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )

    def forward(self, x):
        torso = x[..., :7]
        joints = x[..., 7:23]
        contacts = x[..., 23:]
        return self.fusion(torch.cat([
            self.torso_net(torso),
            self.joint_net(joints),
            self.contact_net(contacts)
        ], dim=-1))

class EnhancedActor(Actor):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        # 替换原第一层为特征提取器
        self.feature_extractor = AntFeatureExtractor(observation_space.shape[0])
        # 保持原有输出层结构不变
        self.fc1 = nn.Linear(256, 256)  # 与特征提取器输出维度匹配
        
    def forward(self, x):
        features = self.feature_extractor(x)
        return super().forward(features)  # 复用父类的动作缩放逻辑

class EnhancedQNetwork(QNetwork):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        self.feature_extractor = AntFeatureExtractor(observation_space.shape[0])
        # 修改输入层结构但保持接口
        self.fc1 = nn.Linear(256 + action_space.shape[0], 256)
        
    def forward(self, x, a):
        features = self.feature_extractor(x)
        return super().forward(features, a)  # 复用父类的Q值计算

def create_networks(observation_space, action_space, device):
    """完全兼容原有调用方式"""
    actor = EnhancedActor(observation_space, action_space).to(device)
    qf1 = EnhancedQNetwork(observation_space, action_space).to(device)
    qf2 = EnhancedQNetwork(observation_space, action_space).to(device)
    
    # 目标网络初始化（与原有逻辑一致）
    target_actor = EnhancedActor(observation_space, action_space).to(device)
    qf1_target = EnhancedQNetwork(observation_space, action_space).to(device)
    qf2_target = EnhancedQNetwork(observation_space, action_space).to(device)
    
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    
    return {  # 保持完全相同的返回结构
        "actor": actor,
        "qf1": qf1,
        "qf2": qf2,
        "qf1_target": qf1_target,
        "qf2_target": qf2_target,
        "target_actor": target_actor
    }

if __name__ == "__main__":
    # 测试网络结构
    from gymnasium.spaces import Box
    obs_space = Box(low=-1, high=1, shape=(10,))
    act_space = Box(low=-2, high=2, shape=(2,))
    
    device = torch.device("cpu")
    networks = create_networks(obs_space, act_space, device)
    
    print("Actor输出示例:", 
          networks["actor"](torch.randn(1, 10)))#randn随机生成一个张量，然后调用forward方法进行前向计算
    print("QNetwork输出示例:", 
          networks["qf1"](torch.randn(1, 10), torch.randn(1, 2)))