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

def create_networks(observation_space, action_space, device):
    """创建所有网络并移动到指定设备的工厂函数"""
    actor = Actor(observation_space, action_space).to(device)
    qf1 = QNetwork(observation_space, action_space).to(device)#两个Q网络和两个目标网络，取较小值减少高估
    qf2 = QNetwork(observation_space, action_space).to(device)
    qf1_target = QNetwork(observation_space, action_space).to(device)
    qf2_target = QNetwork(observation_space, action_space).to(device)
    target_actor = Actor(observation_space, action_space).to(device)
    
    # 初始化目标网络参数，state_dict是可学习参数快照，用来初始时同步参数
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    
    return {
        "actor": actor,
        "qf1": qf1,
        "qf2": qf2,
        "qf1_target": qf1_target,
        "qf2_target": qf2_target,
        "target_actor": target_actor
    }#返回字典

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