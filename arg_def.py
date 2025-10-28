from dataclasses import dataclass
import os

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """实验名称：从py文件路径提取文件名并去掉.py后缀，最终以td3_continuous_action作为实验名"""
    seed: int = 1
    """随机种子，相同的种子生成的随机数序列相同"""
    torch_deterministic: bool = True
    """强制Pytorch进行确定性计算，换取完全可复现性"""
    cuda: bool = True
    """启用GPU加速"""
    capture_video: bool = False
    """是否录制智能体的视频，保存到videos/文件夹"""
    save_model: bool = True
    """是否本地保存模型到`runs/{run_name}`文件夹"""

    # Algorithm specific arguments
    env_id: str = "Ant-v4"
    """任务环境Ant-v4"""
    total_timesteps: int = 500000
    """总训练步数"""
    learning_rate: float = 3e-4
    """优化器的学习率"""
    num_envs: int = 1
    """并行环境数量，可加速数据收集但耗内存"""
    buffer_size: int = int(1e6)
    """经验回放池大小"""
    gamma: float = 0.99
    """折扣率gamma"""
    tau: float = 0.005
    """目标网络软更新的系数"""
    batch_size: int = 256
    """每次从经验池中采样的批量大小"""
    policy_noise: float = 0.2
    """在目标价值网络计算Q值时给动作添加的噪声，扩大可能的打分范围"""
    exploration_noise: float = 0.1
    """给策略网络输出动作添加的噪声，然后执行动作获得经验，促进探索"""
    learning_starts: int = 25e3
    """在多少步之后正式开始训练"""
    policy_frequency: int = 2
    """策略网络更新频率（延迟更新）"""
    noise_clip: float = 0.5
    """噪声裁剪范围[-0.5,0.5]，噪声是0.2，则实际范围是[-0.1,0.1];噪声是0.1，则范围[-0.05,0.05]"""
    beta_init: float = 0.5

def get_args():#提供接口函数
    import tyro
    return tyro.cli(Args)#使用tyro传参，所有超参数打包输出

if __name__ == "__main__":#测试参数解析
    args = get_args()
    print(args)