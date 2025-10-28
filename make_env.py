import gymnasium as gym
import torch
from arg_def import get_args
from gymnasium import Wrapper
import numpy as np

args = get_args()

class AntCurriculum:
    def __init__(self, total_steps=1e6):#默认学习十万步数
        self.phases = [#平衡阶段重力小电机弱，移动阶段重力正常电机偏弱，高效阶段重力偏大电机全力，电机<1避免超限
            {"name": "balance", "steps": 0.3*total_steps, "params": {"gravity": -7.8, "motor_strength": 0.5}},
            {"name": "move", "steps": 0.6*total_steps, "params": {"gravity": -9.8, "motor_strength": 0.7}},
            {"name": "efficient", "steps": total_steps, "params": {"gravity": -11.8, "motor_strength": 1.0}}
        ]
        self.current_phase = 0
        self.current_step = 0

    def update(self, envs, global_step):
        
        self.current_step = global_step
        phase = self._get_current_phase()
        # 遍历所有并行环境
        for env in envs.envs:  # SyncVectorEnv的子环境列表
            mujoco_env = env.env  # 获取底层MuJoCo环境
            mujoco_env.model.opt.gravity[2] = phase["params"]["gravity"]#修改重力参数
            
            # 修改关节力矩
            for j in mujoco_env.model.actuator_actrange:
                j[:] = phase["params"]["motor_strength"]#修改电机属性
        
        

    def _get_current_phase(self):#根据步数获取所在阶段
        for i, phase in enumerate(self.phases):
            if self.current_step <= phase["steps"]:
                return phase
        return self.phases[-1]

class CurriculumRewardWrapper(Wrapper):#奖励包装器
    def __init__(self, env, curriculum):
        super().__init__(env)
        
        self.curriculum = curriculum
        self._rng = np.random.RandomState()  # 本地随机数生成器

    def seed(self, seed=None):#将环境包装后，第98行需要seed方法，所以另外加一个seed函数
        # 同时设置环境和包装器的随机种子
        seeds = []
        if hasattr(self.env, 'seed'):#检查环境是否有seed方法，有的话返回True,直接收集种子
            seeds += self.env.seed(seed)
        self._rng.seed(seed)#初始化包装器内部的随机数生成器，确保包装器层面的随机操作也确定可复现
        seeds.append(seed)
        return seeds

    
        
    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        
        # 根据阶段调整奖励权重
        phase = self.curriculum._get_current_phase()
        if phase["name"] == "balance":#在平衡阶段加入平衡奖励
            reward = 0.7 * self._balance_reward(obs) + 0.3 * reward
        elif phase["name"] == "move":#在移动阶段加入x方向前进速度奖励
            reward = 0.5 * reward + 0.5 * self._velocity_reward(obs)
        else:#在高效阶段加入能耗奖励
            reward = 0.3 * reward + 0.7 * self._energy_efficiency(obs, action)
            
        return obs, reward, done, trunc, info
    
    def _balance_reward(self, obs):
        # 躯干直立奖励（基于俯仰角）
        pitch = obs[4]  # 四元数中的x分量,无旋转时x=0,倒立时x=1
        return 1.0 - abs(pitch) 
    
    def _velocity_reward(self, obs):
        # 前进速度奖励
        return obs[0]  # x方向速度
        
    def _energy_efficiency(self, obs, action):
        # 能量效率奖励
        power = torch.sum(torch.abs(action * obs[15:23]))  # 通过动作*速度计算能耗
        return 1.0 / (power + 1e-6)

def make_env(env_id, seed, idx, capture_video, run_name):#创建单个环境
    def thunk():
        if capture_video and idx == 0:#仅对第一个环境录制视频，避免并行环境重复录制
            env = gym.make(env_id, render_mode="rgb_array")#指定rbg_array捕获帧画面
      
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")#视频按run_name归类存储
        else:
            env = gym.make(env_id)#创建原始环境
        env = gym.wrappers.RecordEpisodeStatistics(env)#通过wrapper为原始环境自动统计当前回合和累计奖励
        curriculum = AntCurriculum(total_steps=args.total_timesteps)#初始化课程实例
        env = CurriculumRewardWrapper(env, curriculum)#用奖励包装器包装环境
    
        env.seed(seed)#设置环境的随机规则
        env.observation_space.seed(seed)#按随机种子采样初始观测状态
        env.action_space.seed(seed) #按随机种子采样动作
        return env

    return thunk #延迟环境创建，避免在并行化使提前占用GPU资源



def create_envs(env_id, num_envs, seed, capture_video, run_name):#创建并行环境,SyncVectorEnv会管理thunk的调用
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed + i, i, capture_video, run_name) 
         for i in range(num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    return envs

if __name__ == "__main__":
    # 测试单个环境创建
    test_env = make_env("Ant-v5", seed=42, idx=0, capture_video=False, run_name="test1")()#相当于thunk()
    print("测试环境动作空间:", test_env.action_space)
    
    # 测试并行环境创建
    test_envs = create_envs(
        env_id="Ant-v5",
        num_envs=3,
        seed=42,
        capture_video=False,
        run_name="test2"
    )
    print("并行环境数量:", test_envs.num_envs)