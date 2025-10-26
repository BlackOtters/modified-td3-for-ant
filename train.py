import random
import time
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import tyro
from torch.utils.data import Dataset
from collections import deque

# 从各模块文件导入所需组件
from arg_def import Args
from make_env import create_envs
from net import create_networks
from make_env import AntCurriculum

class EnhancedReplayBuffer(Dataset):
    def __init__(self, buffer_size, observation_space, action_space, device, 
                 prioritized=True, n_step=1, alpha=0.6, beta=0.4):
        self.buffer_size = buffer_size
        self.obs_shape = observation_space.shape
        self.act_shape = action_space.shape
        self.device = device
        self.prioritized = prioritized
        self.n_step = n_step
        self.alpha = alpha
        self.beta = beta
        
        # 核心存储
        self.observations = torch.zeros((buffer_size, *self.obs_shape), 
                                      dtype=torch.float32, device=device)
        self.next_observations = torch.zeros_like(self.observations)
        self.actions = torch.zeros((buffer_size, *self.act_shape), 
                                 device=device)
        self.rewards = torch.zeros(buffer_size, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool, device=device)
        
        # 优先级采样
        self.priorities = torch.ones(buffer_size, device=device) * 1e-6
        self.max_priority = 1.0
        
        # 轨迹管理
        self.trajectories = deque(maxlen=1000)
        self.current_traj = {
            'obs': [], 'actions': [], 'rewards': [], 'dones': []
        }
        
        self.pos = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, obs, next_obs, action, reward, done, info=None):
        idx = self.pos % self.buffer_size
        
        # 转换为Tensor
        self.observations[idx] = torch.as_tensor(obs, device=self.device)
        self.next_observations[idx] = torch.as_tensor(next_obs, device=self.device)
        self.actions[idx] = torch.as_tensor(action, device=self.device)
        self.rewards[idx] = torch.as_tensor(reward, device=self.device)
        self.dones[idx] = torch.as_tensor(done, device=self.device)
        
        # 初始优先级
        if self.prioritized:
            self.priorities[idx] = self.max_priority
            
        # 轨迹管理
        self._update_trajectory(obs, action, reward, done, info)
        
        self.pos += 1
        self.size = min(self.size + 1, self.buffer_size)

    def _update_trajectory(self, obs, action, reward, done, info):
        """管理完整轨迹用于后续分析"""
        self.current_traj['obs'].append(obs)
        self.current_traj['actions'].append(action)
        self.current_traj['rewards'].append(reward)
        self.current_traj['dones'].append(done)
        
        if done or len(self.current_traj['obs']) >= 1000:  # 防止内存泄漏
            self.trajectories.append(self.current_traj)
            self.current_traj = {'obs': [], 'actions': [], 'rewards': [], 'dones': []}

    def sample(self, batch_size, beta=None):
        if self.prioritized:
            return self._priority_sample(batch_size, beta or self.beta)
        return self._random_sample(batch_size)

    def _priority_sample(self, batch_size, beta):
        # 计算采样概率
        probs = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()
        
        # 重要性采样
        indices = torch.multinomial(probs, batch_size, replacement=True)
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        batch = self._get_batch(indices)
        return {**batch, 'indices': indices, 'weights': weights}

    def _random_sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,))
        return self._get_batch(indices)

    def _get_batch(self, indices):
        # 基础batch
        batch = {
            'observations': self.observations[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_observations': self.next_observations[indices],
            'dones': self.dones[indices]
        }
        
        # 计算n-step回报
        if self.n_step > 1:
            batch = self._compute_n_step_returns(batch, indices)
            
        return batch

    def _compute_n_step_returns(self, batch, indices):
        """计算n-step折扣回报"""
        n_step_rewards = torch.zeros_like(batch['rewards'])
        n_step_dones = torch.zeros_like(batch['dones'])
        next_obs = torch.zeros_like(batch['next_observations'])
        
        for i, idx in enumerate(indices):
            end_idx = min(idx + self.n_step, self.size - 1)
            gamma = 0.99
            
            # 计算折扣回报
            reward = 0
            for j in range(idx, end_idx):
                reward += (gamma ** (j - idx)) * self.rewards[j]
                if self.dones[j]:
                    break
                    
            n_step_rewards[i] = reward
            n_step_dones[i] = self.dones[min(idx + self.n_step - 1, self.size - 1)]
            next_obs[i] = self.next_observations[min(idx + self.n_step, self.size - 1)]
            
        return {
            **batch,
            'rewards': n_step_rewards,
            'next_observations': next_obs,
            'dones': n_step_dones
        }

    def update_priorities(self, indices, priorities):
        """更新优先级"""
        if self.prioritized:
            priorities = priorities.squeeze() + 1e-6  # 防止归零
            self.priorities[indices] = priorities
            self.max_priority = max(self.max_priority, priorities.max())



if __name__ == "__main__":
    # 参数解析，使用tyro传参，所有超参数封装到args
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    # 初始化日志
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )#将超参数以Markdown表格形式保存，便于后续分析，可以在tensorboard的TEXT查看

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # 环境初始化
    envs = create_envs(
        env_id=args.env_id,
        num_envs=args.num_envs,
        seed=args.seed,
        capture_video=args.capture_video,
        run_name=run_name
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # 网络初始化
    networks = create_networks(
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        device=device
    )
    actor = networks["actor"]
    qf1 = networks["qf1"]
    qf2 = networks["qf2"]
    qf1_target = networks["qf1_target"]
    qf2_target = networks["qf2_target"]
    target_actor = networks["target_actor"]
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)#合并两个价值网络的优化器，减少内存占用
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)#同样进行优化器配置

    envs.single_observation_space.dtype = np.float32#所有观测值转换为float32
    
    buffer = EnhancedReplayBuffer(
        buffer_size=args.buffer_size,
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        device=device,
        prioritized=True,
        n_step=3
    )#创建经验回放缓冲区

    # 在训练循环开始前初始化
    curriculum = AntCurriculum(args.total_timesteps)

    start_time = time.time()#记录训练开始时刻

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)#环境重置初始化
    for global_step in range(args.total_timesteps):#进入训练循环
        curriculum.update(envs, global_step)
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:#动作选取：正式训练开始前进行随机探索
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():#禁止梯度跟踪
                actions = actor(torch.Tensor(obs).to(device))#从策略网络获得动作
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)#添加探索噪声
                #截断超出正常范围的动作，断开计算图并转为numpy
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)
     

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)#执行动作并记录数据

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:#在诊断为回合结束时打印和记录信息
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):#进行超时情况的特殊处理，让info不要再记录next_obs
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]#用final_obs替换next_obs，以免影响TD目标计算
        buffer.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs#更新状态

        # ALGO LOGIC: training.#进入网络更新
        if global_step > args.learning_starts:
            beta = min(1.0, args.beta_init + global_step/args.total_timesteps)
            data = buffer.sample(args.batch_size, beta=beta)#经验回放
            with torch.no_grad():
                clipped_noise = (torch.randn_like(data["actions"], device=device) * args.policy_noise).clamp(
                    -args.noise_clip, args.noise_clip
                ) * target_actor.action_scale#计算在目标价值网络计算Q值时给动作添加的截断噪声范围

                next_state_actions = (target_actor(data["next_observations"]) + clipped_noise).clamp(
                    envs.single_action_space.low[0], envs.single_action_space.high[0]
                )#计算得到Q值所需的下一时刻动作，加入policy_noise
                qf1_next_target = qf1_target(data["next_observations"], next_state_actions)
                qf2_next_target = qf2_target(data["next_observations"], next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)#取较小Q值
                next_q_value = data["rewards"].flatten() + (~ data["dones"].flatten()) * args.gamma * (min_qf_next_target).view(-1)#计算目标函数，终止状态done除外，不再计算未来回报

            qf1_a_values = qf1(data["observations"], data["actions"]).view(-1)
            qf2_a_values = qf2(data["observations"], data["actions"]).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss#两个价值网络共用一个优化器，所以将两个损失函数合并来优化

            # optimize the model
            q_optimizer.zero_grad()#清除历史梯度
            qf_loss.backward()#计算参数梯度
            q_optimizer.step()#执行梯度更新

            if global_step % args.policy_frequency == 0:# 延迟策略更新
                actor_loss = -qf1(data["observations"], actor(data["observations"])).mean()#最小化负Q值，虽然取qf1但理论上应该取较小值，取整个批次数据的均值来计算损失
                actor_optimizer.zero_grad()#求解策略网络的优化器，三步走
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network#用软更新系数更新三个目标网络参数
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:#定期记录训练数据，可以在tensorboard的TIME SERIES查看曲线
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))#训练阶段实时输出训练速度
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

    if args.save_model:#完成训练循环，保存模型
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"#设置模型保存路径
        torch.save((actor.state_dict(), qf1.state_dict(), qf2.state_dict()), model_path)#保存神经网络参数
        print(f"model saved to {model_path}")


    envs.close()
    writer.close()