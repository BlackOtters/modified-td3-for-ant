# test.py
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import time

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.register_buffer('action_scale', torch.tensor(1.0))
        self.register_buffer('action_bias', torch.tensor(0.0))
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc_mu(x)) * self.action_scale + self.action_bias

def test_td3(model_path, env_id="Ant-v4", episodes=5, render=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建环境
    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 加载模型
    actor = Actor(obs_dim, action_dim).to(device)
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, tuple):
        state_dict = state_dict[0]
    actor.load_state_dict(state_dict, strict=False)
    actor.eval()
    
    print(f"开始测试 {env_id} | 渲染: {render}")
    
    rewards, steps_list, fps_list = [], [], []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        episode_reward, steps = 0, 0
        start_time = time.time()
        
        while True:
            with torch.no_grad():
                action = actor(torch.tensor(obs, dtype=torch.float32).to(device))
                action = action.cpu().numpy().clip(-1, 1)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1
            
            if render:
                env.render()
            
            if terminated or truncated:
                fps = steps / (time.time() - start_time)
                rewards.append(episode_reward)
                steps_list.append(steps)
                fps_list.append(fps)
                
                print(f"回合 {episode+1}: 奖励={episode_reward:8.2f}, "
                      f"步数={steps:4d}, FPS={fps:5.1f}")
                break
    
    env.close()
    
    # 监控报告
    if episodes > 0:
        print(f"\n{'='*50}")
        print(f"平均奖励: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"平均步数: {np.mean(steps_list):.1f} ± {np.std(steps_list):.1f}")
        print(f"平均FPS:  {np.mean(fps_list):.1f} ± {np.std(fps_list):.1f}")
        print(f"{'='*50}")

if __name__ == "__main__":
    test_td3("runs/your model path", 
             episodes=3, render=True)#for example runs/Ant-v4__arg_def__1__1761308129/arg_def.cleanrl_model