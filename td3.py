import random
import time
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import tyro
from cleanrl_utils.buffers import ReplayBuffer

# 从各模块文件导入所需组件
from arg_def import Args
from env import make_env, create_envs
from net import Actor, QNetwork, create_networks

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
    
    rb = ReplayBuffer(
        buffer_size=args.buffer_size,
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        device=device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,#超时是否设为终止
    )#创建经验回放缓冲区
    start_time = time.time()#记录训练开始时刻

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)#环境重置初始化
    for global_step in range(args.total_timesteps):#进入训练循环
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
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs#更新状态

        # ALGO LOGIC: training.#进入网络更新
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)#经验回放
            with torch.no_grad():
                clipped_noise = (torch.randn_like(data.actions, device=device) * args.policy_noise).clamp(
                    -args.noise_clip, args.noise_clip
                ) * target_actor.action_scale#计算在目标价值网络计算Q值时给动作添加的截断噪声范围

                next_state_actions = (target_actor(data.next_observations) + clipped_noise).clamp(
                    envs.single_action_space.low[0], envs.single_action_space.high[0]
                )#计算得到Q值所需的下一时刻动作，加入policy_noise
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)#取较小Q值
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)#计算目标函数，终止状态done除外，不再计算未来回报

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss#两个价值网络共用一个优化器，所以将两个损失函数合并来优化

            # optimize the model
            q_optimizer.zero_grad()#清除历史梯度
            qf_loss.backward()#计算参数梯度
            q_optimizer.step()#执行梯度更新

            if global_step % args.policy_frequency == 0:# 延迟策略更新
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()#最小化负Q值，虽然取qf1但理论上应该取较小值，取整个批次数据的均值来计算损失
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
        from cleanrl_utils.evals import evaluate_policy

        episodic_returns = evaluate_policy(#执行模型评估
            model_path, #加载模型
            make_env, #创建评估环境
            args.env_id,
            eval_episodes=10,#评估回合数
            run_name=f"{run_name}-eval",
            Model=(Actor, QNetwork),
            device=device,
            exploration_noise=args.exploration_noise,#加入探索噪声
        )
        for idx, episodic_return in enumerate(episodic_returns):#记录评估结果，可以在tensorboard的TIME SERIES查看曲线
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

    envs.close()
    writer.close()