import gymnasium as gym

def make_env(env_id, seed, idx, capture_video, run_name):#创建单个环境
    def thunk():
        if capture_video and idx == 0:#仅对第一个环境录制视频，避免并行环境重复录制
            env = gym.make(env_id, render_mode="rgb_array")#指定rbg_array捕获帧画面
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")#视频按run_name归类存储
        else:
            env = gym.make(env_id)#创建原始环境
        env = gym.wrappers.RecordEpisodeStatistics(env)#通过wrapper为原始环境自动统计当前回合和累计奖励
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