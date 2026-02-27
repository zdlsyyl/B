from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# 控制频率缩放（给 low-freq 版本用）
LOW_FREQ_SCALE = 0.5


@configclass
class MyRobotFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for Tracking-Flat-MyRobot-v0.

    针对你的情况做了更“稳”的设置：
    - 33DOF（动作维度较大） -> 降低学习率、增加 batch 稳定性
    - Tracking 任务通常 reward 更复杂 -> entropy 略增，避免过早收敛到坏局部最优
    - 使用 empirical_normalization，通常对 imitation/tracking 更稳
    """

    # rollout 配置
    num_steps_per_env = 32            # 比 g1 的 24 稍长，提升稳定性
    max_iterations = 20000            # 先给一个中等训练长度（不够再加）
    save_interval = 200
    experiment_name = "myrobot_flat"
    empirical_normalization = True

    # policy/critic 网络
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # PPO 算法超参
    algorithm = RslRlPpoAlgorithmCfg(
        # value function
        value_loss_coef=1.0,
        use_clipped_value_loss=True,

        # PPO clipping
        clip_param=0.2,

        # exploration（tracking 常需要更久探索）
        entropy_coef=0.01,

        # 优化相关：动作维度大 -> 用更稳的 batch 配置
        num_learning_epochs=5,
        num_mini_batches=8,           # 比 g1(4) 更大：每次更新更平滑
        learning_rate=3.0e-4,         # 比 g1(1e-3) 更保守，降低发散风险
        schedule="adaptive",

        # GAE/discount
        gamma=0.99,
        lam=0.95,

        # KL 控制（adaptive schedule 依赖）
        desired_kl=0.01,

        # 梯度裁剪
        max_grad_norm=1.0,
    )


@configclass
class MyRobotFlatLowFreqPPORunnerCfg(MyRobotFlatPPORunnerCfg):
    """用于低频控制版本（如果你注册了 Low-Freq 环境并希望切专用 PPO cfg）。"""

    def __post_init__(self):
        super().__post_init__()
        # 低频：每 step 的时间更长 -> 需要调整 gamma/lam 才等效
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)
