from isaaclab.utils import configclass

from whole_body_tracking.robots.urdf7 import MY_ROBOT_ACTION_SCALE, MY_ROBOT_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg

# 如果你也想要 low-freq 版本，沿用 g1 的 LOW_FREQ_SCALE 写法：
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE


@configclass
class MyRobotFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 绑定机器人资产
        self.scene.robot = MY_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # joint position action 的缩放（每关节一份）
        self.actions.joint_pos.scale = MY_ROBOT_ACTION_SCALE
        

        # 你的 URDF root link 是 Trunk
        self.commands.motion.anchor_body_name = "Trunk"
        self.events.base_com.params["asset_cfg"].body_names = "Trunk"

        # 选一组“比较稳”的 tracking 点：
        # - 躯干 + 腰部 + 双足末端 + 双臂末端 + 手末端（H2）
        # 你后续可以按需求增减（越多越难学、但效果更像“全身跟踪”）
        self.commands.motion.body_names = [
            "Trunk",
            "Waist1",
            "Waist2",
            "Waist3",
            "Right_Hip_Roll",
            "Right_shank_Pitch",
            "Right_Ankle_Pitch",
            "Right_toe_pitch",
            "Left_Hip_Roll",
            "Left_shank_Pitch",
            "Left_Ankle_Pitch",
            "left_toe_pitch",
            "AR6",
            "AL6",
            "H2",
        ]


@configclass
class MyRobotFlatWoStateEstimationEnvCfg(MyRobotFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 和你工程 g1 一样：去掉部分观测用于对比实验
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class MyRobotFlatLowFreqEnvCfg(MyRobotFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 降低控制频率：增大 step 间隔（减少 decimation）
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
