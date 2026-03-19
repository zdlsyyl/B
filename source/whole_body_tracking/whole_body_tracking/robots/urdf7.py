import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

# -----------------------------------------------------------------------------
# Robot asset
# -----------------------------------------------------------------------------
MY_ROBOT_USD_PATH = f"/home/winer/urdf7_SLDASM/urdf/urdf7_SLDASM/urdf7_SLDASM.usd"

# NOTE:
# - 你的 URDF 里有若干 joint 名字包含空格（例如 "Right _Shoulde_Yaw" / "Right_Hip_ Yaw" 等）
# - 为避免 regex 表达式/匹配问题，这里用“显式 joint 名称列表”配置 actuator
MY_ROBOT_JOINT_NAMES = ['Right_Shoulder_Roll', 'Right_Shoulder_Pitch', 'Right_Shoulder_Yaw', 'Right_Elbow_Pitch', 'Right_Elbow_Yaw', 'Right_Elbow_Pitch_1', 'Left_Shoulder_Roll', 'Left_Shoulder_Pitch', 'Left_Shoulder_Yaw', 'Left_Elbow_Pitch', 'Left_Elbow_Yaw', 'Left_Elbow_Pitch_1', 'Hand_yaw', 'Hand_pitch', 'Waist_Roll', 'Waist_Pitch', 'Waist_Yaw', 'Right_Hip_Roll_1', 'Right_Hip_Pitch_2', 'Right_Hip_Yaw_3', 'Right_shank_Pitch_2', 'Right_Ankle_Roll_1', 'Right_Ankle_Pitch_2', 'Right_Ankle_Yaw', 'Right_toe_pitch_1', 'Left_Hip_Roll_1', 'Left_Hip_Pitch_2', 'Left_Hip_Yaw_2', 'Left_shank_Pitch_2', 'Left_Ankle_Roll_1', 'Left_Ankle_Pitch_2', 'Left_Ankle_Yaw_3', 'left_toe_pitch_1']
# -----------------------------------------------------------------------------
# Actuator tuning (给一个“能先跑起来”的保守默认值)
# -----------------------------------------------------------------------------
# 你的 URDF 里 limit/effort/velocity 基本都是 0，这会导致“关节不可动/不可控”。
# 所以我们在 actuator 侧给一套默认 effort/velocity/stiffness/damping。
DEFAULT_EFFORT_LIMIT = 60.0      # N*m（先保守一点）
DEFAULT_VELOCITY_LIMIT = 20.0    # rad/s
DEFAULT_STIFFNESS = 80.0         # N*m/rad（隐式PD）
DEFAULT_DAMPING = 6.0            # N*m*s/rad
DEFAULT_ARMATURE = 0.01          # kg*m^2（用于数值稳定/近似转子惯量）

MY_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        
        usd_path=MY_ROBOT_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
        
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # 先按“站立/默认姿态”给一个安全高度
        pos=(0.0, 0.0, 0.8),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={name: 0.0 for name in MY_ROBOT_JOINT_NAMES},
        joint_vel={name: 0.0 for name in MY_ROBOT_JOINT_NAMES},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*Hip.*", ".*shank.*", ".*Ankle.*"], # 匹配大腿、小腿、脚踝
            effort_limit_sim=80.0,
            velocity_limit_sim=20.0,
            stiffness=80.0,  # 下肢需要强有力的支撑
            damping=5.0,
            armature=DEFAULT_ARMATURE,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*Shoulder.*", ".*Elbow.*"], # 匹配手臂
            effort_limit_sim=40.0,
            velocity_limit_sim=20.0,
            stiffness=40.0,  # 手臂刚度减半
            damping=3.0,
            armature=DEFAULT_ARMATURE,
        ),
        "extremities": ImplicitActuatorCfg(
            joint_names_expr=["Hand.*", ".*toe.*", "Waist.*"], # 手腕、脚趾、腰部等轻量级关节
            effort_limit_sim=20.0,
            velocity_limit_sim=20.0,
            stiffness=20.0,  # 大幅降低刚度，消除高频抖动
            damping=2.0,
            armature=DEFAULT_ARMATURE,
        ),
    },
)

# -----------------------------------------------------------------------------
# Action scale
# IsaacLab 里 joint_pos action 通常会乘以 scale（每个关节一个值也行）
# 这里按你工程 g1.py 的做法： scale ~= 0.25 * effort / stiffness
# -----------------------------------------------------------------------------
MY_ROBOT_ACTION_SCALE = {}
for a in MY_ROBOT_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            MY_ROBOT_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
