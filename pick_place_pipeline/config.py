"""
机器人抓取系统配置文件
所有可配置的参数按功能分类定义
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from galbot_sdk.g1 import SensorType


@dataclass(frozen=True)
class FoundationPoseObjectConfig:
    """单个 FoundationPose 物体的 CAD 和抓取偏移配置。"""

    mesh_path: str
    grasp_point_object: Tuple[float, float, float]
    pre_grasp_distance: float
    gripper_ee_distance: float


@dataclass
class RobotConfig:
    """机器人配置（机械臂、夹爪、运动参数等）。"""

    arm_name: str = "right_arm"  # 机械臂名称
    gripper_name: str = "right_gripper"  # 夹爪名称
    debug_mode: bool = True  # 调试模式（启用交互确认、详细可视化等）

    # 夹爪参数
    # 是否根据识别到的物体宽度自动计算关闭宽度
    auto_gripper_close_width: bool = False
    gripper_open_width: float = 0.10  # 打开宽度（米）
    gripper_close_width: float = 0.04  # 关闭宽度（米）
    gripper_width_margin: float = 0.02  # 自动关闭时预留的夹紧量（米）
    gripper_effort: float = 25  # 夹爪力矩
    # 抓取闭合使用非阻塞命令。夹持力连续达标且开口宽度稳定后才允许抬升；
    # 不以 SDK 的 is_moving 为准，兼容夹住物体后仍报告运动的情形。
    gripper_grasp_effort_threshold: float = 8.0
    gripper_grasp_effort_stable_samples: int = 3
    gripper_grasp_width_stable_tolerance_m: float = 0.003
    gripper_grasp_timeout_s: float = 3.0
    gripper_state_poll_interval_s: float = 0.1

    # FoundationPose 抓取规划参数
    # 物体 Z 轴（对称轴）与竖直方向的夹角超过该阈值时，判定为斜躺/平躺，
    # 改用竖直向下 + yaw 对齐抓取，而非假设物体竖立的侧面接近抓取。
    foundationpose_lying_angle_threshold_deg: float = 45.0
    lift_height: float = 0.12  # 抓取后抬起高度（米）
    work_angle_min_deg: float = -10.0  # 右臂接近方向最小角度
    work_angle_max_deg: float = 10.0  # 右臂接近方向最大角度

    # 运动参数
    move_timeout: float = 8.0  # 位姿运动超时（秒）
    joint_speed: float = 0.2  # 关节运动速度
    gripper_speed: float = 0.1  # 夹爪运动速度（米/秒）

    # 任务中用于收臂和导航前回退的默认关节姿态。
    # 未配置时，主流程记录启动时当前关节姿态作为兼容回退。
    home_joint_positions: Optional[List[float]] = None


@dataclass
class TaskConfig:
    """任务编排的重试、结束条件与安全预算。"""

    perception_retries: int = 3
    perception_retry_interval_s: float = 0.5
    empty_confirmations: int = 3
    max_cycles: int = 20


@dataclass
class PerceptionConfig:
    """视觉感知配置。"""

    # 相机固定为头部左目 RGB（HEAD_LEFT_CAMERA），深度由 FOUNDATION_STEREO
    # 双目推理生成（头部没有独立深度传感器）。
    head_model_ready_wait_s: float = 12.0  # 头部双目深度模型加载等待（秒）
    head_inference_timeout_s: float = 6.0  # 单次双目深度推理超时（秒）

    # FoundationPose 用分割模型生成种子 mask。该模型只负责分割，不参与
    # 抓取方向规划；抓取位姿始终来自 FoundationPose 的 6DoF 结果。
    mask_conf_threshold: float = 0.40
    mask_weights: str = "assets/models/yolo26s-seg_finetuned.pt"

    # 当前 yolo26s 微调权重把类别名保存为英文描述；主流程用短类别名作为
    # FoundationPose CAD 配置 key，因此这里做一次归一化映射。
    yolo_class_aliases: Dict[str, str] = field(
        default_factory=lambda: {
            "a tall green cylindrical can of Lay's Stax potato chips": "leshishupian",
            "a red cup of spicy instant noodles": "xinlamian",
            "a beige rectangular soft pack of facial tissues": "chouzhi",
            "a brown plastic bottle of oolong tea with a white cap": "wulongcha",
            "a red plastic bottle of Coca-Cola with a red cap": "kekoukele",
        }
    )

    # 可抓物料白名单，同时保存每个类别的 CAD 和抓取偏移。
    foundationpose_objects: Dict[str, FoundationPoseObjectConfig] = field(
        default_factory=lambda: {
            "wulongcha": FoundationPoseObjectConfig(
                mesh_path="../assets/objects/wulongcha/normalized.obj",
                grasp_point_object=(0.0, 0.0, 0.1062),
                pre_grasp_distance=0.08,
                gripper_ee_distance=0.18,
            ),
            "chouzhi": FoundationPoseObjectConfig(
                mesh_path="../assets/objects/chouzhi/normalized.obj",
                grasp_point_object=(0.0822, 0.0747, 0.0789),
                pre_grasp_distance=0.08,
                gripper_ee_distance=0.18,
            ),
            "leshishupian": FoundationPoseObjectConfig(
                mesh_path="../assets/objects/leshishupian/normalized.obj",
                grasp_point_object=(0.0, 0.0, 0.105),
                pre_grasp_distance=0.08,
                gripper_ee_distance=0.18,
            ),
            "xinlamian": FoundationPoseObjectConfig(
                mesh_path="../assets/objects/xinlamian/normalized.obj",
                grasp_point_object=(0.0, 0.0, 0.055),
                pre_grasp_distance=0.08,
                gripper_ee_distance=0.18,
            ),
            "kekoukele": FoundationPoseObjectConfig(
                mesh_path="../assets/objects/kekoukele/normalized.obj",
                grasp_point_object=(0.0, 0.0, 0.0812),
                pre_grasp_distance=0.08,
                gripper_ee_distance=0.18,
            ),
        }
    )

    # 本地 FoundationPose 配置
    fp_root: str = "/home/galbot/FoundationPose"  # FoundationPose 仓库根目录
    # 默认以头部相机原始分辨率运行；显存受限时可设为小于 1 的比例。
    fp_input_scale: float = 1.0

    # 深度图处理
    depth_median_radius: int = 5  # 深度中位数半径（像素）

    def validate_foundationpose_candidates(self) -> None:
        """校验全部 FoundationPose 候选的 CAD 和抓取偏移。"""

        if (
            not math.isfinite(self.head_model_ready_wait_s)
            or self.head_model_ready_wait_s < 0
        ):
            raise ValueError("head_model_ready_wait_s 必须是非负有限数值")
        if (
            not math.isfinite(self.head_inference_timeout_s)
            or self.head_inference_timeout_s <= 0
        ):
            raise ValueError("head_inference_timeout_s 必须是正的有限数值")
        if not self.foundationpose_objects:
            raise ValueError("foundationpose_objects 至少需要配置一种可抓物料")
        if not self.mask_weights:
            raise ValueError("mask_weights 不能为空")
        if any(not name or not alias for name, alias in self.yolo_class_aliases.items()):
            raise ValueError("yolo_class_aliases 的 key 和 value 都不能为空")
        if (
            not math.isfinite(self.mask_conf_threshold)
            or not 0.0 < self.mask_conf_threshold <= 1.0
        ):
            raise ValueError("mask_conf_threshold 必须在 (0, 1] 范围内")
        if not self.fp_root:
            raise ValueError("fp_root 不能为空")
        if (
            not math.isfinite(self.fp_input_scale)
            or not 0.0 < self.fp_input_scale <= 1.0
        ):
            raise ValueError("fp_input_scale 必须在 (0, 1] 范围内")
        if self.depth_median_radius < 0:
            raise ValueError("depth_median_radius 必须为非负整数")
        for class_name, object_config in self.foundationpose_objects.items():
            if not class_name or class_name.startswith("<trained_class_"):
                raise ValueError(
                    "foundationpose_objects 必须使用训练模型中的实际类别名"
                )
            if class_name in self.yolo_class_aliases and self.yolo_class_aliases[class_name] != class_name:
                raise ValueError(
                    f"yolo_class_aliases 不能把配置类别 {class_name!r} 映射到其他类别"
                )
            if not object_config.mesh_path:
                raise ValueError(
                    f"物料 {class_name} 的 FoundationPose mesh_path 不能为空"
                )
            if len(object_config.grasp_point_object) != 3 or not all(
                math.isfinite(value) for value in object_config.grasp_point_object
            ):
                raise ValueError(
                    f"物料 {class_name} 的 grasp_point_object 必须是三个有限数值"
                )
            for name in ("pre_grasp_distance", "gripper_ee_distance"):
                value = getattr(object_config, name)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(
                        f"物料 {class_name} 的 {name} 必须是正的有限数值"
                    )


@dataclass
class NavigationConfig:
    """导航配置。"""

    enable_navigation: bool = False  # 是否启用导航

    # 任务正常结束时返回的原点位姿；循环启动时不会先导航到这里
    origin_pose: Optional[List[float]] = None  # [x, y, z, qx, qy, qz, qw]

    # 抓取点导航位姿 [x, y, z, qx, qy, qz, qw]（启用导航时必须指定）
    pickup_pose: Optional[List[float]] = None

    # 放置点导航位姿 [x, y, z, qx, qy, qz, qw]（启用导航时必须指定）
    place_pose: Optional[List[float]] = None

    # 导航时保持当前朝向（不转向）：目标朝向强制为当前朝向，
    # 并把导航 yaw 速度上限压到 SDK 最小值，适用于各路点朝向一致的场景
    lock_heading: bool = True

    def validate(self) -> None:
        """验证导航配置是否完整"""
        if self.enable_navigation:
            if self.origin_pose is None:
                raise ValueError("启用导航时必须指定 origin_pose")
            if self.pickup_pose is None:
                raise ValueError("启用导航时必须指定 pickup_pose")
            if self.place_pose is None:
                raise ValueError("启用导航时必须指定 place_pose")
            if len(self.origin_pose) != 7:
                raise ValueError(
                    "origin_pose 格式必须为 [x, y, z, qx, qy, qz, qw]"
                )
            if len(self.pickup_pose) != 7:
                raise ValueError(
                    "pickup_pose 格式必须为 [x, y, z, qx, qy, qz, qw]"
                )
            if len(self.place_pose) != 7:
                raise ValueError(
                    "place_pose 格式必须为 [x, y, z, qx, qy, qz, qw]"
                )


@dataclass
class LoggingConfig:
    """日志配置。"""

    log_to_console: bool = True  # 是否输出到控制台
    log_to_file: bool = True  # 是否保存到文件
    log_level: str = "INFO"  # 日志级别
    log_dir: str = "logs"  # 日志文件目录


@dataclass
class PipelineConfig:
    """抓取流程配置，整合所有子配置。"""

    robot: RobotConfig = field(default_factory=RobotConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Robot配置快捷访问
    @property
    def arm_name(self) -> str:
        return self.robot.arm_name

    @arm_name.setter
    def arm_name(self, value: str) -> None:
        self.robot.arm_name = value

    @property
    def gripper_name(self) -> str:
        return self.robot.gripper_name

    @property
    def debug_mode(self) -> bool:
        return self.robot.debug_mode

    @property
    def gripper_open_width(self) -> float:
        return self.robot.gripper_open_width

    @property
    def gripper_close_width(self) -> float:
        return self.robot.gripper_close_width

    @property
    def auto_gripper_close_width(self) -> bool:
        return self.robot.auto_gripper_close_width

    @property
    def gripper_width_margin(self) -> float:
        return self.robot.gripper_width_margin

    @property
    def gripper_effort(self) -> float:
        return self.robot.gripper_effort

    @property
    def gripper_grasp_effort_threshold(self) -> float:
        return self.robot.gripper_grasp_effort_threshold

    @property
    def gripper_grasp_effort_stable_samples(self) -> int:
        return self.robot.gripper_grasp_effort_stable_samples

    @property
    def gripper_grasp_width_stable_tolerance_m(self) -> float:
        return self.robot.gripper_grasp_width_stable_tolerance_m

    @property
    def gripper_grasp_timeout_s(self) -> float:
        return self.robot.gripper_grasp_timeout_s

    @property
    def gripper_state_poll_interval_s(self) -> float:
        return self.robot.gripper_state_poll_interval_s

    @property
    def foundationpose_lying_angle_threshold_deg(self) -> float:
        return self.robot.foundationpose_lying_angle_threshold_deg

    @property
    def lift_height(self) -> float:
        return self.robot.lift_height

    @property
    def work_angle_min_deg(self) -> float:
        return self.robot.work_angle_min_deg

    @property
    def work_angle_max_deg(self) -> float:
        return self.robot.work_angle_max_deg

    @property
    def move_timeout(self) -> float:
        return self.robot.move_timeout

    @property
    def joint_speed(self) -> float:
        return self.robot.joint_speed

    @property
    def gripper_speed(self) -> float:
        return self.robot.gripper_speed

    @property
    def home_joint_positions(self) -> Optional[List[float]]:
        return self.robot.home_joint_positions

    # Perception配置快捷访问
    @property
    def enabled_sensors(self) -> tuple:
        # 头部深度来自 FOUNDATION_STEREO 双目推理，robot.init 只需启用左目 RGB
        return (SensorType.HEAD_LEFT_CAMERA,)

    @property
    def depth_median_radius(self) -> int:
        return self.perception.depth_median_radius

    # Navigation配置快捷访问
    @property
    def enable_navigation(self) -> bool:
        return self.navigation.enable_navigation

    @property
    def origin_pose(self) -> Optional[List[float]]:
        return self.navigation.origin_pose

    @property
    def pickup_pose(self) -> Optional[List[float]]:
        return self.navigation.pickup_pose

    @property
    def place_pose(self) -> Optional[List[float]]:
        return self.navigation.place_pose

    # Logging配置快捷访问
    @property
    def log_to_console(self) -> bool:
        return self.logging.log_to_console

    @property
    def log_to_file(self) -> bool:
        return self.logging.log_to_file

    @property
    def log_level(self) -> str:
        return self.logging.log_level

    @property
    def log_dir(self) -> str:
        return self.logging.log_dir

    # Interaction配置快捷访问
    @property
    def safety_check(self) -> bool:
        return True

    def validate_task(self, *, enable_navigation: bool) -> None:
        """验证公共抓取任务以及所选入口需要的配置。"""

        expected_grippers = {
            "left_arm": "left_gripper",
            "right_arm": "right_gripper",
        }
        expected_gripper = expected_grippers.get(self.arm_name)
        if expected_gripper is None:
            raise ValueError("arm_name 仅支持 'left_arm' 或 'right_arm'")
        if self.gripper_name != expected_gripper:
            raise ValueError(
                f"{self.arm_name} 必须使用 {expected_gripper}，"
                f"当前为 {self.gripper_name}"
            )
        for name in (
            "gripper_grasp_effort_threshold",
            "gripper_grasp_width_stable_tolerance_m",
            "gripper_grasp_timeout_s",
            "gripper_state_poll_interval_s",
        ):
            value = getattr(self.robot, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须是正的有限数值")
        if self.robot.gripper_grasp_effort_stable_samples < 1:
            raise ValueError(
                "gripper_grasp_effort_stable_samples 必须大于等于 1"
            )
        if (
            self.robot.gripper_state_poll_interval_s
            >= self.robot.gripper_grasp_timeout_s
        ):
            raise ValueError(
                "gripper_state_poll_interval_s 必须小于 "
                "gripper_grasp_timeout_s"
            )
        self.perception.validate_foundationpose_candidates()
        threshold = self.robot.foundationpose_lying_angle_threshold_deg
        if not math.isfinite(threshold) or not (0.0 < threshold < 90.0):
            raise ValueError(
                "foundationpose_lying_angle_threshold_deg 必须在 (0, 90) 度之间"
            )
        if enable_navigation:
            if not self.enable_navigation:
                raise ValueError("循环抓取任务必须启用导航")
            self.navigation.validate()
        elif self.enable_navigation:
            raise ValueError("单物体流程不能同时启用导航")
        if (
            self.home_joint_positions is not None
            and not self.home_joint_positions
        ):
            raise ValueError(
                "home_joint_positions 不能为空；未标定时请设为 None"
            )
        if self.task.perception_retries < 1:
            raise ValueError("perception_retries 必须大于等于 1")
        if (
            not math.isfinite(self.task.perception_retry_interval_s)
            or self.task.perception_retry_interval_s < 0
        ):
            raise ValueError("perception_retry_interval_s 必须是非负有限数值")
        if self.task.empty_confirmations < 1:
            raise ValueError("empty_confirmations 必须大于等于 1")
        if self.task.max_cycles < 1:
            raise ValueError("max_cycles 必须大于等于 1")

    def validate_material_replenishment(self) -> None:
        """兼容旧调用；新入口应显式调用 :meth:`validate_task`。"""

        self.validate_task(enable_navigation=self.navigation.enable_navigation)


# 默认配置实例
cfg = PipelineConfig()
"""全局默认配置实例。如需自定义配置，直接修改 cfg 的属性即可：

例:
    cfg.robot.arm_name = "left_arm"
    cfg.navigation.enable_navigation = True
    cfg.navigation.origin_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    cfg.navigation.pickup_pose = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
"""
