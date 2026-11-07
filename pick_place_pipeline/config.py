"""视觉感知与日志配置，可在没有机器人 SDK 的环境中读取。"""

import math
from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class FoundationPoseObjectConfig:
    """单个 FoundationPose 物体的 CAD 配置。"""

    mesh_path: str


@dataclass
class PerceptionConfig:
    """视觉感知配置。"""

    # 相机固定为头部左目 RGB（HEAD_LEFT_CAMERA），深度由 FOUNDATION_STEREO
    # 双目推理生成（头部没有独立深度传感器）。
    head_model_ready_wait_s: float = 12.0  # 头部双目深度模型加载等待（秒）
    head_inference_timeout_s: float = 6.0  # 单次双目深度推理超时（秒）

    # FoundationPose 用分割模型生成种子 mask。该模型只负责分割，不参与
    # 动作规划；物体位姿来自 FoundationPose 的 6DoF 结果。
    mask_conf_threshold: float = 0.40
    mask_weights: str = "assets/models/yolo11n-seg_finetuned.pt"

    # 兼容将类别名保存为英文描述的微调权重；主流程用短类别名作为
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

    # 识别物料白名单，保存每个类别的 CAD 模型。
    foundationpose_objects: Dict[str, FoundationPoseObjectConfig] = field(
        default_factory=lambda: {
            "wulongcha": FoundationPoseObjectConfig(
                mesh_path="../assets/objects/wulongcha/normalized.obj",
            ),
            "chouzhi": FoundationPoseObjectConfig(
                mesh_path="../assets/objects/chouzhi/normalized.obj",
            ),
            "leshishupian": FoundationPoseObjectConfig(
                mesh_path="../assets/objects/leshishupian/normalized.obj",
            ),
            "xinlamian": FoundationPoseObjectConfig(
                mesh_path="../assets/objects/xinlamian/normalized.obj",
            ),
            "kekoukele": FoundationPoseObjectConfig(
                mesh_path="../assets/objects/kekoukele/normalized.obj",
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
        """校验全部 FoundationPose 候选的 CAD 和感知参数。"""

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
            raise ValueError("foundationpose_objects 至少需要配置一种识别物料")
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


@dataclass
class LoggingConfig:
    """日志配置。"""

    log_to_console: bool = True  # 是否输出到控制台
    log_to_file: bool = True  # 是否保存到文件
    log_level: str = "INFO"  # 日志级别
    log_dir: str = "logs"  # 日志文件目录


@dataclass
class PipelineConfig:
    """视觉感知配置及日志选项。"""

    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

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

# 默认感知配置实例
cfg = PipelineConfig()
