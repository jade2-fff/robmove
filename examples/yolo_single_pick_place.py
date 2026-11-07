"""独立的 YOLO 单物体抓取、原地放回与归位示例。

主任务只使用 FoundationPose；本文件把 YOLO 专用配置和点位抓取规划限制在
example 内，用于单独跑通相机、分割和基础抓取动作。

从项目根目录运行：

    conda activate foundationpose
    python3 examples/yolo_single_pick_place.py \
        --target_class wulongcha

退出码：0=正常完成或有效空场，1=启动/异常错误，2=进入 HOLD，130=用户中断。
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pick_place_pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

# config 会先加载 Galbot SDK；必须早于 NumPy/OpenCV/PyTorch/Ultralytics。
from config import cfg
from logger import logger

import numpy as np

from perception import Perception, PerceptionUnavailableError, YoloDetection
from pick_place_task import (
    GraspPlan,
    PerceptionResult,
    PerceptionStatus,
    PickPlaceTask,
    PickTarget,
)
from utils import pixel_to_3d, reachability_sort_key, rot_to_quat


@dataclass(frozen=True)
class YoloExampleConfig:
    """只供本 example 使用的 YOLO 水平抓取参数。"""

    target_class: str = "leshishupian"
    weights: str = "assets/models/yolo11n-seg_finetuned.pt"
    conf_threshold: float = 0.25
    pre_grasp_distance: float = 0.05
    gripper_ee_distance: float = 0.15
    approach_yaw_deg: float = 0.0

    def validate(self) -> None:
        if not self.target_class:
            raise ValueError("target_class 不能为空")
        if not self.weights:
            raise ValueError("weights 不能为空")
        if (
            not np.isfinite(self.conf_threshold)
            or not 0.0 < self.conf_threshold <= 1.0
        ):
            raise ValueError("conf_threshold 必须在 (0, 1] 范围内")
        for name in ("pre_grasp_distance", "gripper_ee_distance"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} 必须是正的有限数值")
        if not np.isfinite(self.approach_yaw_deg):
            raise ValueError("approach_yaw_deg 必须是有限数值")


def _horizontal_plan(
    position_base: np.ndarray, settings: YoloExampleConfig
) -> GraspPlan:
    yaw = np.deg2rad(settings.approach_yaw_deg)
    rotation = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    approach_axis = rotation[:, 0]
    quaternion = rot_to_quat(rotation).tolist()
    grasp_position = position_base - approach_axis * settings.gripper_ee_distance
    pre_position = grasp_position - approach_axis * settings.pre_grasp_distance
    return GraspPlan(
        pre_grasp=[*pre_position.tolist(), *quaternion],
        grasp=[*grasp_position.tolist(), *quaternion],
    )


def _perceive_yolo(
    task: PickPlaceTask,
    perception: Perception,
    settings: YoloExampleConfig,
) -> PerceptionResult:
    """获取一帧并把 YOLO 3D 点转换为公共 ``PickTarget``。"""

    robot = task.robot
    model = perception.detection_model
    if robot is None or not isinstance(model, YoloDetection):
        return PerceptionResult(
            PerceptionStatus.UNAVAILABLE,
            reason="YOLO example 感知资源尚未初始化",
        )
    try:
        if settings.target_class not in model.available_class_names():
            raise PerceptionUnavailableError(
                f"YOLO 权重不包含类别 {settings.target_class!r}；"
                f"权重类别={sorted(model.available_class_names())}；"
                f"权重原始类别={sorted(model.raw_class_names())}"
            )
        rgb, depth = perception.get_images(robot)
        if rgb is None or depth is None:
            raise PerceptionUnavailableError("未获取到有效 RGB-D 帧")
        intrinsics = perception.get_camera_intrinsics(robot)
        transform_base_camera = perception.camera_transform(robot)
        raw_detections = model.detect_raw(
            rgb,
            depth,
            {settings.target_class},
            depth_required=True,
            debug=cfg.debug_mode,
        )
        if not raw_detections:
            return PerceptionResult(
                PerceptionStatus.EMPTY,
                reason=f"未发现 YOLO 类别 {settings.target_class!r}",
            )

        def candidate_data(raw):
            if raw.depth_m is None:
                raise PerceptionUnavailableError("YOLO 候选缺少有效深度")
            point_camera = pixel_to_3d(*raw.pixel, raw.depth_m, *intrinsics)
            point_base = (
                transform_base_camera @ np.append(point_camera, 1.0)
            )[:3]
            key = reachability_sort_key(
                point_base, raw.confidence, cfg.arm_name
            )
            return key, point_base

        ranked = [(raw, *candidate_data(raw)) for raw in raw_detections]
        selected, _, position_base = min(ranked, key=lambda item: item[1])
        object_width_m = (
            selected.mask_width_px * selected.depth_m / intrinsics[0]
            if selected.mask_width_px is not None and selected.depth_m is not None
            else None
        )
        target = PickTarget(
            class_name=selected.class_name,
            confidence=selected.confidence,
            top_down=False,
            plan=_horizontal_plan(position_base, settings),
            object_width_m=object_width_m,
        )
        logger.info(
            "YOLO example 选择目标: "
            f"class={target.class_name}, conf={target.confidence:.2f}, "
            f"base={np.round(position_base, 3).tolist()}"
        )
        return PerceptionResult(PerceptionStatus.FOUND, target=target)
    except PerceptionUnavailableError as error:
        return PerceptionResult(PerceptionStatus.UNAVAILABLE, reason=str(error))
    except Exception as error:
        logger.exception(f"YOLO example 感知或规划异常: {error}")
        return PerceptionResult(
            PerceptionStatus.UNAVAILABLE,
            reason=f"YOLO example 感知或规划异常: {error}",
        )


def run(settings: YoloExampleConfig) -> bool:
    settings.validate()
    model = YoloDetection(
        use_segmentation=True,
        conf_threshold=settings.conf_threshold,
        yolo_weights=settings.weights,
        class_aliases=cfg.perception.yolo_class_aliases,
    )
    perception = Perception(
        config=cfg.perception,
        detection_model=model,
        arm_name=cfg.arm_name,
        debug_mode=cfg.debug_mode,
    )

    with PickPlaceTask(
        cfg,
        enable_navigation=False,
        perception=perception,
    ) as task:
        if not task.return_arm_home():
            task.hold("感知前机械臂归位失败")
            return False

        result: Optional[PerceptionResult] = None
        target: Optional[PickTarget] = None
        for attempt in range(1, cfg.task.perception_retries + 1):
            result = _perceive_yolo(task, perception, settings)
            logger.info(
                f"YOLO example 感知结果: {result.status.name}, "
                f"attempt={attempt}/{cfg.task.perception_retries}, "
                f"reason={result.reason}"
            )
            if result.status is PerceptionStatus.FOUND:
                target = result.target
                break
            if attempt < cfg.task.perception_retries:
                time.sleep(cfg.task.perception_retry_interval_s)

        if target is None:
            if result is not None and result.status is PerceptionStatus.EMPTY:
                logger.info("YOLO example 未发现目标，安全退出")
                return True
            reason = result.reason if result and result.reason else "感知链路不可用"
            task.hold(reason)
            return False

        for action, reason in (
            (lambda: task.pick(target), "抓取动作失败"),
            (lambda: task.lift(target), "抓取后抬升失败"),
            (lambda: task.place(target), "放置或释放失败"),
            (lambda: task.retreat_from_place(target), "放置后退回失败"),
            (task.return_arm_home, "释放后机械臂归位失败"),
        ):
            if not action():
                task.hold(reason)
                return False
        logger.info("YOLO 单物体抓取放置完成")
        return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target_class", default="leshishupin")
    parser.add_argument("--weights", default="assets/models/yolo11n-seg_finetuned.pt")
    parser.add_argument("--conf_threshold", type=float, default=0.25)
    parser.add_argument("--pre_grasp_distance", type=float, default=0.05)
    parser.add_argument("--gripper_ee_distance", type=float, default=0.15)
    parser.add_argument("--approach_yaw_deg", type=float, default=0.0)
    parser.add_argument("--perception_retries", type=int)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.perception_retries is not None:
        cfg.task.perception_retries = args.perception_retries
    cfg.navigation.enable_navigation = False
    settings = YoloExampleConfig(
        target_class=args.target_class,
        weights=args.weights,
        conf_threshold=args.conf_threshold,
        pre_grasp_distance=args.pre_grasp_distance,
        gripper_ee_distance=args.gripper_ee_distance,
        approach_yaw_deg=args.approach_yaw_deg,
    )
    try:
        return 0 if run(settings) else 2
    except KeyboardInterrupt:
        logger.warning("YOLO example 已被用户中断")
        return 130
    except Exception as error:
        logger.exception(f"YOLO example 启动或执行异常: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
