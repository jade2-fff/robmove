"""抓取任务的公共执行层。

本模块统一管理机器人资源、感知结果、抓取规划和原子动作。流程入口只
状态编排，不直接调用 Galbot SDK。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

# Galbot SDK 必须先于 NumPy/OpenCV/PyTorch/Ultralytics 导入。
import galbot_sdk.g1 as gm
from galbot_sdk.g1 import GalbotMotion, GalbotNavigation, GalbotRobot

import numpy as np

from config import FoundationPoseObjectConfig, PipelineConfig
from logger import logger
from motion_planning import MotionPlanner, print_motion_status
from navigation import Navigator
from perception import (
    Perception,
    PerceptionUnavailableError,
)
from utils import (
    calculate_gripper_close_width,
    check_robot_safety,
    pose_to_matrix,
    rot_to_quat,
)


HOME_JOINT_TOLERANCE_RAD = 1e-3


class PerceptionStatus(Enum):
    """一次感知的业务状态。"""

    FOUND = auto()
    EMPTY = auto()
    UNAVAILABLE = auto()


@dataclass(frozen=True)
class GraspPlan:
    """预抓取与抓取末端位姿。"""

    pre_grasp: List[float]
    grasp: List[float]


@dataclass(frozen=True)
class PickTarget:
    """与底层感知算法解耦的统一抓取目标。"""

    class_name: str
    confidence: float
    top_down: bool
    plan: GraspPlan
    object_width_m: Optional[float] = None


@dataclass(frozen=True)
class PerceptionResult:
    """一次感知的状态、目标与诊断原因。"""

    status: PerceptionStatus
    target: Optional[PickTarget] = None
    reason: Optional[str] = None


class PickPlaceTask:
    """公共机器人资源与抓取、放置动作的唯一实现。"""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        enable_navigation: bool,
        perception: Optional[Perception] = None,
        load_perception_on_initialize: bool = True,
    ) -> None:
        self.config = config
        self.enable_navigation = enable_navigation
        self.load_perception_on_initialize = load_perception_on_initialize

        self.robot: Optional[GalbotRobot] = None
        self.motion: Optional[GalbotMotion] = None
        self.navigation: Optional[GalbotNavigation] = None
        self.perception: Optional[Perception] = perception
        self.motion_planner: Optional[MotionPlanner] = None
        self.navigator: Optional[Navigator] = None

        self.home_joint_positions: List[float] = []
        self.carrying_object = False
        self.hold_reason: Optional[str] = None
        self._perception_model_loaded = False
        self._shutdown = False

    def __enter__(self) -> "PickPlaceTask":
        try:
            self.initialize()
        except BaseException as error:
            reason = (
                "用户在初始化阶段中断任务"
                if isinstance(error, KeyboardInterrupt)
                else f"任务初始化失败: {error}"
            )
            self.hold(reason)
            self.shutdown()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_value is not None and not isinstance(exc_value, KeyboardInterrupt):
            self.hold(f"未处理异常: {exc_value}")
        elif isinstance(exc_value, KeyboardInterrupt):
            self.hold("用户中断任务")
        self.shutdown()
        return False

    def initialize(self) -> None:
        """验证配置并初始化任务所需的全部资源。"""

        self.config.validate_task(enable_navigation=self.enable_navigation)
        if self.config.safety_check:
            check_robot_safety()

        logger.info(
            "初始化抓取任务资源: "
            f"arm={self.config.arm_name}, navigation={self.enable_navigation}"
        )
        self.robot = GalbotRobot()
        self.motion = GalbotMotion()
        self.navigation = GalbotNavigation() if self.enable_navigation else None

        if not self.robot.init(set(self.config.enabled_sensors), True):
            raise RuntimeError("机器人初始化失败")
        if not self.motion.init():
            raise RuntimeError("运动模块初始化失败")
        if self.navigation is not None and not self.navigation.init():
            raise RuntimeError("导航模块初始化失败")
        time.sleep(2.0)

        self.motion_planner = MotionPlanner(self.motion)
        self.navigator = Navigator(self.config)

        if self.navigation is not None:
            if not self.navigator.ensure_nav_controller(self.robot):
                raise RuntimeError("无法切换到底盘导航控制器")
            if not self.navigator.ensure_localized(
                self.navigation, self.config.origin_pose
            ):
                raise RuntimeError("机器人重定位失败")

        self.home_joint_positions = self._get_home_joints()
        if self.perception is None:
            self.perception = Perception(
                config=self.config.perception,
                arm_name=self.config.arm_name,
                debug_mode=self.config.debug_mode,
            )
        if self.load_perception_on_initialize:
            self.load_perception_model()
        logger.info("抓取任务资源初始化完成")

    def load_perception_model(self) -> None:
        """幂等加载感知模型；循环流程可在导航到抓取点后显式调用。"""

        if self._perception_model_loaded:
            logger.info("感知模型已加载，跳过重复加载")
            return
        logger.info("加载感知模型")
        self._require_perception().load_model()
        self._perception_model_loaded = True
        logger.info("感知模型加载完成")

    def return_arm_home(self) -> bool:
        """将机械臂移到标定默认姿态，已到位时跳过命令。"""

        robot = self._require_robot()
        current_joints = robot.get_joint_positions([self.config.arm_name], [])
        target_joints = self.home_joint_positions
        if len(current_joints) == len(target_joints) and current_joints:
            current = np.asarray(current_joints, dtype=float)
            target = np.asarray(target_joints, dtype=float)
            if np.all(np.isfinite(current)) and np.all(np.isfinite(target)):
                max_error = float(np.max(np.abs(current - target)))
                if max_error <= HOME_JOINT_TOLERANCE_RAD:
                    logger.info(
                        f"{self.config.arm_name} 已在默认关节姿态，"
                        "跳过归位命令 "
                        f"(max_error={max_error:.6f}rad)"
                    )
                    return True

        result = robot.set_joint_positions(
            joint_positions=target_joints,
            joint_groups=[self.config.arm_name],
            joint_names=[],
            is_blocking=True,
            speed_rad_s=self.config.joint_speed,
            timeout_s=25.0,
        )
        if result == gm.ControlStatus.SUCCESS:
            logger.info("机械臂已回到默认关节姿态")
            return True

        active_controller = robot.get_active_controller(self.config.arm_name)
        if result == gm.ControlStatus.FAULT:
            logger.error(
                f"机械臂归位失败: {result}, "
                f"active_controller={active_controller}；"
                "关节组存在 fatal error，请检查急停、驱动器和控制器"
            )
        else:
            logger.error(
                f"机械臂归位失败: {result}, "
                f"active_controller={active_controller}"
            )
        return False

    def perceive_one(self) -> PerceptionResult:
        """获取新帧并返回 FOUND、EMPTY 或 UNAVAILABLE。"""

        if not self._perception_model_loaded:
            logger.warning("感知模型尚未加载，执行按需加载")
            self.load_perception_model()
        perception = self._require_perception()
        robot = self._require_robot()
        try:
            candidates = perception.detect_foundationpose_candidates(robot)
            if not candidates:
                return PerceptionResult(
                    PerceptionStatus.EMPTY,
                    reason="有效 RGB-D 感知未发现 FoundationPose 候选",
                )
            selected = candidates[0]
            result = self._plan_foundationpose_grasp(
                selected.pose_base, selected.class_name
            )
            if result is None:
                return PerceptionResult(
                    PerceptionStatus.UNAVAILABLE,
                    reason="FoundationPose 抓取方向规划失败",
                )
            plan, is_top_down = result
            target = PickTarget(
                class_name=selected.class_name,
                confidence=selected.confidence,
                top_down=is_top_down,
                plan=plan,
                object_width_m=selected.obj_width_m,
            )
        except PerceptionUnavailableError as error:
            return PerceptionResult(PerceptionStatus.UNAVAILABLE, reason=str(error))
        except Exception as error:
            logger.exception(f"感知或抓取规划异常: {error}")
            return PerceptionResult(
                PerceptionStatus.UNAVAILABLE,
                reason=f"感知或模型推理异常: {error}",
            )

        logger.info(
            f"选择目标: class={target.class_name}, conf={target.confidence:.2f}, "
            f"approach={'top_down' if target.top_down else 'side'}, "
            f"pre={np.round(target.plan.pre_grasp[:3], 3).tolist()}, "
            f"grasp={np.round(target.plan.grasp[:3], 3).tolist()}"
        )
        return PerceptionResult(PerceptionStatus.FOUND, target=target)

    def pick(self, target: PickTarget) -> bool:
        """打开夹爪并依次执行预抓取、抓取和闭合。"""

        close_width = self.config.gripper_close_width
        if (
            self.config.auto_gripper_close_width
            and target.object_width_m is not None
        ):
            close_width = calculate_gripper_close_width(
                target.object_width_m, self.config.gripper_width_margin
            )

        logger.info(f"打开夹爪，计划闭合宽度: {close_width:.4f}m")
        if not self._command_gripper(self.config.gripper_open_width):
            return False
        if not self._move_to_pose(target.plan.pre_grasp, "预抓取位姿"):
            return False
        if not self._move_to_pose(target.plan.grasp, "抓取位姿"):
            return False
        if not self._command_gripper_for_grasp(close_width):
            return False

        self.carrying_object = True
        logger.info(f"夹爪已闭合，当前持有目标: {target.class_name}")
        return True

    def lift(self, target: PickTarget) -> bool:
        """闭合夹爪后沿抓取姿态原地竖直上抬 lift_height，避免拖拽物体或磕碰周边。"""

        lift_pose = self._lift_pose(target.plan.grasp)
        return self._move_to_pose(lift_pose, "抬升至安全高度")

    def lift_safe(self, target: PickTarget) -> bool:
        """在抬升高度上水平回退，形成携物导航前的安全姿态（lift_safe）。

        侧面接近沿用预抓取位姿的水平位置，仅把 z 换成抬升高度；竖直向下
        接近没有有效水平接近轴，沿 base_link -X 回退该物体配置的
        ``pre_grasp_distance``。两种情况均保持抓取朝向。
        """

        lift_pose = self._lift_pose(target.plan.grasp)
        safe_pose = self._lift_safe_pose(target, lift_pose)
        return self._move_to_pose(safe_pose, "水平回退至安全携物位姿")

    def place(self, target: PickTarget) -> bool:
        """单物体流程：下降回抓取位姿并打开夹爪，原地放回原位（与 grasp 对称）。

        不依赖 place_release_pose 标定：放置位姿直接取本轮抓取位姿
        （抓取点相对 base_link 在哪，放置点就在哪）。仅用于不导航的单物体
        流程；循环流程放置到料框时请用 :meth:`place_local`（从抬升高度
        上方释放，不下压到框底）。
        """

        if not self.carrying_object:
            logger.error("放置失败：任务未记录到持有物体")
            return False
        if not self._move_to_pose(target.plan.grasp, "放置位姿"):
            return False
        if not self._command_gripper(self.config.gripper_open_width):
            return False

        self.carrying_object = False
        logger.info("物品已释放")
        return True

    def retreat_from_place(self, target: PickTarget) -> bool:
        """单物体流程：释放后沿原接近路径退回本轮预抓取位姿。"""

        return self._move_to_pose(target.plan.pre_grasp, "放置后退回预抓取位姿")

    def place_local(self, target: PickTarget) -> bool:
        """循环流程：经放置点 lift_safe 进入 lift 位姿，从抬升高度上方释放物体。

        不依赖 place_release_pose 标定：放置位姿直接取本轮抓取位姿
        （抓取点相对 base_link 在哪，导航到位后放置点就在哪）。到达放置
        导航点后，先走到放置点坐标系下的 lift_safe，再水平进入其正上方
        lift_height 处松爪，与抓取侧的 lift_safe -> lift 路径保持一致，
        也避免从携物归位姿态直接斜插到料框上方。
        """

        if not self.carrying_object:
            logger.error("放置失败：任务未记录到持有物体")
            return False
        place_lift_pose = self._lift_pose(target.plan.grasp)
        place_safe_pose = self._lift_safe_pose(target, place_lift_pose)
        if not self._move_to_pose(place_safe_pose, "放置区 lift_safe 位姿"):
            return False
        if not self._move_to_pose(place_lift_pose, "放置区安全抬升位姿"):
            return False
        if not self._command_gripper(self.config.gripper_open_width):
            return False

        self.carrying_object = False
        logger.info("物品已在放置区抬升位姿释放")
        return True

    def retreat_to_lift_safe(self, target: PickTarget) -> bool:
        """循环流程：释放后水平退回放置点坐标系下的 lift_safe 位姿。"""

        lift_pose = self._lift_pose(target.plan.grasp)
        safe_pose = self._lift_safe_pose(target, lift_pose)
        return self._move_to_pose(safe_pose, "放置后退回 lift_safe")

    def _lift_pose(self, pose: List[float]) -> List[float]:
        """将给定位姿沿 z 轴上抬 lift_height，朝向保持不变。"""

        lifted = list(pose)
        lifted[2] += self.config.lift_height
        return lifted

    def _lift_safe_pose(
        self, target: PickTarget, lift_pose: List[float]
    ) -> List[float]:
        """在 lift_pose 高度上计算水平回退后的安全位姿。

        FoundationPose 对斜躺/平躺物体采用竖直接近，此时沿 base_link -X
        回退；侧面接近则沿用预抓取位姿的水平位置。
        """

        if target.top_down:
            safe = list(lift_pose)
            object_config = self.config.perception.foundationpose_objects[
                target.class_name
            ]
            safe[0] -= object_config.pre_grasp_distance
            return safe
        pre_grasp = target.plan.pre_grasp
        return [pre_grasp[0], pre_grasp[1], lift_pose[2], *lift_pose[3:]]

    def navigate_to(
        self,
        pose: Optional[List[float]],
        name: str,
        lock_heading: bool = False,
    ) -> bool:
        """导航到命名路点。

        Args:
            pose: 目标位姿 [x, y, z, qx, qy, qz, qw]
            name: 路点名称（用于日志）
            lock_heading: 是否保持当前朝向平移（受配置
                navigation.lock_heading 总开关约束）
        """

        if self.navigation is None or self.navigator is None:
            logger.error(f"无法导航到{name}：导航模块未初始化")
            return False
        if pose is None:
            logger.error(f"无法导航到{name}：路点未配置")
            return False
        logger.info(f"导航目标: {name}")
        success = self.navigator.navigate_to_waypoint(
            self.navigation,
            pose,
            lock_heading=lock_heading and self.config.navigation.lock_heading,
        )
        logger.info(f"导航结果: name={name}, success={success}")
        return success

    def hold(self, reason: str) -> None:
        """停止底盘与轨迹，并保留首个故障原因。"""

        if self.hold_reason is not None:
            return
        self.hold_reason = reason
        if self.robot is not None:
            stop_commands = (
                ("底盘", self.robot.stop_base),
                ("轨迹", self.robot.stop_trajectory_execution),
            )
            for name, stop in stop_commands:
                try:
                    stop()
                except Exception as error:
                    logger.error(f"停止{name}失败: {error}")
        logger.error(f"任务进入 HOLD: {reason}；已停止底盘和轨迹，禁止恢复")
        if self.carrying_object:
            logger.warning("机器人仍持有物体，请人工处理；不会自动归位")

    def shutdown(self) -> None:
        """幂等释放机器人 SDK 资源，不在退出阶段发送运动命令。"""

        if self._shutdown:
            return
        self._shutdown = True
        if self.robot is None:
            return
        try:
            self.robot.request_shutdown()
            self.robot.wait_for_shutdown()
        except Exception as error:
            logger.error(f"等待机器人关闭失败: {error}")
        finally:
            try:
                self.robot.destroy()
            except Exception as error:
                logger.error(f"释放机器人资源失败: {error}")
        logger.info("机器人资源已释放")

    def _get_home_joints(self) -> List[float]:
        joints = self._require_robot().get_joint_positions(
            [self.config.arm_name], []
        )
        if not joints:
            raise RuntimeError("无法读取机械臂初始关节位置")
        if self.config.home_joint_positions is not None:
            if len(self.config.home_joint_positions) != len(joints):
                raise RuntimeError(
                    "home_joint_positions 长度 "
                    f"{len(self.config.home_joint_positions)} 与 "
                    f"{self.config.arm_name} 实际关节数 {len(joints)} 不一致"
                )
            return list(self.config.home_joint_positions)
        logger.warning(
            "未配置 home_joint_positions，使用任务启动姿态作为默认姿态"
        )
        return list(joints)

    def _plan_foundationpose_grasp(
        self, pose_base: np.ndarray, class_name: str
    ) -> Optional[Tuple[GraspPlan, bool]]:
        """返回抓取规划及是否为竖直向下（斜躺/平躺物体）接近。

        调用方把结果写入 ``PickTarget.top_down``；竖直接近时
        :meth:`_lift_safe_pose` 沿 base_link -X 做水平安全回退。
        """

        object_config = self.config.perception.foundationpose_objects.get(class_name)
        if object_config is None:
            logger.error(
                f"FoundationPose 未配置目标物体 {class_name!r} 的抓取偏移；可选值: "
                f"{sorted(self.config.perception.foundationpose_objects)}"
            )
            return None

        center = pose_base @ np.array([*object_config.grasp_point_object, 1.0])
        object_z = pose_base[:3, 2].copy()
        object_z_norm = np.linalg.norm(object_z)
        if object_z_norm < 1e-6:
            logger.error("FoundationPose 物体 Z 轴无效")
            return None
        object_z /= object_z_norm

        # 物体对称轴接近水平（斜躺/平躺）时，绕该轴的滚转角本身不可观测也不重要：
        # 相当于物体绕 Y 轴倒下了约 90 度，夹爪也整体跟着倒下，改为竖直向下接近，
        # 再和竖立情形一样让夹爪 Z 轴与物体 Z 轴保持平行（而不是只对齐水平投影的
        # yaw，这样物体本身有残余倾斜时夹爪姿态依然贴合）。
        angle_from_vertical_deg = np.degrees(
            np.arccos(np.clip(abs(object_z[2]), -1.0, 1.0))
        )
        is_top_down = (
            angle_from_vertical_deg > self.config.foundationpose_lying_angle_threshold_deg
        )
        if is_top_down:
            approach_seed = np.array([0.0, 0.0, -1.0])
        else:
            if self.config.arm_name == "left_arm":
                min_angle = np.deg2rad(-self.config.work_angle_max_deg)
                max_angle = np.deg2rad(-self.config.work_angle_min_deg)
            else:
                min_angle = np.deg2rad(self.config.work_angle_min_deg)
                max_angle = np.deg2rad(self.config.work_angle_max_deg)
            approach_seed = self._foundationpose_approach(
                pose_base, min_angle, max_angle
            )
            if approach_seed is None:
                logger.error("FoundationPose 无法找到有效水平接近方向")
                return None

        plan = self._build_foundationpose_grasp(
            center, object_z, approach_seed, object_config
        )
        if plan is None:
            return None
        return plan, is_top_down

    def _build_foundationpose_grasp(
        self,
        center: np.ndarray,
        object_z: np.ndarray,
        approach_seed: np.ndarray,
        object_config: FoundationPoseObjectConfig,
    ) -> Optional[GraspPlan]:
        """按 Gram-Schmidt 把接近方向投影到与物体 Z 轴正交，夹爪 Z 轴取物体 Z 轴。"""

        gripper_x = approach_seed - object_z * np.dot(approach_seed, object_z)
        gripper_x_norm = np.linalg.norm(gripper_x)
        if gripper_x_norm < 0.1:
            logger.error("FoundationPose 接近方向与物体 Z 轴接近平行")
            return None
        gripper_x /= gripper_x_norm
        gripper_y = np.cross(object_z, gripper_x)
        gripper_y /= np.linalg.norm(gripper_y)
        gripper_x = np.cross(gripper_y, object_z)
        gripper_x /= np.linalg.norm(gripper_x)

        rotation = np.column_stack([gripper_x, gripper_y, object_z])
        quaternion = rot_to_quat(rotation).tolist()
        grasp_position = (
            center[:3]
            - gripper_x * object_config.gripper_ee_distance
        )
        pre_position = (
            grasp_position
            - gripper_x * object_config.pre_grasp_distance
        )

        if self.config.debug_mode and self.perception is not None:
            self.perception.save_grasp_debug_image(
                pose_to_matrix([*grasp_position.tolist(), *quaternion])
            )

        return GraspPlan(
            pre_grasp=[*pre_position.tolist(), *quaternion],
            grasp=[*grasp_position.tolist(), *quaternion],
        )

    def _foundationpose_approach(
        self, pose_base: np.ndarray, min_angle: float, max_angle: float
    ) -> Optional[np.ndarray]:
        def angle_distance(angle: float, target: float) -> float:
            return abs((angle - target + np.pi) % (2 * np.pi) - np.pi)

        best: Optional[np.ndarray] = None
        best_distance = float("inf")
        for axis_index in (0, 1):
            for sign in (1.0, -1.0):
                candidate = pose_base[:3, axis_index] * sign
                horizontal = np.array([candidate[0], candidate[1], 0.0])
                norm = np.linalg.norm(horizontal)
                if norm < 0.1:
                    continue
                horizontal /= norm
                angle = np.arctan2(horizontal[1], horizontal[0])
                if min_angle <= angle <= max_angle:
                    return horizontal
                target_angle = (
                    min_angle
                    if angle_distance(angle, min_angle)
                    <= angle_distance(angle, max_angle)
                    else max_angle
                )
                distance = angle_distance(angle, target_angle)
                if distance < best_distance:
                    best_distance = distance
                    best = np.array(
                        [np.cos(target_angle), np.sin(target_angle), 0.0]
                    )
        return best

    def _command_gripper(self, width_m: float) -> bool:
        """阻塞执行夹爪开合；用于打开和释放等必须到达目标宽度的动作。"""

        result = self._require_robot().set_gripper_command(
            end_effector=self.config.gripper_name,
            width_m=width_m,
            velocity_mps=self.config.gripper_speed,
            effort=self.config.gripper_effort,
            is_blocking=True,
        )
        if result != gm.ControlStatus.SUCCESS:
            logger.error(f"夹爪命令失败: {result}")
            return False
        return True

    def _command_gripper_for_grasp(self, width_m: float) -> bool:
        """非阻塞闭合夹爪，确认夹持力和开口宽度稳定后返回成功。"""

        robot = self._require_robot()
        result = robot.set_gripper_command(
            end_effector=self.config.gripper_name,
            width_m=width_m,
            velocity_mps=self.config.gripper_speed,
            effort=self.config.gripper_effort,
            is_blocking=False,
        )
        if result != gm.ControlStatus.SUCCESS:
            logger.error(f"抓取闭合命令发送失败: {result}")
            return False

        threshold = self.config.gripper_grasp_effort_threshold
        required_samples = self.config.gripper_grasp_effort_stable_samples
        width_tolerance = self.config.gripper_grasp_width_stable_tolerance_m
        poll_interval = self.config.gripper_state_poll_interval_s
        deadline = time.monotonic() + self.config.gripper_grasp_timeout_s
        stable_samples = 0
        last_width: Optional[float] = None
        last_effort: Optional[float] = None
        last_is_moving: Optional[bool] = None

        logger.info(
            f"夹爪已开始非阻塞闭合: target_width={width_m:.4f}m, "
            f"effort_threshold={threshold:.2f}, "
            f"stable_samples={required_samples}, "
            f"width_stable_tolerance={width_tolerance:.4f}m"
        )
        while time.monotonic() < deadline:
            state = robot.get_gripper_state(self.config.gripper_name)
            if state is None:
                logger.warning("暂时无法读取夹爪状态，继续等待")
            else:
                previous_width = last_width
                last_width = float(state.width)
                last_effort = abs(float(state.effort))
                last_is_moving = bool(state.is_moving)
                width_stable = (
                    previous_width is not None
                    and abs(last_width - previous_width) <= width_tolerance
                )
                logger.debug(
                    f"夹爪抓取状态: width={last_width:.4f}m, "
                    f"effort={last_effort:.2f}, moving={last_is_moving}, "
                    f"width_stable={width_stable}"
                )
                if (
                    np.isfinite(last_effort)
                    and last_effort >= threshold
                    and width_stable
                ):
                    stable_samples += 1
                    if stable_samples >= required_samples:
                        logger.info(
                            f"夹爪已夹稳: width={last_width:.4f}m, "
                            f"effort={last_effort:.2f} >= {threshold:.2f}, "
                            f"width_stable=True, moving={last_is_moving}"
                        )
                        return True
                else:
                    stable_samples = 0
            time.sleep(poll_interval)

        width_text = "unknown" if last_width is None else f"{last_width:.4f}m"
        effort_text = "unknown" if last_effort is None else f"{last_effort:.2f}"
        moving_text = "unknown" if last_is_moving is None else str(last_is_moving)
        logger.error(
            "抓取闭合确认超时：夹爪未停止或夹持力未稳定达到阈值；"
            f"last_width={width_text}, last_effort={effort_text}, "
            f"last_moving={moving_text}, threshold={threshold:.2f}"
        )
        return False

    def _move_to_pose(self, pose: List[float], operation: str) -> bool:
        planner = self._require_motion_planner()
        current_pose = planner.get_current_pose(self.config.arm_name)
        if current_pose is None:
            logger.warning(
                f"{operation}: 无法读取当前末端位姿，继续尝试目标位姿"
            )
        else:
            current_position = np.asarray(current_pose[:3], dtype=float)
            target_position = np.asarray(pose[:3], dtype=float)
            delta = target_position - current_position
            logger.info(
                f"{operation}: current={np.round(current_pose, 4).tolist()}, "
                f"target={np.round(pose, 4).tolist()}, "
                f"delta_xyz={np.round(delta, 4).tolist()}, "
                f"distance={np.linalg.norm(delta):.4f}m"
            )

        status = planner.move_to_pose(
            pose,
            self.config.arm_name,
            timeout=self.config.move_timeout,
        )
        if status != gm.MotionStatus.SUCCESS:
            print_motion_status(status, operation)
            return False
        return True

    def _require_robot(self) -> GalbotRobot:
        if self.robot is None:
            raise RuntimeError("机器人资源尚未初始化")
        return self.robot

    def _require_perception(self) -> Perception:
        if self.perception is None:
            raise RuntimeError("感知资源尚未初始化")
        return self.perception

    def _require_motion_planner(self) -> MotionPlanner:
        if self.motion_planner is None:
            raise RuntimeError("运动规划资源尚未初始化")
        return self.motion_planner
