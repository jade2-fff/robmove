"""
通用工具函数
包含：坐标变换、几何计算、安全确认等纯工具函数
"""

import sys
from typing import Sequence, Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R

from logger import logger

# ══════════════════════════════════════════════════════════════════
# 坐标变换（纯数学计算，无SDK依赖）
# ══════════════════════════════════════════════════════════════════

def pose_to_matrix(pose: Sequence[float]) -> np.ndarray:
    """
    [x, y, z, qx, qy, qz, qw] → 4×4 齐次变换矩阵
    """
    mat = np.eye(4)
    try:
        mat[:3, :3] = R.from_quat(pose[3:]).as_matrix()
    except AttributeError:
        mat[:3, :3] = R.from_quat(pose[3:]).as_dcm()
    mat[:3, 3] = pose[:3]
    return mat


def pixel_to_3d(
    u: int,
    v: int,
    depth_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """
    针孔反投影：像素(u, v) + 深度 → 相机系3D点
    纯数学计算
    """
    if depth_m <= 0:
        raise ValueError("深度值必须大于0")
    xc = (u - cx) * depth_m / fx
    yc = (v - cy) * depth_m / fy
    return np.array([xc, yc, depth_m])


def transform_3d_point(point_cam: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """
    将相机系3D点变换到base_link系
    纯数学计算
    """
    p_cam = np.append(point_cam, 1.0)
    return (transform_matrix @ p_cam)[:3]


def reachability_sort_key(
    position_base: Sequence[float], confidence: float, arm_name: str
) -> Tuple[float, float, float]:
    """
    多个候选目标同时可见时的排序依据：优先离机械臂更近、更顺手的一个，
    而不是置信度最高的一个（置信度仅在位置相同时作为最后的 tie-break）。

    position_base: base_link 系下的候选位置 [x, y, z, ...]
    arm_name: 当前使用的机械臂，用于决定 y 方向的优先侧
    """
    x, y = position_base[0], position_base[1]
    y_key = -y if arm_name == "left_arm" else y
    return float(x), float(y_key), -float(confidence)


# ══════════════════════════════════════════════════════════════════
# 几何计算
# ══════════════════════════════════════════════════════════════════

def rot_to_quat(rot: np.ndarray) -> np.ndarray:
    """3x3 旋转矩阵 → [qx, qy, qz, qw]"""
    try:
        return R.from_matrix(rot).as_quat()
    except AttributeError:
        return R.from_dcm(rot).as_quat()


def calculate_gripper_close_width(
    obj_width_m: float,
    margin: float,
    min_width: float = 0.005,
) -> float:
    """根据物体宽度计算夹爪闭合宽度"""
    return max(min_width, obj_width_m - margin)


# ══════════════════════════════════════════════════════════════════
# 用户交互
# ══════════════════════════════════════════════════════════════════

def check_robot_safety() -> None:
    """机器人安全检查"""
    logger.info(
        "\n⚠️  安全确认：\n"
        "  1. 急停按钮已释放\n"
        "  2. 机器人四周无障碍物\n"
        "  3. 桌面上只放待抓物体，其余区域已清空\n"
    )
    while True:
        k = input("确认后继续？(y/n): ").strip().lower()
        if k == "y":
            logger.info("已确认，继续...\n")
            return
        if k == "n":
            logger.error("未确认，退出。")
            sys.exit(0)
        logger.info("请输入 y 或 n")
