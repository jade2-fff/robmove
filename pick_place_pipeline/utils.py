"""
通用工具函数
包含：坐标变换和几何计算等纯工具函数
"""

from typing import Sequence
import numpy as np
from scipy.spatial.transform import Rotation as R


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




# ══════════════════════════════════════════════════════════════════
# 几何计算
# ══════════════════════════════════════════════════════════════════

def rot_to_quat(rot: np.ndarray) -> np.ndarray:
    """3x3 旋转矩阵 → [qx, qy, qz, qw]"""
    try:
        return R.from_matrix(rot).as_quat()
    except AttributeError:
        return R.from_dcm(rot).as_quat()
