"""
感知模块
核心职责：管理图像输入输出，运行 FoundationPose 感知链路

主流程固定为 FoundationPose：
- YoloDetection: YOLO-seg 分割模型，为 FoundationPose 提供种子 mask
  （``detect_raw()`` 也可独立提供检测结果）
- FoundationPoseModel: 按物料类别的 6DoF 位姿估计（GPU 本地推理）
- Perception: 图像获取（头部双目）、内参/外参读取、多类别候选选择与位姿配准
- HeadCameraPerception: 头部左目 RGB + FOUNDATION_STEREO 双目深度采集
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# galbot_sdk must be loaded before OpenCV/PyTorch. Loading it after those native
# runtimes changes YOLO inference results on the robot and can make it return no boxes.
from galbot_sdk.g1 import GalbotRobot, SensorType

import numpy as np
import cv2

from utils import pixel_to_3d, pose_to_matrix, transform_3d_point
from config import PerceptionConfig
from logger import logger


class PerceptionUnavailableError(RuntimeError):
    """RGB-D、相机内参或 TF 等感知链路当前不可用。"""


_WARP_CUDA_READY = False


def _ensure_warp_cuda_ready() -> None:
    """在首次加载 FoundationPose 前初始化并验证其 CUDA 路径。

    共享 YOLO 先完成 GPU 分割，再由此函数初始化 FoundationPose 所需的
    PyTorch/Warp CUDA runtime。
    """

    global _WARP_CUDA_READY
    if _WARP_CUDA_READY:
        return

    import torch
    import warp as wp

    logger.info("预初始化 FoundationPose 的 PyTorch/Warp CUDA runtime")
    if not torch.cuda.is_available():
        raise RuntimeError("FoundationPose 需要 CUDA，但 PyTorch 未检测到可用 GPU")
    torch_probe = torch.empty(1, device="cuda:0")
    del torch_probe
    wp.init()
    probe = wp.zeros(1, dtype=wp.float32, device="cuda:0")
    del probe
    _WARP_CUDA_READY = True
    logger.info("FoundationPose 的 PyTorch/Warp CUDA runtime 已就绪")


@dataclass
class RawYoloDetection:
    """YOLO 的二维检测及其对应深度，不包含任务策略。"""

    class_name: str
    confidence: float
    pixel: Tuple[int, int]
    mask_width_px: Optional[float]
    depth_m: Optional[float]
    # 分割轮廓（图像像素坐标），仅在 use_segmentation 时有值；用于按需重建
    # 二值 mask（例如喂给 FoundationPose），避免为此单独重新推理一次。
    contour: Optional[np.ndarray] = None


@dataclass
class FoundationPoseCandidate:
    """FoundationPose 候选，物体位姿统一表达在 ``base_link`` 系。"""

    class_name: str
    confidence: float
    position_base: np.ndarray
    pose_base: np.ndarray
    obj_width_m: Optional[float] = None


# ══════════════════════════════════════════════════════════════════
# YOLO 分割模型 - 为 FoundationPose 提供种子 mask；example 可独立使用
# ══════════════════════════════════════════════════════════════════


class YoloDetection:
    """YOLOv8 分割/检测模型

    主流程用它做实例分割，选出目标后把 mask 喂给 FoundationPose；
    可通过 :meth:`detect_raw` 独立获取检测结果
    （检测 + 深度图 → base_link 3D 点）。
    """

    def __init__(
        self,
        use_segmentation: bool = True,
        conf_threshold: float = 0.5,
        yolo_weights: str = "assets/models/yolov8l-seg.pt",
        device: str = "cuda:0",
        depth_median_radius: int = 5,
        class_aliases: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            use_segmentation: 是否使用分割模型
            conf_threshold: 置信度阈值
            device: YOLO 推理设备；主流程固定使用 Orin GPU
            depth_median_radius: 深度取值邻域半径（像素），见 :meth:`_get_robust_depth`
        """
        self.use_segmentation = use_segmentation
        self.conf_threshold = conf_threshold
        self.yolo_weights = yolo_weights
        self.device = device
        self.depth_median_radius = depth_median_radius
        self.class_aliases = class_aliases or {}
        self.model: Optional[object] = None
        self.loaded_weights_path: Optional[str] = None

    def load(self) -> None:
        """加载 YOLO 模型"""
        from ultralytics import YOLO

        code_dir = Path(__file__).resolve().parent
        weights_path = self._ensure_weights_file(self.yolo_weights, code_dir)
        self.loaded_weights_path = weights_path
        if self.use_segmentation:
            self.model = YOLO(weights_path)
            logger.info(f"YOLO 分割模型加载成功: {weights_path}")
        else:
            self.model = YOLO(weights_path)
            logger.info(f"YOLO 检测模型加载成功: {weights_path}")

    @staticmethod
    def _resolve_weights_path(weights: str, base_dir: Path) -> str:
        raw_weights = weights.strip()
        if raw_weights.startswith("..assets/") or raw_weights.startswith("..assets\\"):
            raw_weights = f"../assets/{raw_weights[len('..assets/'):]}"

        p = Path(raw_weights)
        if p.is_absolute():
            return str(p)

        repo_dir = base_dir.parent
        # 默认模型文件统一放在仓库根目录 assets/models/ 下。
        # 同时兼容历史上以 pick_place_pipeline/ 为基准写的 ../assets/...。
        candidates = []
        if p.parts[:1] == ("assets",):
            candidates.append(repo_dir / p)
        candidates.extend([base_dir / p, repo_dir / p])
        if p.parts[:1] in (("pick_place_pipeline",), ("code",)) and len(p.parts) > 1:
            candidates.append(repo_dir / Path(*p.parts[1:]))
        if p.name:
            candidates.append(repo_dir / "assets" / "models" / p.name)
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())
        return str(candidates[0].resolve())

    @classmethod
    def _ensure_weights_file(cls, weights: str, base_dir: Path) -> str:
        resolved = Path(cls._resolve_weights_path(weights, base_dir))
        if resolved.exists():
            return str(resolved)

        if resolved.suffix != ".pt":
            return str(resolved)

        if not cls._can_auto_download_yolo_weights(resolved.name):
            raise FileNotFoundError(
                f"YOLO 权重不存在: {resolved}。这是本地训练/自定义权重，"
                "不会从 Ultralytics 自动下载；请检查 mask_weights 是否写错，"
                "或把权重文件放到 assets/models/ 下。"
            )

        logger.info(f"YOLO 权重不存在，准备自动下载: {resolved}")
        cls._download_yolo_weights(resolved)
        return str(resolved)

    # Ultralytics 官方 YOLOv8 stock 权重命名（如 yolov8l-seg.pt）；自训练/
    # 微调权重（如 yolo26s-seg_finetuned.pt、best.pt）不匹配，不会自动下载。
    _OFFICIAL_YOLO_WEIGHTS_RE = re.compile(r"^yolov8[nsmlx](-seg|-cls|-pose|-obb)?\.pt$")

    @classmethod
    def _can_auto_download_yolo_weights(cls, weights_name: str) -> bool:
        return bool(cls._OFFICIAL_YOLO_WEIGHTS_RE.match(weights_name))

    @staticmethod
    def _download_yolo_weights(target: Path) -> None:
        model_name = target.name
        base_url = os.environ.get(
            "YOLO_WEIGHTS_BASE_URL",
            "https://github.com/ultralytics/assets/releases/download/v8.3.0",
        ).rstrip("/")
        urls = [
            f"{base_url}/{model_name}",
            f"https://github.com/ultralytics/assets/releases/download/v8.2.0/{model_name}",
            f"https://github.com/ultralytics/assets/releases/download/v8.1.0/{model_name}",
            f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}",
        ]

        target.parent.mkdir(parents=True, exist_ok=True)
        last_error = None
        for url in dict.fromkeys(urls):
            tmp_path = None
            try:
                logger.info(f"下载 YOLO 权重: {url}")
                with urllib.request.urlopen(url, timeout=60) as response:
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        dir=str(target.parent),
                        suffix=".part",
                    ) as tmp:
                        shutil.copyfileobj(response, tmp)
                        tmp_path = Path(tmp.name)
                tmp_path.replace(target)
                logger.info(f"YOLO 权重已保存: {target}")
                return
            except (OSError, urllib.error.URLError) as e:
                last_error = e
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink()

        raise RuntimeError(
            f"无法自动下载 YOLO 权重 {model_name}。"
            f"请检查网络，或手动放到 {target}。最后错误: {last_error}"
        )

    def available_class_names(self) -> set[str]:
        """返回当前权重中声明的类别名，并应用别名后的业务类别名。"""
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load()")
        names = self.model.names
        raw_names = names.values() if isinstance(names, dict) else names
        return {self.normalize_class_name(str(name)) for name in raw_names}

    def raw_class_names(self) -> set[str]:
        """返回当前权重中声明的原始类别名。"""
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load()")
        names = self.model.names
        return set(names.values() if isinstance(names, dict) else names)

    def normalize_class_name(self, class_name: str) -> str:
        return self.class_aliases.get(class_name, class_name)

    def _predict(self, model: object, **kwargs: object) -> object:
        """在配置的 GPU 设备上执行 YOLO 推理。"""

        return model.predict(device=self.device, **kwargs)

    @staticmethod
    def _file_sha256(path: str) -> str:
        """Return a short SHA-256 for debug comparison across processes."""
        digest = hashlib.sha256()
        with open(path, "rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()[:16]

    def _debug_fresh_model_probe(self, source: object) -> None:
        """Repeat a failed prediction with a new YOLO instance in debug mode."""
        if self.loaded_weights_path is None:
            logger.warning("YOLO 同进程复测跳过：未记录已加载权重路径")
            return

        from ultralytics import YOLO

        logger.info("YOLO 同进程复测：使用新模型实例和独立预测脚本参数")
        probe_model = YOLO(self.loaded_weights_path)
        probe_results = self._predict(
            probe_model,
            source=source,
            imgsz=640,
            conf=self.conf_threshold,
            iou=0.7,
            classes=None,
            visualize=False,
            save=False,
            show=False,
            verbose=False,
            task="segment",
        )
        probe_boxes = [box for result in probe_results for box in result.boxes]
        logger.info(f"YOLO 同进程复测原始框数量: {len(probe_boxes)}")
        for box in probe_boxes:
            class_id = int(box.cls[0])
            logger.info(
                "YOLO 同进程复测候选: "
                f"class={probe_model.names[class_id]}, "
                f"conf={float(box.conf[0]):.4f}, "
                f"bbox={box.xyxy[0].tolist()}"
            )

    def detect_raw(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray],
        allowed_classes: set[str],
        *,
        depth_required: bool,
        debug: bool = False,
        source_path: Optional[str] = None,
    ) -> List[RawYoloDetection]:
        """检测白名单类别，保留像素坐标以便由上层完成三维定位。"""
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load()")

        detections: List[RawYoloDetection] = []
        # Keep runtime inference consistent with the imgsz used by GroundedSAM-AutoYOLO training exports.
        source = source_path or rgb
        if source_path:
            logger.info(f"YOLO 调试推理使用已保存相机帧: {source_path}")
        results = self._predict(
            self.model,
            source=source,
            imgsz=640,
            conf=self.conf_threshold,
            verbose=debug,
            task="segment",
        )
        raw_box_count = sum(len(result.boxes) for result in results)
        logger.info(f"YOLO 原始框数量: {raw_box_count}")
        if debug and raw_box_count == 0:
            self._debug_fresh_model_probe(source)
        for result in results:
            for index, box in enumerate(result.boxes):
                raw_class_name = str(self.model.names[int(box.cls[0])])
                class_name = self.normalize_class_name(raw_class_name)
                if class_name not in allowed_classes:
                    if debug:
                        logger.debug(
                            f"过滤非物料类别: raw={raw_class_name}, "
                            f"normalized={class_name}"
                        )
                    continue
                if debug and raw_class_name != class_name:
                    logger.debug(
                        f"YOLO 类别别名映射: raw={raw_class_name} -> {class_name}"
                    )

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if self.use_segmentation and result.masks is not None and index < len(result.masks):
                    pixel, contour, mask_width_px = self._centroid_from_mask(
                        result.masks.xy[index], rgb.shape, (x1, y1, x2, y2)
                    )
                else:
                    pixel, _, contour = self._centroid_from_shape(rgb, (x1, y1, x2, y2))
                    mask_width_px = None

                depth_m = self._get_robust_depth(
                    depth,
                    *pixel,
                    radius=self.depth_median_radius,
                )
                if depth_required and depth_m is None:
                    logger.warning(
                        f"跳过深度无效候选: class={class_name}, pixel={pixel}"
                    )
                    continue
                detections.append(
                    RawYoloDetection(
                        class_name=class_name,
                        confidence=float(box.conf[0]),
                        pixel=pixel,
                        mask_width_px=mask_width_px,
                        depth_m=depth_m,
                        contour=contour,
                    )
                )

        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections

    @staticmethod
    def _get_robust_depth(
        depth: np.ndarray,
        u: int,
        v: int,
        radius: int = 5,
    ) -> Optional[float]:
        """获取鲁棒的深度值（邻域中位数）"""
        if depth is None:
            return None
        h, w = depth.shape
        u0 = int(np.clip(u, 0, w - 1))
        v0 = int(np.clip(v, 0, h - 1))
        patch = depth[
            max(0, v0 - radius) : min(h, v0 + radius + 1),
            max(0, u0 - radius) : min(w, u0 + radius + 1),
        ]
        valid = patch[(patch > 0.05) & (patch < 2.0)]
        return float(np.median(valid)) if valid.size > 0 else None

    @staticmethod
    def _centroid_from_mask(
        mask_xy: np.ndarray,
        img_shape: tuple,
        bbox: tuple,
    ) -> Tuple[tuple, Optional[np.ndarray], Optional[float]]:
        """从 YOLO 分割掩码计算质心"""
        x1, y1, x2, y2 = bbox
        bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        if mask_xy is None or len(mask_xy) < 3:
            return bbox_center, None, None

        contour = mask_xy.astype(np.int32)
        h, w = img_shape[:2]
        filled = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(filled, [contour], 255)

        M = cv2.moments(filled)
        if M["m00"] < 1:
            return bbox_center, contour, None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        row = filled[cy, :]
        xs = np.where(row > 0)[0]
        mask_width_px = float(xs[-1] - xs[0]) if len(xs) >= 2 else None

        return (cx, cy), contour, mask_width_px

    @staticmethod
    def _centroid_from_shape(
        image: np.ndarray,
        bbox: tuple,
    ) -> Tuple[tuple, None, Optional[np.ndarray]]:
        """从形状轮廓计算质心（YOLO 无分割时的 fallback）"""
        x1, y1, x2, y2 = bbox
        h_img, w_img = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        if x2 <= x1 or y2 <= y1:
            return bbox_center, None, None

        roi = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return bbox_center, None, None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 50:
            return bbox_center, None, None

        M = cv2.moments(largest)
        if M["m00"] < 1:
            return bbox_center, None, None

        cx = int(M["m10"] / M["m00"]) + x1
        cy = int(M["m01"] / M["m00"]) + y1
        contour_global = largest + np.array([x1, y1])

        return (cx, cy), None, contour_global


# ══════════════════════════════════════════════════════════════════
# FoundationPose 位姿估计模型 - 本地 GPU 推理
# ══════════════════════════════════════════════════════════════════


class FoundationPoseModel:
    """FoundationPose 6DoF 位姿估计模型（本地推理，无需 PC 服务器）

    分割 mask 由 Perception 的共享 YOLO-seg 提供（见 :meth:`register_pose`），
    本类只负责结合 mask + 深度图 + CAD 模型估计 6DoF 位姿。
    """

    def __init__(
        self,
        mesh_path: str,
        fp_root: str = "/home/galbot/FoundationPose",
        target_class: str = "bottle",
        input_scale: float = 1.0,
    ):
        p = Path(mesh_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent / p
        self.mesh_path = str(p.resolve())
        self.fp_root = fp_root
        self.target_class = target_class
        self.input_scale = input_scale
        self._fp = None

    def load(self) -> None:
        """按需加载该类别的 FoundationPose GPU 模型（幂等）。"""

        import sys
        import trimesh

        if self._fp is not None:
            logger.info(
                f"FoundationPose 模型已加载，跳过重复加载: {self.target_class}"
            )
            return
        _ensure_warp_cuda_ready()
        if self.fp_root not in sys.path:
            sys.path.insert(0, self.fp_root)
        from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor

        loaded = trimesh.load(self.mesh_path)
        if isinstance(loaded, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
        elif isinstance(loaded, trimesh.PointCloud):
            mesh = loaded.convex_hull
        else:
            mesh = loaded
        self._ensure_mesh_texture(mesh)

        self._fp = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=ScorePredictor(),
            refiner=PoseRefinePredictor(),
            debug=0,
            debug_dir="/tmp/fp_debug",
        )
        logger.info(f"FoundationPose 本地模型加载完成 (mesh: {self.mesh_path})")

    @property
    def is_loaded(self) -> bool:
        """是否已加载该类别的 FoundationPose GPU 模型。"""

        return self._fp is not None

    @staticmethod
    def _ensure_mesh_texture(mesh) -> None:
        """部分 CAD 导出（如 Blender 只给纯色材质、无贴图）的 mesh 会让
        ``mesh.visual.material.image`` 为 None；FoundationPose 的
        ``make_mesh_tensors`` 不判空直接 ``.convert('RGB')``，因此这里用材质的
        漫反射色补一张纯色贴图，其余顶点/UV 数据不变。
        """
        material = getattr(getattr(mesh, "visual", None), "material", None)
        if material is None or getattr(material, "image", None) is not None:
            return
        import numpy as _np
        from PIL import Image

        diffuse = getattr(material, "diffuse", None)
        color = (
            _np.asarray(diffuse[:3], dtype=_np.uint8)
            if diffuse is not None
            else _np.array([200, 200, 200], dtype=_np.uint8)
        )
        material.image = Image.fromarray(
            _np.tile(color, (4, 4, 1)).astype(_np.uint8), mode="RGB"
        )
        logger.warning(
            "mesh 材质无贴图，已用漫反射色补一张纯色贴图: "
            f"color={color.tolist()}"
        )

    def register_pose(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        camera_intrinsics: Tuple[float, float, float, float],
        mask: np.ndarray,
        debug: bool = False,
    ) -> Tuple[np.ndarray, Optional[float]]:
        """用外部已给定的分割 mask 直接跑 FoundationPose 位姿估计，跳过内部 YOLO-seg。

        用于多类别主流程：共用一次分割结果选出唯一目标后，只对该目标调用
        本方法，避免每个类别各自重复执行分割和 GPU 位姿估计。

        Returns:
            (object_pose_cam, mask_width_px)
        """
        if self._fp is None:
            raise RuntimeError("模型未加载，请先调用 load()")

        fx, fy, cx, cy = camera_intrinsics
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        mask_width_px = self._get_mask_width(mask)

        # 默认直接使用原始 RGB-D；仅在显存受限时按配置缩放输入。
        rgb_fp = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb_s, depth_s, mask_s, K_s = self._resize_inputs(rgb_fp, depth, mask, K)

        import torch
        torch.cuda.empty_cache()
        pose_cam = self._fp.register(
            K=K_s, rgb=rgb_s, depth=depth_s, ob_mask=mask_s, iteration=5
        ).astype(np.float64)

        if debug:
            logger.debug(f"本地 FP 位姿(相机系): {pose_cam[:3, 3].tolist()}")
            self._save_debug_result(rgb, mask, pose_cam, K)

        return pose_cam, mask_width_px

    @staticmethod
    def _get_mask_width(mask: np.ndarray) -> Optional[float]:
        """用 mask 质心所在行估算目标像素宽度。"""
        M = cv2.moments(mask)
        if M["m00"] < 1:
            return None
        cy = int(M["m01"] / M["m00"])
        xs = np.where(mask[cy, :] > 0)[0]
        return float(xs[-1] - xs[0]) if len(xs) >= 2 else None

    def _resize_inputs(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """等比缩放 RGB/depth/mask，同步调整相机内参 K。"""
        s = self.input_scale
        if s == 1.0:
            return rgb, depth, mask, K
        h, w = rgb.shape[:2]
        nh, nw = int(h * s), int(w * s)
        rgb_s   = cv2.resize(rgb,   (nw, nh), interpolation=cv2.INTER_LINEAR)
        depth_s = cv2.resize(depth, (nw, nh), interpolation=cv2.INTER_NEAREST)
        mask_s  = cv2.resize(mask,  (nw, nh), interpolation=cv2.INTER_NEAREST)
        K_s = K.copy()
        K_s[0] *= s  # fx, cx
        K_s[1] *= s  # fy, cy
        return rgb_s, depth_s, mask_s, K_s

    @staticmethod
    def _project_point(K: np.ndarray, pt: np.ndarray) -> Optional[tuple]:
        """将相机系 3D 点投影到像素坐标。"""
        if pt[2] < 0.001:
            return None
        p = K @ pt
        return int(p[0] / p[2]), int(p[1] / p[2])

    @classmethod
    def _draw_pose_axes(
        cls,
        image: np.ndarray,
        K: np.ndarray,
        pose_cam: np.ndarray,
        axis_len: float = 0.05,
    ) -> None:
        """在图像上绘制相机系下的 6D pose 坐标轴。"""
        origin = pose_cam[:3, 3]
        origin_px = cls._project_point(K, origin)
        if origin_px is None:
            return

        cv2.circle(image, origin_px, 5, (0, 255, 255), -1)
        for col, (color, label) in enumerate(
            zip(
                [(0, 0, 255), (0, 255, 0), (255, 0, 0)],
                ["X", "Y", "Z"],
            )
        ):
            tip = cls._project_point(K, origin + pose_cam[:3, col] * axis_len)
            if tip is None:
                continue
            cv2.arrowedLine(image, origin_px, tip, color, 2, tipLength=0.2)
            cv2.putText(
                image,
                label,
                tip,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )


    def _save_debug_result(
        self,
        bgr: np.ndarray,
        mask: np.ndarray,
        pose_cam: np.ndarray,
        K: np.ndarray,
    ) -> None:
        """保存 FoundationPose 的 mask 和物体位姿坐标轴可视化。"""
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "output"
        )
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        vis = bgr.copy()
        vis[mask > 0] = (vis[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(
            np.uint8
        )
        self._draw_pose_axes(vis, K, pose_cam)

        save_path = os.path.join(output_dir, f"foundationpose_pose_{timestamp}.png")
        cv2.imwrite(save_path, vis)
        logger.info(f"FoundationPose 位姿可视化已保存: {save_path}")


# ══════════════════════════════════════════════════════════════════
# 感知主类 - 管理图像输入输出和模型调用
# ══════════════════════════════════════════════════════════════════


class Perception:
    """视觉感知类

    负责：
    1. 从机器人获取 RGB/深度图像（头部左目 RGB + 双目推理深度）
    2. 读取相机内参和 base_link <- camera 变换
    3. 多类别 FoundationPose 链路：共享 YOLO-seg 选目标 → 按类别配准 6DoF 位姿

    Attributes:
        config: 感知配置
        detection_model: 独立 example 注入的 YOLO 模型；主流程为 None
    """

    def __init__(
        self,
        config: Optional[PerceptionConfig] = None,
        detection_model: Optional[YoloDetection] = None,
        debug_mode: bool = False,
    ):
        """
        Args:
            config: 感知配置，None 则使用默认配置
            detection_model: 可注入独立的
                YoloDetection 复用相机读取能力；主流程不传，创建 FoundationPose
                多类别感知链路。
            debug_mode: 是否启用调试模式（可视化、详细日志等）
        """
        self.config = config or PerceptionConfig()
        self.debug_mode = debug_mode
        # 取图/内参/相机外参统一走头部相机链路
        self._head_camera = HeadCameraPerception(
            model_ready_wait_s=self.config.head_model_ready_wait_s,
            inference_timeout_s=self.config.head_inference_timeout_s,
        )
        self._debug_rgb_path: Optional[Path] = None
        # cv2.imshow() 在 SSH / 无桌面机器人上会由 Qt 直接 abort，无法依靠
        # cv2.error 捕获。默认保存调试图；只有显式允许时才打开窗口。
        self._debug_gui_available = (
            os.environ.get("PERCEPTION_DEBUG_GUI") == "1"
            and bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        )

        self.detection_model: Optional[YoloDetection] = None
        self._mask_model: Optional[YoloDetection] = None
        self._foundationpose_models: dict[str, FoundationPoseModel] = {}

        if detection_model is not None:
            self.detection_model = detection_model
            return

        self._mask_model = YoloDetection(
            use_segmentation=True,
            conf_threshold=self.config.mask_conf_threshold,
            yolo_weights=self.config.mask_weights,
            depth_median_radius=self.config.depth_median_radius,
            class_aliases=self.config.yolo_class_aliases,
        )
        for class_name, object_config in self.config.foundationpose_objects.items():
            self._foundationpose_models[class_name] = FoundationPoseModel(
                mesh_path=object_config.mesh_path,
                fp_root=self.config.fp_root,
                target_class=class_name,
                input_scale=self.config.fp_input_scale,
            )

    def load_model(self) -> None:
        """加载感知入口。

        多类别主流程这里只加载一次共享 YOLO 分割模型；YOLO 选出目标类别后，
        再按需加载并复用对应的 FoundationPose GPU 模型。显式注入的独立模型
        仍按原有方式完整加载。
        """

        self._head_camera.init_depth_module()
        if self.detection_model is not None:
            self.detection_model.load()
            return
        if self._mask_model is None:
            raise RuntimeError("FoundationPose mask 模型未初始化")
        self._mask_model.load()
        logger.info(
            "共享 YOLO 分割模型已就绪；FoundationPose 将在选定目标后按类别加载"
        )

    @staticmethod
    def _field(message, name: str, default=None):
        if message is None:
            return default
        if isinstance(message, dict):
            return message.get(name, default)
        return getattr(message, name, default)

    @classmethod
    def _timestamp_ns(cls, message) -> int:
        header = cls._field(message, "header", {})
        return int(cls._field(header, "timestamp_ns", 0) or 0)

    @classmethod
    def decode_rgb_image(cls, compressed_image) -> np.ndarray:
        """解码 RGB 图像"""
        nparr = np.frombuffer(cls._field(compressed_image, "data"), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("RGB 图像解码失败")
        return img

    def _show_debug_images(
        self, rgb: Optional[np.ndarray], depth: Optional[np.ndarray]
    ) -> Optional[Path]:
        """保存 RGB 与深度伪彩图调试帧；启用 GUI 时额外显示窗口。"""
        if not self.debug_mode or rgb is None:
            return None

        output_dir = Path(__file__).resolve().parent.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = output_dir / "camera_image_1.png"
        if not cv2.imwrite(str(rgb_path), rgb):
            logger.warning(f"无法保存调试相机帧: {rgb_path}")
            return None

        depth_vis = None
        if depth is not None:
            depth_f = np.asarray(depth, dtype=np.float32)
            valid = depth_f[np.isfinite(depth_f) & (depth_f > 0)]
            if valid.size > 0:
                vmin, vmax = np.percentile(valid, [1, 99])
                normalized = np.clip(
                    (depth_f - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0
                )
            else:
                normalized = np.zeros_like(depth_f, dtype=np.float32)
            depth_vis = cv2.applyColorMap(
                (normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO
            )
            depth_path = output_dir / "camera_depth_1.png"
            if not cv2.imwrite(str(depth_path), depth_vis):
                logger.warning(f"无法保存调试深度伪彩图: {depth_path}")

        if self._debug_gui_available:
            try:
                cv2.imshow("camera_image_1", rgb)
                if depth_vis is not None:
                    cv2.imshow("camera_depth_1", depth_vis)
                cv2.waitKey(1)
            except cv2.error as error:
                self._debug_gui_available = False
                logger.warning(f"无法创建调试相机窗口，改为保存图像: {error}")

        return rgb_path

    def get_images(
        self, robot: GalbotRobot
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        获取一帧 RGB + 深度图

        Args:
            robot: GalbotRobot 实例

        Returns:
            (rgb_bgr, depth_m) 失败时对应值为 None
        """
        # RGB 来自头部左目，深度由 FOUNDATION_STEREO 双目推理生成
        rgb, depth = self._head_camera.get_images(robot)
        self._debug_rgb_path = self._show_debug_images(rgb, depth)
        return rgb, depth

    def get_camera_intrinsics(
        self, robot: GalbotRobot
    ) -> Tuple[float, float, float, float]:
        """
        读取相机内参

        Args:
            robot: GalbotRobot 实例

        Returns:
            (fx, fy, cx, cy) 读取失败时抛出错误
        """
        return self._head_camera.get_camera_intrinsics(robot)

    def camera_transform(self, robot: GalbotRobot) -> np.ndarray:
        """返回采集帧时刻的 ``base_link <- camera`` 变换。"""

        return self._head_camera.camera_transform(robot)

    def detect_foundationpose_candidates(
        self, robot: GalbotRobot
    ) -> List[FoundationPoseCandidate]:
        """拍摄一帧并返回按检测置信度选出的 FoundationPose 候选。

        分割模型先在全部 ``foundationpose_objects`` 类别中找实例；按
        检测置信度选出一个，再仅对该实例执行一次较慢的
        FoundationPose 配准。有效空场返回空列表，链路故障抛出明确异常。
        """

        if self.detection_model is not None or self._mask_model is None:
            raise RuntimeError("当前 Perception 不是 FoundationPose 主流程实例")
        expected_classes = set(self._foundationpose_models)
        available_classes = self._mask_model.available_class_names()
        missing_classes = expected_classes - available_classes
        if missing_classes:
            raise ValueError(
                "分割权重缺少 FoundationPose 物料类别: "
                f"{sorted(missing_classes)}；"
                f"配置类别={sorted(expected_classes)}；"
                f"权重类别={sorted(available_classes)}；"
                f"权重原始类别={sorted(self._mask_model.raw_class_names())}；"
                f"权重路径={self._mask_model.loaded_weights_path}"
            )

        rgb, depth = self.get_images(robot)
        if rgb is None:
            raise PerceptionUnavailableError("未获取到 RGB 帧")
        if depth is None:
            raise PerceptionUnavailableError("未获取到深度帧，拒绝估计 6DoF 位姿")
        try:
            intrinsics = self.get_camera_intrinsics(robot)
            transform_base_camera = self.camera_transform(robot)
        except RuntimeError as error:
            raise PerceptionUnavailableError(str(error)) from error

        self._last_rgb = rgb
        self._last_transform_base_camera = transform_base_camera
        fx, fy, cx, cy = intrinsics
        self._last_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        source_path = (
            str(self._debug_rgb_path)
            if self.debug_mode and self._debug_rgb_path is not None
            else None
        )
        raw_detections = self._mask_model.detect_raw(
            rgb,
            depth,
            expected_classes,
            depth_required=True,
            debug=self.debug_mode,
            source_path=source_path,
        )
        if not raw_detections:
            logger.info("FoundationPose 分割阶段未发现候选")
            return []

        ranked_candidates = []
        for raw in raw_detections:
            try:
                point_cam = pixel_to_3d(
                    *raw.pixel, raw.depth_m, *intrinsics
                )
            except ValueError as error:
                logger.warning(
                    f"跳过无效 YOLO 候选: class={raw.class_name}, reason={error}"
                )
                continue
            point_base = (
                transform_base_camera @ np.append(point_cam, 1.0)
            )[:3]
            key = -raw.confidence
            ranked_candidates.append((key, raw, point_base))

        if not ranked_candidates:
            raise PerceptionUnavailableError(
                "YOLO 找到候选，但候选深度均无效，无法估计物体位置"
            )

        ranked_candidates.sort(key=lambda item: item[0])
        for rank, (_, raw, point_base) in enumerate(ranked_candidates, start=1):
            logger.info(
                "YOLO 感知候选排序: "
                f"rank={rank}, class={raw.class_name}, "
                f"conf={raw.confidence:.2f}, "
                f"base={np.round(point_base, 3).tolist()}"
            )

        _, chosen, chosen_point_base = ranked_candidates[0]
        logger.info(
            "YOLO 已选定 FoundationPose 目标: "
            f"class={chosen.class_name}, conf={chosen.confidence:.2f}, "
            f"base={np.round(chosen_point_base, 3).tolist()}"
        )
        if chosen.contour is None:
            raise PerceptionUnavailableError(
                f"分割模型未返回 {chosen.class_name} 的 mask 轮廓"
            )

        mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [chosen.contour], 255)
        foundationpose_model = self._foundationpose_models[chosen.class_name]
        if not foundationpose_model.is_loaded:
            logger.info(
                f"首次选中 {chosen.class_name}，按需加载对应 FoundationPose 模型"
            )
            foundationpose_model.load()
        else:
            logger.info(
                f"复用已加载的 FoundationPose 模型: {chosen.class_name}"
            )
        pose_cam, mask_width_px = foundationpose_model.register_pose(
            rgb, depth, intrinsics, mask, debug=self.debug_mode
        )
        pose_base = transform_base_camera @ pose_cam
        depth_m = float(pose_cam[2, 3])
        obj_width_m = (
            mask_width_px * depth_m / fx
            if mask_width_px is not None
            else None
        )
        candidate = FoundationPoseCandidate(
            class_name=chosen.class_name,
            confidence=chosen.confidence,
            position_base=pose_base[:3, 3],
            pose_base=pose_base,
            obj_width_m=obj_width_m,
        )
        logger.info(
            "FoundationPose 候选: "
            f"class={candidate.class_name}, conf={candidate.confidence:.2f}, "
            f"base={np.round(candidate.position_base, 3).tolist()}"
        )
        return [candidate]



# ══════════════════════════════════════════════════════════════════
# 头部相机采集 - RGB 来自左目，深度由 FOUNDATION_STEREO 双目推理生成
# ══════════════════════════════════════════════════════════════════


class HeadCameraPerception:
    """头部双目相机采集（SDK 1.9.0）。

    头部只有左右两个 RGB 相机（``HEAD_LEFT_CAMERA`` / ``HEAD_RIGHT_CAMERA``），
    SDK 没有对应的头部深度传感器；深度图需要通过 ``GalbotPerception`` 的
    ``FOUNDATION_STEREO`` 模块在机器人端做一次双目推理得到，结果通过
    ``DetectionResult.instance_mask`` 字段返回。

    使用前提：``robot.init()`` 的传感器集合需包含 ``SensorType.HEAD_LEFT_CAMERA``。

    典型用法::

        head = HeadCameraPerception()
        head.init_depth_module()      # 双目深度模型加载约需 10s，只需一次
        rgb, depth = head.get_images(robot)
    """

    RGB_SENSOR = SensorType.HEAD_LEFT_CAMERA

    def __init__(
        self,
        model_ready_wait_s: float = 12.0,
        inference_timeout_s: float = 6.0,
    ):
        """
        Args:
            model_ready_wait_s: perception.init() 后等待双目模型加载的时间
            inference_timeout_s: 单次 run_once 推理结果的等待超时
        """
        self.model_ready_wait_s = model_ready_wait_s
        self.inference_timeout_s = inference_timeout_s
        self._perception = None
        self._module = None
        self._last_rgb_timestamp_ns = 0
        self._last_depth_timestamp_ns = 0
        self._last_rgb_frame_id = ""

    def init_depth_module(self) -> None:
        """初始化 FOUNDATION_STEREO 双目深度模块并等待模型就绪（幂等）。"""
        if self._perception is not None:
            return
        from galbot_sdk.g1 import GalbotPerception, MachineType, PerceptionModule

        if hasattr(GalbotPerception, "get_instance"):
            perception = GalbotPerception.get_instance(MachineType.G1)
        else:
            perception = GalbotPerception()
        if not perception.init({PerceptionModule.FOUNDATION_STEREO}):
            raise PerceptionUnavailableError("FOUNDATION_STEREO 感知模块初始化失败")
        logger.info(
            f"等待头部双目深度模型加载（约 {self.model_ready_wait_s:.0f}s）..."
        )
        time.sleep(self.model_ready_wait_s)
        self._perception = perception
        self._module = PerceptionModule.FOUNDATION_STEREO
        logger.info("FOUNDATION_STEREO 双目深度模块已就绪")

    def get_rgb(self, robot: GalbotRobot) -> Optional[np.ndarray]:
        """获取头部左目相机一帧 RGB 图（BGR 排列），失败返回 None。"""
        rgb_data = robot.get_rgb_data(self.RGB_SENSOR)
        if not rgb_data:
            logger.warning(
                "头部相机 RGB 获取失败，请确认 robot.init() 已启用 HEAD_LEFT_CAMERA"
            )
            return None
        self._last_rgb_timestamp_ns = Perception._timestamp_ns(rgb_data)
        header = Perception._field(rgb_data, "header", {})
        self._last_rgb_frame_id = str(
            Perception._field(header, "frame_id", "") or ""
        )
        try:
            return Perception.decode_rgb_image(rgb_data)
        except ValueError as error:
            logger.error(f"头部相机 RGB 解码失败: {error}")
            return None

    def get_depth(self) -> Optional[np.ndarray]:
        """触发一次双目推理并返回头部深度图（float32，单位米），失败返回 None。"""
        if self._perception is None:
            self.init_depth_module()
        if not self._perception.run_once(self._module):
            logger.error("头部双目深度 run_once 指令发送失败")
            return None
        if not self._perception.wait_for_new_result(
            self._module, self.inference_timeout_s
        ):
            logger.error(f"等待头部双目深度结果超时（{self.inference_timeout_s}s）")
            return None
        ok, result = self._perception.get_latest_result(self._module)
        if not ok:
            logger.error("头部双目深度 get_latest_result 失败")
            return None
        depth = result.instance_mask
        if depth is None:
            logger.error("头部双目深度结果为空（instance_mask 为空）")
            return None
        self._last_depth_timestamp_ns = int(result.timestamp_ns)

        depth = np.asarray(depth)
        # 深度编码 16UC1 时为毫米，32FC1 时为米，统一转换成米
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0
        else:
            depth = depth.astype(np.float32)
        return depth

    def get_images(
        self, robot: GalbotRobot
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """获取头部相机的一帧 RGB + 深度图，失败的一路为 None。

        注意：RGB 为相机流最新帧，深度为按需触发的双目推理结果，两者时间戳
        接近但非硬同步；对时间敏感时用 ``last_frame_timestamps_ns`` 自行判断。
        """
        rgb = self.get_rgb(robot)
        depth = self.get_depth()
        return rgb, depth

    @property
    def last_frame_timestamps_ns(self) -> Tuple[int, int]:
        """最近一次采集的 (rgb_timestamp_ns, depth_timestamp_ns)。"""
        return self._last_rgb_timestamp_ns, self._last_depth_timestamp_ns

    def get_camera_intrinsics(
        self, robot: GalbotRobot
    ) -> Tuple[float, float, float, float]:
        """读取头部左目相机内参 (fx, fy, cx, cy)，失败抛出异常。"""
        cam_info = robot.get_camera_intrinsic(self.RGB_SENSOR)
        if cam_info and "K" in cam_info:
            K = cam_info["K"]
            return K[0], K[4], K[2], K[5]
        raise PerceptionUnavailableError("无法读取头部相机内参，请检查相机是否正常工作")

    def camera_transform(self, robot: GalbotRobot) -> np.ndarray:
        """返回 ``base_link <- 头部左目相机`` 的 4x4 变换矩阵。

        头部随颈部关节转动，优先用 RGB 帧的 frame_id 在采集时刻查 TF；
        帧头缺少 frame_id 或 TF 查询失败时退回传感器静态外参。
        """
        if self._last_rgb_frame_id:
            pose, tf_status = robot.get_transform(
                "base_link",
                self._last_rgb_frame_id,
                self._last_rgb_timestamp_ns,
                100,
            )
            if pose:
                return pose_to_matrix(pose)
            logger.warning(
                f"TF 查询失败（frame_id={self._last_rgb_frame_id}）: "
                f"status={tf_status}；退回头部相机静态外参"
            )
        pose, extrinsic_status = robot.get_sensor_extrinsic(
            self.RGB_SENSOR, "base_link"
        )
        if not pose:
            raise PerceptionUnavailableError(
                "无法获取 base_link <- HEAD_LEFT_CAMERA 的外参: "
                f"status={extrinsic_status}"
            )
        return pose_to_matrix(pose)
