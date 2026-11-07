"""完整循环抓取流程的兼容入口。

推荐从项目根目录运行 ``examples/cyclic_pick_place.py``。本文件保留命令
兼容性，并把所有任务执行转交给公共流程。
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

# 保持 SDK 在数值与机器学习依赖之前加载。
import galbot_sdk.g1  # noqa: F401

from config import PipelineConfig, cfg
from logger import logger
from task_workflows import run_cyclic_pick_place, run_single_pick_place


POSE_METAVAR = ("X", "Y", "Z", "QX", "QY", "QZ", "QW")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Galbot 连续感知、抓取、放置与返航任务"
    )
    parser.add_argument(
        "--enable_navigation",
        action="store_true",
        help="启用完整循环所需的底盘导航",
    )
    parser.add_argument(
        "--single_grasp_no_navigation",
        action="store_true",
        help="兼容参数：改为执行单物体抓取、本地放置和归位",
    )
    parser.add_argument(
        "--perception_retries",
        type=int,
        help="感知链路不可用时的最大尝试次数",
    )
    parser.add_argument(
        "--empty_confirmations",
        type=int,
        help="判定抓取区为空所需的连续有效空场次数",
    )
    parser.add_argument(
        "--max_cycles",
        type=int,
        help="成功放置上限；达到后进入 HOLD",
    )
    parser.add_argument(
        "--perception_retry_interval_s",
        type=float,
        help="两次感知之间的等待时间（秒）",
    )
    for name, help_text in (
        ("origin_pose", "任务正常结束时返回的原点导航位姿"),
        ("pickup_pose", "抓取点导航位姿"),
        ("place_pose", "放置点导航位姿"),
    ):
        parser.add_argument(
            f"--{name}",
            nargs=7,
            type=float,
            metavar=POSE_METAVAR,
            help=f"{help_text} [x, y, z, qx, qy, qz, qw]",
        )
    return parser


def apply_args(config: PipelineConfig, args: argparse.Namespace) -> None:
    """只覆盖命令行显式传入的配置。"""

    for name in ("origin_pose", "pickup_pose", "place_pose"):
        value = getattr(args, name)
        if value is not None:
            setattr(config.navigation, name, value)

    if args.enable_navigation:
        config.navigation.enable_navigation = True
    for name in (
        "perception_retries",
        "perception_retry_interval_s",
        "empty_confirmations",
        "max_cycles",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(config.task, name, value)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    apply_args(cfg, args)

    try:
        if args.single_grasp_no_navigation:
            logger.warning(
                "--single_grasp_no_navigation 为兼容参数；"
                "建议改用 python3 examples/single_pick_place.py"
            )
            cfg.navigation.enable_navigation = False
            result = run_single_pick_place(cfg)
        else:
            result = run_cyclic_pick_place(cfg)
    except KeyboardInterrupt:
        logger.warning("任务已被用户中断")
        return 130
    except Exception as error:
        logger.exception(f"任务启动或执行异常: {error}")
        return 1

    return 0 if result.success else 2


if __name__ == "__main__":
    sys.exit(main())
