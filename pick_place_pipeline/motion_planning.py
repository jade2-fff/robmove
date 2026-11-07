"""
运动规划模块（底层控制层）

核心职责：封装 GalbotMotion API，提供到达指定位姿的运动执行能力

设计原则：
- 极简主义：只做底层能做的事，不做上层业务逻辑
- 直接返回 SDK 原生 MotionStatus，不做额外封装
- 原子操作：每个函数只做一件事，不做组合

提供的原子能力：
1. get_current_pose()  - 获取当前末端位姿（FK 反馈）
2. move_to_pose()       - 到达指定位姿，返回 MotionStatus
3. print_motion_status() - 打印详细的运动状态说明
"""

import time
from typing import Optional, List

import galbot_sdk.g1 as gm
from galbot_sdk.g1 import GalbotMotion, MotionStatus, GalbotRobot

from logger import logger


def print_motion_status(status: MotionStatus, operation: str = "操作") -> None:
    """
    打印详细的运动状态说明（使用 logger）

    Args:
        status: MotionStatus 枚举值
        operation: 操作名称，用于日志上下文
    """
    status_map = {
        MotionStatus.SUCCESS: (logger.info, "执行成功"),
        MotionStatus.TIMEOUT: (logger.warning, "执行超时"),
        MotionStatus.FAULT: (logger.error, "发生故障，无法继续执行"),
        MotionStatus.INVALID_INPUT: (logger.error, "输入参数不符合要求"),
        MotionStatus.INIT_FAILED: (logger.error, "内部通信组件创建失败"),
        MotionStatus.IN_PROGRESS: (logger.info, "运动中，尚未到达位置"),
        MotionStatus.STOPPED_UNREACHED: (logger.warning, "已停止，但未到达目标"),
        MotionStatus.DATA_FETCH_FAILED: (logger.error, "数据获取失败"),
        MotionStatus.PUBLISH_FAIL: (logger.error, "数据发送失败"),
        MotionStatus.COMM_DISCONNECTED: (logger.error, "连接断开"),
    }

    log_func, message = status_map.get(status, (logger.warning, f"未知状态: {status}"))
    log_func(f"{operation}: {status.name} - {message}")


class MotionPlanner:
    """底层运动执行器

    设计原则：只做原子操作，不做业务组合

    原子 API：
    - get_current_pose()  : 获取当前位姿（正运动学 FK）
    - move_to_pose()      : 规划并运动到指定位姿，返回详细状态
    """

    def __init__(self, motion: GalbotMotion):
        """
        Args:
            motion: 已初始化的 GalbotMotion 实例
        """
        self.motion = motion

    # =========================================================================
    # 原子 API 1: 获取当前位姿（正运动学 FK）
    # =========================================================================

    def get_current_pose(
        self,
        arm: str,
        reference_frame: str = "base_link",
    ) -> Optional[List[float]]:
        """
        获取当前末端位姿（正运动学反馈）

        相当于机器人的"我现在在哪里"

        Args:
            arm: 机械臂名称 (如 'right_arm', 'left_arm')
            reference_frame: 参考坐标系

        Returns:
            位姿 [x, y, z, qx, qy, qz, qw]，失败返回 None
        """
        status, pose = self.motion.get_end_effector_pose_on_chain(
            chain_name=arm,
            frame_id="EndEffector",
            reference_frame=reference_frame,
        )

        if status != MotionStatus.SUCCESS:
            logger.error(f"读取当前末端位姿失败: status={status}")
            return None

        return list(pose)

    # =========================================================================
    # 原子 API 2: 运动到指定位姿（规划 + 执行）
    # =========================================================================

    def move_to_pose(
        self,
        target_pose: List[float],
        arm: str,
        reference_frame: str = "base_link",
        enable_collision_check: bool = False,
        timeout: float = 10.0,
        wait: bool = True,
    ) -> MotionStatus:
        """
        规划并运动到指定位姿

        这是底层唯一的运动原语，所有复杂运动都应该在上层通过组合它来实现。

        设计理念：
        - 只负责从 A 到 B 的规划和执行
        - 返回 SDK 原生 MotionStatus，让上层决定如何处理失败
        - 不做任何"聪明"的猜测，只做明确要求的事

        Args:
            target_pose: 目标位姿 [x, y, z, qx, qy, qz, qw]
            arm: 机械臂名称
            reference_frame: 参考坐标系
            enable_collision_check: 是否启用碰撞检测
            timeout: 超时时间（秒）
            wait: 是否等待运动完成

        Returns:
            MotionStatus - SDK 原生状态枚举，直接使用
        """
        if len(target_pose) != 7:
            logger.error(f"位姿格式错误: 期望 7 个值，实际 {len(target_pose)} 个")
            return MotionStatus.INVALID_INPUT

        logger.debug(
            f"运动到目标位姿: x={target_pose[0]:.3f}, y={target_pose[1]:.3f}, "
            f"z={target_pose[2]:.3f}"
        )

        status = self.motion.set_end_effector_pose(
            target_pose=target_pose,
            end_effector_frame=arm,
            reference_frame=reference_frame,
            enable_collision_check=enable_collision_check,
            is_blocking=wait,
            timeout=timeout,
            params=gm.Parameter(),
        )

        if wait:
            time.sleep(0.3)

        return status


# =============================================================================
# 直接运行示例 - 真实可运行的演示代码
# =============================================================================

if __name__ == "__main__":
    """
    直接运行此文件，演示 MotionPlanner 的真实使用方式

    运行方式：
        cd pick_place_pipeline
        python motion_planning.py

    注意：实际运行需要机器人硬件连接
    """
    import sys
    from galbot_sdk.g1 import GalbotMotion, MotionStatus

    print("=" * 60)
    print("  MotionPlanner 真实运行示例")
    print("=" * 60)
    print()

    # -------------------------------------------------------------------------
    # 1. 初始化机器人运动模块
    # -------------------------------------------------------------------------
    print("【1/3】初始化运动模块...")
    try:
        robot = GalbotRobot()
        motion = GalbotMotion()
        robot.init()
        motion.init()
        planner = MotionPlanner(motion)
        print("  ✅ 初始化成功")
    except Exception as e:
        print(f"  ❌ 初始化失败: {e}")
        print("  请检查机器人是否连接")
        sys.exit(1)
    print()

    try:
        # -------------------------------------------------------------------------
        # 2. 示例 1: 获取当前位姿（FK 正运动学）
        # -------------------------------------------------------------------------
        print("【2/3】演示: get_current_pose() - 获取当前末端位姿")
        print()

        current_pose = planner.get_current_pose(arm="right_arm")
        if current_pose:
            print("  ✅ 获取成功")
            print(
                f"     位置 (m): x={current_pose[0]:.4f}, y={current_pose[1]:.4f}, z={current_pose[2]:.4f}"
            )
            print(
                f"     姿态四元数: qx={current_pose[3]:.4f}, qy={current_pose[4]:.4f}, qz={current_pose[5]:.4f}, qw={current_pose[6]:.4f}"
            )
        else:
            print("  ❌ 获取失败")
        print()

        # -------------------------------------------------------------------------
        # 3. 示例 2: 使用 move_to_pose 运动到目标位姿
        # -------------------------------------------------------------------------
        print("【3/3】演示: move_to_pose() - 运动到目标位姿")
        print()

        if current_pose:
            # 在当前位姿基础上，x 轴向前移动 0.1m
            # 姿态保持不变
            target_pose = [
                current_pose[0] + 0.1,  # x 向前 +0.1m
                current_pose[1],  # y 不变
                current_pose[2],  # z 不变
                current_pose[3],  # qx 不变
                current_pose[4],  # qy 不变
                current_pose[5],  # qz 不变
                current_pose[6],  # qw 不变
            ]

            print("  目标位姿:")
            print(
                f"     位置 (m): x={target_pose[0]:.4f}, y={target_pose[1]:.4f}, z={target_pose[2]:.4f}"
            )
            print(
                f"     姿态四元数: qx={target_pose[3]:.4f}, qy={target_pose[4]:.4f}, qz={target_pose[5]:.4f}, qw={target_pose[6]:.4f}"
            )
            print()
            print("  开始运动...")

            status = planner.move_to_pose(
                target_pose=target_pose,
                arm="right_arm",
                reference_frame="base_link",
                enable_collision_check=False,
                timeout=10.0,
                wait=True,
            )

            # 使用辅助函数打印详细状态
            print_motion_status(status, "运动到目标位姿")
            print()

            # 运动完成后再次获取当前位姿验证
            print("  验证运动结果...")
            new_pose = planner.get_current_pose(arm="right_arm")
            if new_pose:
                print("  ✅ 运动后位姿:")
                print(
                    f"     位置 (m): x={new_pose[0]:.4f}, y={new_pose[1]:.4f}, z={new_pose[2]:.4f}"
                )
        else:
            print("  ⚠️  无法获取当前位姿，跳过运动演示")
        print()

    except KeyboardInterrupt:
        print("\n\n用户中断测试")
    except Exception as e:
        print(f"\n❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()

    # -------------------------------------------------------------------------
    # 总结
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("  示例运行完成")
    print("=" * 60)
    print()
    print("  核心 API 总结:")
    print("    1. planner.get_current_pose(arm='right_arm')")
    print("       -> 返回 [x, y, z, qx, qy, qz, qw]")
    print()
    print("    2. planner.move_to_pose(target_pose, arm='right_arm')")
    print("       -> 返回 MotionStatus 枚举")
    print()
    print("  状态处理:")
    print("    if status != MotionStatus.SUCCESS:")
    print("        print_motion_status(status, '操作名称')")
    print()
