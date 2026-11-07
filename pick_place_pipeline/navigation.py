"""
导航模块
核心职责：管理底盘导航的所有功能

使用示例：
    from config import PipelineConfig
    from navigation import Navigator
    from galbot_sdk.g1 import GalbotNavigation

    config = PipelineConfig()
    navigator = Navigator(config)

    # 确保定位完成
    navigator.ensure_localized(nav)

    # 导航到目标点
    navigator.navigate_to_waypoint(nav, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
"""

import sys
import time
from typing import Optional, List

from galbot_sdk.g1 import (
    GalbotRobot,
    GalbotNavigation,
    ControlStatus,
    G1ControllerName,
)

from config import PipelineConfig
from logger import logger


class Navigator:
    """底盘导航管理类

    负责：
    1. 切换底盘导航控制器
    2. 确保定位完成
    3. 路径可达性检查
    4. 导航到目标点

    Attributes:
        config: PipelineConfig 配置实例
    """

    # 地图原点的默认位姿
    DEFAULT_MAP_ORIGIN = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    def __init__(self, config: PipelineConfig):
        """
        Args:
            config: PipelineConfig 配置实例，从中获取导航参数
        """
        self.config = config
        # 当前是否处于锁朝向限速状态（速度上限是导航栈的持久配置，
        # 锁过之后需要在下一次普通导航前恢复）
        self._heading_locked = False

    def ensure_nav_controller(self, robot: GalbotRobot) -> bool:
        """
        确保切换到底盘导航控制器

        Args:
            robot: GalbotRobot 实例

        Returns:
            成功返回 True，失败返回 False
        """
        res = robot.switch_controller(G1ControllerName.CHASSIS_POSE_CTRL)
        if res != ControlStatus.SUCCESS:
            logger.warning(f"切换底盘控制器失败: {res}")
            return False
        logger.info("底盘控制器切换成功")
        return True

    def ensure_localized(
        self,
        nav: GalbotNavigation,
        init_pose: Optional[List[float]] = None,
        max_retry: int = 3,
    ) -> bool:
        """
        确保机器人已完成定位，未定位时提示用户并尝试重定位。

        Args:
            nav: GalbotNavigation 实例
            init_pose: 初始位姿 [x, y, z, qx, qy, qz, qw]，None 时使用地图原点
            max_retry: 最大重试次数
        """
        if init_pose is None:
            init_pose = self.DEFAULT_MAP_ORIGIN

        if nav.is_localized():
            logger.info("已完成定位")
            return True

        logger.info("未定位，请将机器人移至地图原点...")
        while True:
            k = input("      -> 移至地图原点后输入 r 继续: ").strip().lower()
            if k == "r":
                break
            logger.info("请输入 r")

        for i in range(max_retry):
            logger.info(f"重定位尝试 {i + 1}/{max_retry}...")
            nav.relocalize(init_pose)
            time.sleep(1.0)
            if nav.is_localized():
                logger.info("重定位成功")
                return True
            logger.warning(f"第 {i + 1} 次重定位未成功")

        logger.error("重定位失败，已达最大重试次数")
        return False

    def get_current_pose(self, nav: GalbotNavigation) -> List[float]:
        """
        获取当前位姿

        Args:
            nav: GalbotNavigation 实例

        Returns:
            当前位姿 [x, y, z, qx, qy, qz, qw]
        """
        return nav.get_current_pose()

    def check_path_reachable(
        self,
        nav: GalbotNavigation,
        goal_pose: List[float],
        start_pose: Optional[List[float]] = None,
    ) -> bool:
        """
        检查路径是否可达

        Args:
            nav: GalbotNavigation 实例
            goal_pose: 目标位姿
            start_pose: 起始位姿，None 时使用当前位姿

        Returns:
            可达返回 True，不可达返回 False
        """
        if start_pose is None:
            start_pose = self.get_current_pose(nav)
        return nav.check_path_reachability(goal_pose, start_pose)

    # 抑制转向时的导航速度上限 [vx, vy, vyaw]
    # SDK 要求每个分量在 [0.05, 1.5] 内，vyaw 取最小值 0.05 rad/s（约 3°/s）
    LOCK_HEADING_VEL_LIMIT = [0.5, 0.5, 0.5]

    # 解除锁朝向后恢复的导航速度上限 [vx, vy, vyaw]
    DEFAULT_VEL_LIMIT = [0.5, 0.5, 1.0]

    def navigate_to_waypoint(
        self,
        nav: GalbotNavigation,
        goal_pose: List[float],
        retry: int = 3,
        timeout: float = 30.0,
        enable_collision_check: bool = True,
        omni_plan: bool = True,
        lock_heading: bool = False,
    ) -> bool:
        """
        导航到目标路点

        Args:
            nav: GalbotNavigation 实例
            goal_pose: 目标位姿 [x, y, z, qx, qy, qz, qw]
            retry: 重试次数
            timeout: 单次导航超时时间（秒）
            enable_collision_check: 是否启用碰撞检测
            omni_plan: 是否使用全向规划（底盘可斜向/横向移动，不原地转向）
            lock_heading: 保持当前朝向平移过去（不转向）。
                实现：目标朝向强制改为当前朝向，并把导航 yaw 速度上限压到
                SDK 允许的最小值 0.05 rad/s，路径跟踪过程中基本不再旋转。
                该速度上限是导航栈的持久配置，本方法会在下一次
                lock_heading=False 的导航前自动恢复为 DEFAULT_VEL_LIMIT。

        Returns:
            成功到达返回 True，失败返回 False
        """
        cur_pose = self.get_current_pose(nav)

        if lock_heading:
            goal_pose = list(goal_pose[:3]) + list(cur_pose[3:])
            ok, status = nav.set_navigation_velocity_limit(
                self.LOCK_HEADING_VEL_LIMIT
            )
            if ok:
                self._heading_locked = True
                logger.info("锁朝向导航：保持当前朝向，yaw 限速 0.05 rad/s")
            else:
                logger.warning(
                    f"锁朝向限速设置失败: {status}，本次导航可能仍会转向"
                )
            # 回读 PNS 实际配置（输出在 SDK 日志中，用于验证限速是否生效）
            nav.dump_navigation_configs()
        elif self._heading_locked:
            ok, status = nav.set_navigation_velocity_limit(
                self.DEFAULT_VEL_LIMIT
            )
            if ok:
                self._heading_locked = False
                logger.info("已恢复默认导航限速")
            else:
                logger.warning(f"恢复默认导航限速失败: {status}")
        logger.info(f"当前位置: {[round(v, 3) for v in cur_pose[:3]]}")
        logger.info(f"目标位置: {[round(v, 3) for v in goal_pose[:3]]}")

        if not self.check_path_reachable(nav, goal_pose, cur_pose):
            logger.warning("路径不可达或存在障碍，跳过导航")
            return False

        for i in range(retry):
            status = nav.navigate_to_goal(
                goal_pose,
                enable_collision_check=enable_collision_check,
                is_blocking=True,
                timeout=timeout,
                omni_plan=omni_plan,
            )
            time.sleep(0.5)
            if nav.check_goal_arrival():
                logger.success("已到达目标点")
                return True
            logger.warning(f"导航未到达 ({i + 1}/{retry}): status={status}")

        logger.warning("导航重试次数耗尽，可能未精确到达目标点")
        return False


if __name__ == "__main__":
    """
    导航模块测试程序：让机器人向前移动 10cm

    运行方式：
        cd pick_place_pipeline
        python navigation.py
    """
    from config import PipelineConfig

    print("=" * 60)
    print("导航模块测试程序")
    print("=" * 60)

    # 1. 初始化配置和导航模块
    print("\n[1/4] 初始化配置和导航模块...")
    config = PipelineConfig()
    navigator = Navigator(config)

    # 2. 连接机器人
    print("\n[2/4] 连接机器人...")
    try:
        robot = GalbotRobot()
        nav = GalbotNavigation()
        robot.init(set())
        nav.init()
        time.sleep(2)
        print("  - 机器人连接成功")
    except Exception as e:
        print(f"  ⚠ 机器人连接失败: {e}")
        sys.exit(1)

    try:
        # 3. 切换底盘控制器 + 确保定位
        print("\n[3/4] 切换底盘控制器并确认定位...")
        navigator.ensure_nav_controller(robot)
        navigator.ensure_localized(nav)

        # 4. 向前移动 10cm（x 轴正方向）
        print("\n[4/4] 向前移动 10cm...")
        cur_pose = navigator.get_current_pose(nav)
        logger.info(f"当前位姿: {[round(v, 3) for v in cur_pose]}")

        goal_pose = list(cur_pose)
        goal_pose[0] += 0.1  # x 轴前进 10cm

        logger.info(f"目标位姿: {[round(v, 3) for v in goal_pose]}")
        success = navigator.navigate_to_waypoint(nav, goal_pose)

        if success:
            print("\n✅ 移动成功，已前进 10cm")
        else:
            print("\n❌ 移动失败，请检查障碍物或定位状态")

    except KeyboardInterrupt:
        print("\n\n用户中断测试")
    except Exception as e:
        print(f"\n❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n清理资源...")
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()

    print("\n" + "=" * 60)
    print("测试结束")
    print("=" * 60)
