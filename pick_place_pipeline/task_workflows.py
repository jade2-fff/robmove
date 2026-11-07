"""单物体与循环抓取流程编排。

抓取侧共用的末端位姿序列
------------------------
pre_grasp -> grasp -> 闭合夹爪(GRASP) -> lift

- lift：闭合夹爪后，末端沿抓取位姿原地竖直上抬 ``lift_height``，避免抬手
  过程中拖拽物体或磕碰周边物品。两个流程都会执行这一步。

两个流程都不依赖任何现场标定的固定释放位姿（没有 ``place_release_pose``
这类字段）：放置位姿统一直接取本轮抓取位姿 ``target.plan.grasp``——抓取点
相对 ``base_link`` 在哪，放置点就在哪；循环流程会先导航到放置点，所以这个
"相对 base_link 的位置"落地到的是导航后的新底盘位置附近。

单物体流程的放置侧（不导航，与 grasp 对称）
------------------------------------------
lift -> place（下降回 target.plan.grasp，打开夹爪）(LOCAL_PLACE)
-> retreat_from_place（沿原接近路径退回 target.plan.pre_grasp）
(RETREAT_FROM_PLACE) -> 机械臂归位

单物体流程不导航，因此和 grasp 对称：直接下压回抓取位姿释放物体（原地
放回），再沿原接近路径退回预抓取位姿，不需要循环流程的 lift_safe。

循环流程的放置侧（导航后从抬升高度上方释放）
--------------------------------------------------
lift -> lift_safe -> 携物归位 -> 导航到放置点 -> 放置点下的 lift_safe
-> 放置区安全抬升位姿（RELEASE，对应 ``place_local``）
-> 退回放置点下的 lift_safe（RETREAT_TO_LIFT_SAFE）-> 机械臂归位

- lift_safe：在抬升高度上做一次水平回退，形成携物导航前的安全姿态。
  FoundationPose 侧面接近复用预抓取位姿的水平位置；斜躺/平躺物体采用
  竖直接近时，沿 base_link -X 回退该物体的 ``pre_grasp_distance``。
- 循环流程的放置点通常是一个料框：到达放置导航点后，不下压到抓取位姿
  对应的高度，而是先走到放置点坐标系下的 ``lift_safe``，再水平进入其
  正上方 ``lift_height`` 处打开夹爪释放物体，避免从携物归位姿态直接
  斜插到料框上方，也避免碰到框壁。这个抬升高度沿用已有的可配置
  ``lift_height``（``cfg.robot.lift_height``），不新增专用字段。
- 释放后先在同一抬升高度水平退回放置点坐标系下的 ``lift_safe``，再执行
  机械臂归位，避免从料框正上方直接走关节归位轨迹。

主流程只使用 FoundationPose。可抓类别直接取自
``PerceptionConfig.foundationpose_objects``，每次返回抓取点后重新感知。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from config import PipelineConfig
from logger import logger
from pick_place_task import (
    PerceptionResult,
    PerceptionStatus,
    PickPlaceTask,
    PickTarget,
)


class TaskState(Enum):
    """两个任务入口共享的可观测状态。"""

    INIT = auto()
    NAV_TO_ORIGIN = auto()
    NAV_TO_PICKUP = auto()
    ARM_HOME_FOR_PERCEPTION = auto()
    LOAD_PERCEPTION_MODEL = auto()
    CAMERA_PERCEPTION = auto()
    GRASP = auto()
    LIFT = auto()
    LIFT_SAFE = auto()
    ARM_HOME_WITH_OBJECT = auto()
    NAV_TO_PLACE = auto()
    LOCAL_PLACE = auto()
    RETREAT_FROM_PLACE = auto()
    RELEASE = auto()
    RETREAT_TO_LIFT_SAFE = auto()
    ARM_HOME_AFTER_RELEASE = auto()
    HOLD = auto()
    DONE = auto()


@dataclass(frozen=True)
class WorkflowResult:
    """任务入口的最终结果。"""

    success: bool
    placed_count: int
    reason: str
    held: bool = False


def _log_state(state: TaskState, cycle: int, placed_count: int) -> None:
    logger.info(
        f"任务状态: {state.name}, cycle={cycle}, placed={placed_count}"
    )


def _log_perception(result: PerceptionResult) -> None:
    detail = f", reason={result.reason}" if result.reason else ""
    logger.info(f"感知结果: {result.status.name}{detail}")


def _hold_result(
    task: PickPlaceTask, reason: str, placed_count: int
) -> WorkflowResult:
    _log_state(TaskState.HOLD, placed_count + 1, placed_count)
    task.hold(reason)
    logger.error(
        f"任务结束: placed={placed_count}, reason={reason}, status=HOLD"
    )
    return WorkflowResult(False, placed_count, reason, held=True)


def _done_result(reason: str, placed_count: int) -> WorkflowResult:
    _log_state(TaskState.DONE, placed_count, placed_count)
    logger.info(f"任务结束: placed={placed_count}, reason={reason}, status=DONE")
    return WorkflowResult(True, placed_count, reason)


def _perceive_with_unavailable_retries(
    task: PickPlaceTask,
    config: PipelineConfig,
    *,
    cycle: int,
    placed_count: int,
) -> PerceptionResult:
    """只重试链路故障；有效空场立即交由循环做连续确认。"""

    result = PerceptionResult(
        PerceptionStatus.UNAVAILABLE, reason="尚未执行感知"
    )
    for attempt in range(1, config.task.perception_retries + 1):
        _log_state(TaskState.CAMERA_PERCEPTION, cycle, placed_count)
        result = task.perceive_one()
        _log_perception(result)
        if result.status is not PerceptionStatus.UNAVAILABLE:
            return result
        logger.warning(
            "感知链路不可用，"
            f"重试 {attempt}/{config.task.perception_retries}: {result.reason}"
        )
        if attempt < config.task.perception_retries:
            time.sleep(config.task.perception_retry_interval_s)
    return result


def run_single_pick_place(config: PipelineConfig) -> WorkflowResult:
    """执行一次无导航的感知、抓取、本地放置与归位。"""

    _log_state(TaskState.INIT, 1, 0)
    with PickPlaceTask(config, enable_navigation=False) as task:
        _log_state(TaskState.ARM_HOME_FOR_PERCEPTION, 1, 0)
        if not task.return_arm_home():
            return _hold_result(task, "感知前机械臂归位失败", 0)

        target: Optional[PickTarget] = None
        last_result: Optional[PerceptionResult] = None
        for attempt in range(1, config.task.perception_retries + 1):
            _log_state(TaskState.CAMERA_PERCEPTION, 1, 0)
            last_result = task.perceive_one()
            _log_perception(last_result)
            if last_result.status is PerceptionStatus.FOUND:
                target = last_result.target
                break
            logger.info(
                f"单物体感知尝试 {attempt}/{config.task.perception_retries} "
                f"未获得目标"
            )
            if attempt < config.task.perception_retries:
                time.sleep(config.task.perception_retry_interval_s)

        if target is None:
            if (
                last_result is not None
                and last_result.status is PerceptionStatus.EMPTY
            ):
                return _done_result(
                    "重试后仍未发现可抓目标，安全退出", 0
                )
            reason = (
                last_result.reason
                if last_result is not None and last_result.reason
                else "感知链路不可用"
            )
            return _hold_result(task, reason, 0)

        _log_state(TaskState.GRASP, 1, 0)
        if not task.pick(target):
            return _hold_result(task, "抓取动作失败", 0)

        _log_state(TaskState.LIFT, 1, 0)
        if not task.lift(target):
            return _hold_result(task, "抓取后抬升失败", 0)

        _log_state(TaskState.LOCAL_PLACE, 1, 0)
        if not task.place(target):
            return _hold_result(task, "放置或释放失败", 0)

        _log_state(TaskState.RETREAT_FROM_PLACE, 1, 1)
        if not task.retreat_from_place(target):
            return _hold_result(task, "放置后退回失败", 1)

        _log_state(TaskState.ARM_HOME_AFTER_RELEASE, 1, 1)
        if not task.return_arm_home():
            return _hold_result(task, "释放后机械臂归位失败", 1)

        return _done_result("单物体抓取放置完成", 1)


def run_cyclic_pick_place(config: PipelineConfig) -> WorkflowResult:
    """从当前位置前往抓取点并与放置点循环，空场后返回原点。"""

    placed_count = 0
    empty_count = 0
    _log_state(TaskState.INIT, 1, placed_count)

    with PickPlaceTask(
        config,
        enable_navigation=True,
        load_perception_on_initialize=False,
    ) as task:
        _log_state(TaskState.NAV_TO_PICKUP, 1, placed_count)
        if not task.navigate_to(config.pickup_pose, "抓取点"):
            return _hold_result(task, "导航到抓取点失败", placed_count)

        _log_state(TaskState.ARM_HOME_FOR_PERCEPTION, 1, placed_count)
        if not task.return_arm_home():
            return _hold_result(task, "感知前机械臂归位失败", placed_count)

        _log_state(TaskState.LOAD_PERCEPTION_MODEL, 1, placed_count)
        try:
            task.load_perception_model()
        except Exception as error:
            logger.exception(f"感知模型加载失败: {error}")
            return _hold_result(task, f"感知模型加载失败: {error}", placed_count)

        while True:
            cycle = placed_count + 1
            result = _perceive_with_unavailable_retries(
                task,
                config,
                cycle=cycle,
                placed_count=placed_count,
            )

            if result.status is PerceptionStatus.UNAVAILABLE:
                reason = f"感知重试耗尽: {result.reason}"
                return _hold_result(task, reason, placed_count)

            if result.status is PerceptionStatus.EMPTY:
                empty_count += 1
                required_empty = config.task.empty_confirmations
                logger.info(f"连续空场确认: {empty_count}/{required_empty}")
                if empty_count >= required_empty:
                    _log_state(
                        TaskState.NAV_TO_ORIGIN, cycle, placed_count
                    )
                    if not task.navigate_to(config.origin_pose, "原点"):
                        return _hold_result(
                            task, "空场确认后返回原点失败", placed_count
                        )
                    return _done_result("抓取区已连续确认为空", placed_count)

                time.sleep(config.task.perception_retry_interval_s)
                continue

            empty_count = 0
            target = result.target
            if target is None:
                return _hold_result(
                    task, "FOUND 感知结果缺少抓取目标", placed_count
                )

            _log_state(TaskState.GRASP, cycle, placed_count)
            if not task.pick(target):
                return _hold_result(task, "抓取动作失败", placed_count)

            _log_state(TaskState.LIFT, cycle, placed_count)
            if not task.lift(target):
                return _hold_result(task, "抓取后抬升失败", placed_count)

            _log_state(TaskState.LIFT_SAFE, cycle, placed_count)
            if not task.lift_safe(target):
                return _hold_result(
                    task, "抓取后水平回退至安全位姿失败", placed_count
                )

            _log_state(TaskState.ARM_HOME_WITH_OBJECT, cycle, placed_count)
            if not task.return_arm_home():
                return _hold_result(
                    task, "持物时机械臂归位失败", placed_count
                )

            _log_state(TaskState.NAV_TO_PLACE, cycle, placed_count)
            if not task.navigate_to(
                config.place_pose, "放置点", lock_heading=True
            ):
                return _hold_result(task, "导航到放置点失败", placed_count)

            _log_state(TaskState.RELEASE, cycle, placed_count)
            if not task.place_local(target):
                return _hold_result(task, "放置或释放失败", placed_count)

            placed_count += 1
            _log_state(
                TaskState.RETREAT_TO_LIFT_SAFE, cycle, placed_count
            )
            if not task.retreat_to_lift_safe(target):
                return _hold_result(
                    task,
                    "放置后退回 lift_safe 失败",
                    placed_count,
                )

            _log_state(TaskState.ARM_HOME_AFTER_RELEASE, cycle, placed_count)
            if not task.return_arm_home():
                return _hold_result(
                    task, "释放后机械臂归位失败", placed_count
                )

            if placed_count >= config.task.max_cycles:
                return _hold_result(
                    task,
                    f"成功放置数达到 max_cycles={config.task.max_cycles}，"
                    "需要人工检查持续误检或现场残留",
                    placed_count,
                )

            _log_state(TaskState.NAV_TO_PICKUP, placed_count + 1, placed_count)
            if not task.navigate_to(
                config.pickup_pose, "抓取点", lock_heading=True
            ):
                return _hold_result(
                    task, "放置后返回抓取点失败", placed_count
                )

            _log_state(
                TaskState.ARM_HOME_FOR_PERCEPTION,
                placed_count + 1,
                placed_count,
            )
            if not task.return_arm_home():
                return _hold_result(
                    task, "返回抓取点后机械臂归位失败", placed_count
                )
