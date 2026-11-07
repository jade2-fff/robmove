# 迁移范围与验证

## 已迁入 C++ 的职责

| 原 Python 代码 | C++ 实现 |
| --- | --- |
| `config.py` 的任务、机器人、物料参数 | `config.hpp`、`config.cpp` 与 `config/*.json` |
| `utils.py` 的位姿变换、反投影、夹爪宽度 | `geometry.cpp` |
| `perception.py` 的深度中位数与候选优先级 | `geometry.cpp`、`planning.cpp` |
| `pick_place_task.py` 的抓取姿态算法 | `planning.cpp` |
| `pick_place_task.py` 的动作和夹爪反馈判断 | `task.cpp`、`GripMonitor` |
| `task_workflows.py` 的单次和循环状态编排 | `workflow.cpp` |
| `main.py` 的命令入口与退出码 | `app/main.cpp`（离线入口，参数采用新命名） |
| `logger.py` 的日志职责 | `EventSink` 与 CLI JSON Lines |
| 独立 YOLO 示例的水平点位抓取算法 | `plan_horizontal` |

## 延后到设备版本确认的职责

Galbot SDK 调用、相机采集、双目深度推理、YOLO 分割、FoundationPose 配准以及调试图像绘制没有被替换为真实 C++ 后端。`RobotPort` 和 `PerceptionPort` 已定义边界；当前实现是模拟机器人与 JSON 位姿回放。已有 `.pt` 模型不能直接视为可供 C++ 推理的导出模型。

没有连接机器人，没有安装、替换或升级机器人系统/SDK，没有执行运动、使能、夹爪或控制器切换。原 Python 模块与资源继续保留，方便后续对照和接入；原项目说明保存在 `docs/python_reference.md`。

## 有意调整的行为

- 碰撞检查默认 `true`；原 `motion_planning.py` 默认 `false`。真实接入时仍需确认 SDK 的碰撞场景和实际支持。
- 配置不再依赖全局 `cfg` 和 Python 导入顺序。
- 无效关节反馈直接失败，不在无法确认当前位置时继续执行归位。
- 空缺/异常夹爪反馈清空连续稳定计数；晚于截止时间的反馈不允许抬升。
- 闭爪后未确认成功时记录 `PossiblyHolding`，退出时不自动开爪。
- 阻塞模拟动作没有原流程每次调用后固定的 0.3 秒等待。真实后端须根据到位反馈返回，不能靠删除等待宣称运动变快。
- 导航持久参数的管理由未来的硬件适配器负责；当前核心不会修改设备配置。
- 回放文件耗尽返回 `Unavailable`，绝不把 EOF 当成有效空场。
- 本地可执行程序仅运行模拟后端；`--demo` 或 `--replay` 显式选择输入。

## 数学一致性

`tools/reference_oracle.py` 只从原 Python 文件提取指定的纯算法 AST 节点，不导入 SDK 或运行真实动作。固定随机种子生成 200 组原算法结果，覆盖五类物料、左右臂、直立、倒伏、倒置、阈值附近和随机姿态。

`tests/golden_test.cpp` 对比 pre_grasp、grasp、lift_safe、抓取分支，位置和旋转角误差阈值为 `1e-9`。比较四元数所代表的旋转，允许 q 和 -q 等价。参考文件记录用于生成对照数据的源码 SHA-256；`legacy_integrity` 同时检查迁移清单中当前保留的 27 个文件与已记录哈希一致。

正常构建直接读取已生成的参考数据，不需要 NumPy、SciPy 或 Galbot SDK。重生成参考数据时才需要原算法的 NumPy/SciPy 开发环境：

```bash
python3 tools/reference_oracle.py
```

## 本机验证记录

2026-11-07 按用户要求移除 `yolo26s-seg_finetuned.pt`（23,353,255 字节），C++ 和 Python 默认模型均切换为现有的 `yolo11n-seg_finetuned.pt`。迁移清单保留已删除资源的信息及修改前的配置来源哈希，规划对照数据同步更新配置哈希。此次调整不改变规划算法，未运行真实模型推理或机器人动作。

- GCC 13.3.0、CMake 3.28.3、Eigen 3、nlohmann/json 3.11.3。
- Debug / Release 下执行核心行为测试、命令行集成测试、原始文件哈希检查。
- 38 个 C++ 测试，其中一项覆盖 200 组原算法对照；6 个 Python 标准库 CLI 测试。
- CMake 安装导出后由单独消费项目 `find_package(robmove)` 验证公共头文件与库链接。
- AddressSanitizer / UndefinedBehaviorSanitizer 检查；当前受跟踪环境下 LeakSanitizer 自检报错，因此本地用 `ASAN_OPTIONS=detect_leaks=0`，未声称完成泄漏检测。CI 配置保留默认泄漏检查。
- cppcheck 检查新 C++ 代码；编译器告警作为错误处理。

以上是本机离线验证，不是机器人实测，也不验证碰撞可行性、物理可达性、真实夹持或推理精度。

## 性能范围

本机 x86_64、GCC Release，固定直立乌龙茶物体规划，逐轮改变 x 坐标；预热后重复五次。Python 原方法每轮执行 10,000 次，C++ 每轮执行 500,000 次，两端校验累计结果以防循环被消除。

本次中位数：Python **48.22 µs/次**，C++ **0.109 µs/次**，规划内核约 **443 倍**。原始采样在 `benchmarks/local_result.json`。该数据仅覆盖 CPU 抓取规划，不包含模型加载、图像传输、YOLO、FoundationPose、SDK 往返或机械动作；不能外推为整机吞吐提升。

```bash
python3 tools/compare_performance.py \
  --binary build/release/robmove_benchmark \
  --output benchmarks/local_result.json
```

本机使用 `python3 -s` 运行可选对照工具，以选用系统内匹配的 NumPy/SciPy；项目本身不修改 Python、SDK 或系统依赖。
