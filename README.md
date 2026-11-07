# robmove — C++20 抓取流程工程

把原 G1 Python 抓放流程的规划计算、任务编排、动作执行逻辑和配置迁入独立 C++ 库。提供可运行的离线 CLI、位姿回放、模拟机器人、自动测试及性能基准。

**当前范围：本机 C++ 核心。** 可执行程序只使用模拟后端，尚未接入真实 Galbot SDK、相机或 YOLO/FoundationPose 推理。设备版本确认后可通过接口接入。原 Python 和模型/CAD 资源完整保留。

## 构建与运行

需要 C++20 编译器、CMake 3.21+（使用预设时）、Ninja、Eigen 3 和 nlohmann/json。构建不联网、不自动下载依赖。本机已有所需依赖；其他 Ubuntu 环境对应开发包为 `libeigen3-dev`、`nlohmann-json3-dev`、`ninja-build`。

```bash
git clone https://github.com/jade2-fff/robmove.git
cd robmove
cmake --preset release
cmake --build --preset release --parallel 2
ctest --preset release

# 单次抓取、原地放回、归位（模拟）
./build/release/robmove --demo

# 循环抓取、放置，连续空场后回原点（模拟导航）
./build/release/robmove --demo --mode cyclic --config config/simulation.json

# 回放候选位姿，输出 JSON Lines 日志与模拟指令轨迹
./build/release/robmove --replay examples/replay.json \
  --config config/default.json --json --trace build/trace.json
```

`--demo` 使用内置演示观测；`--replay` 读取实际或合成位姿记录；二者互斥。默认使用虚拟时钟，无需等待模拟动作；`--realtime` 可演示等待与 Ctrl+C 取消。无参数或 `--help` 显示帮助。

退出码：`0` 完成或确认有效空场，`1` 参数/配置/输出错误，`2` 执行进入 HOLD，`130` 用户中断。模拟器不计算 IK、动力学或真实碰撞。

## 工程结构

```text
app/                      命令行入口、事件输出
include/robmove/           公共 C++ API、类型、端口
src/                      配置、几何、规划、任务执行、状态机
src/adapters/             JSON 感知回放、模拟机器人
config/                   默认参数与模拟导航路点
examples/replay.json       可直接运行的回放输入
tests/                    行为、异常、原算法对照、CLI、安装测试
benchmarks/               规划性能基准及本机采样
cmake/                    可安装的 find_package 配置
docs/                     架构、迁移范围、验证记录
pick_place_pipeline/      保留的原 Python 参考实现
assets/                   原模型、CAD、纹理
migration_manifest.json   原始文件来源与哈希
```

核心库使用固定大小 Eigen 数据结构；任务编排不依赖 SDK 类型。`RobotPort`、`PerceptionPort`、`Clock`、`EventSink` 分别隔离设备、感知、时间和日志。新物料通过配置扩展，新设备/推理实现通过端口替换。

## 配置与接入

修改 `config/default.json` 中的机器人、夹爪和重试参数。字段采用严格校验，拼写错误和重复 JSON key 会报错。路径相对配置文件的 `resource_root` 解析，与进程工作目录无关。

```bash
./build/release/robmove --config config/default.json --validate-config
```

默认五类物料与原配置一致。新增物料示例：

```json
{
  "resource_root": "..",
  "perception": {
    "objects": {
      "new_box": {
        "mesh_path": "assets/objects/new_box/normalized.obj",
        "grasp_point_object": [0.0, 0.0, 0.08],
        "pre_grasp_distance": 0.08,
        "gripper_ee_distance": 0.18
      }
    },
    "aliases": {"new box": "new_box"}
  }
}
```

物料与别名按 key 合并默认配置。真实模型是否包含新增类别，仍由后续感知接入验证。`config/simulation.json` 内的导航坐标仅供模拟。

详细约定见 [架构与设备接入](docs/architecture.md)；模块映射、行为调整和验证范围见 [迁移记录](docs/migration.md)。

## 检查与性能

```bash
cmake --preset dev
cmake --build --preset dev --parallel 2
ctest --preset dev

cmake --preset asan
cmake --build --preset asan --parallel 2
ctest --preset asan
```

如果运行环境明确报告 LeakSanitizer 无法在 ptrace 下工作，可在本次测试进程关闭泄漏检测，保留地址与未定义行为检查：

```bash
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 ctest --preset asan
```

测试包含 200 组原 Python 规划算法对照、抓放顺序、失败停止、反馈超时、连续空场、资源清理、CLI 及安装导出。普通构建无需 Python 推理依赖；可选的原算法对照生成和性能比较工具需要 NumPy/SciPy。

本机规划内核中位耗时从约 **48.2 µs** 降到 **0.109 µs**。这只衡量 CPU 规划计算；端到端机器人性能尚未测试。数据和复现方法见 [性能记录](docs/migration.md#性能范围)。

## 作为库使用

```bash
cmake --install build/release --prefix "$PWD/out/install"
```

消费项目设置 `CMAKE_PREFIX_PATH` 指向该目录：

```cmake
find_package(robmove CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE robmove::core)
# 如需模拟与回放适配器，再链接 robmove::adapters
```

安装导出二进制、库和头文件；模型、CAD 与配置仍通过外部资源路径管理。

## 协作开发

协作者接受 GitHub 仓库邀请后，可以克隆仓库并推送自己的开发分支，再通过 Pull Request 合并到 `main`。详细步骤和提交前检查见 [协作指南](CONTRIBUTING.md)。
