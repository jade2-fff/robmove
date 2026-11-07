# robmove — 视觉感知与位姿回放

项目保留相机读取、YOLO 分割、FoundationPose 物体位姿估计参考，以及可独立构建的 C++20 几何、感知配置和离线回放工具。0.2.0 已移除机械臂、夹爪、电机与导航控制、抓取规划、动作编排及对应模拟器。

C++ 命令行处理记录好的感知结果；它不加载推理模型，也不连接设备。Python 感知模块保留现有相机读取适配，需要匹配实际设备的 SDK 和推理环境。项目保留 YOLO11n 权重及五类物体的 CAD/纹理，YOLO26 权重已移除。

## 构建与检查

需要 C++20 编译器、CMake 3.21+、Ninja、Eigen 3 和 nlohmann/json。构建不联网、不自动下载依赖。Ubuntu 开发包包括 `libeigen3-dev`、`nlohmann-json3-dev`、`ninja-build`。

```bash
git clone https://github.com/jade2-fff/robmove.git
cd robmove
cmake --preset release
cmake --build --preset release --parallel 2
ctest --preset release
```

## 离线感知回放

```bash
# 输出内置演示观测
./build/release/robmove --demo --json

# 输出记录中的所有物体候选、位置与姿态
./build/release/robmove --replay examples/replay.json --json

# 校验感知配置
./build/release/robmove --config config/default.json --validate-config
```

`--demo` 和 `--replay` 互斥。JSON Lines 输出包含每帧 `observation`，最后是帧数、候选总数和不可用帧数的 `summary`。读取有限帧后退出。退出码：`0` 有效回放完成，`1` 参数、配置或数据错误，`2` 没有观测帧或包含不可用帧。

旧的 `--mode`、`--trace`、`--realtime` 参数及 `robot`、`navigation`、`task` 配置块已移除，传入时会明确报错。

## 工程结构

```text
app/                      只读感知回放命令行
include/robmove/           感知类型、配置、几何与接口
src/                      几何计算、配置校验
src/adapters/             JSON 感知回放
config/                   感知与胸部 YOLO 调试配置
examples/replay.json       离线观测示例
tests/                    几何、配置、回放、CLI、安装与资源完整性检查
pick_place_pipeline/      保留的 Python 感知、配置、坐标变换和日志模块
assets/                   YOLO11n 权重、物体 CAD 和纹理
migration_manifest.json   当前文件哈希、原始来源及移除记录
```

`Candidate.object_pose` 表示 `base_link <- object`，位置和宽度使用米，四元数顺序为 `[qx,qy,qz,qw]`。输出描述观测到的物体，不生成执行轨迹。

## 感知配置

`config/default.json` 只保留感知参数。物料定义只包含 CAD 路径，类别别名按 key 合并默认目录。相对路径以配置文件的 `resource_root` 为基准。

```json
{
  "resource_root": "..",
  "perception": {
    "objects": {"new_box": {"mesh_path": "assets/objects/new_box/normalized.obj"}},
    "aliases": {"new box": "new_box"}
  }
}
```

配置检查只验证字段和数值，不代表已验证模型类别、推理精度或设备兼容性。

## 库与协作

```bash
cmake --install build/release --prefix "$PWD/out/perception-install"
```

使用新的安装目录可避免混入旧版本头文件。消费项目通过 `find_package(robmove 0.2 CONFIG REQUIRED)` 引入 `robmove::core`；回放适配器使用 `robmove::adapters`。

开发检查也可使用 `dev` 和 `asan` 预设。详细约定见 [架构说明](docs/architecture.md)、[变更记录](docs/migration.md)、[Python 感知参考](docs/python_reference.md)；提交步骤见 [协作指南](CONTRIBUTING.md)。
