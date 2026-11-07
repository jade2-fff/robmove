# Python 感知参考

目录名 `pick_place_pipeline` 沿用原始迁移目录；当前仅保留四个模块：

- `perception.py`：YOLO 检测/分割、FoundationPose 物体位姿估计、相机数据和坐标变换读取。
- `config.py`：感知模型、物料 CAD、深度处理及日志配置，可在无设备 SDK 的环境中读取。
- `utils.py`：像素反投影、位姿矩阵和坐标变换等纯数学函数。
- `logger.py`：日志输出。

原抓取执行入口、机械臂/夹爪封装、导航和抓放任务已删除。感知模块也不再包含自动连接机器人的直接运行示例。

## 感知链路

头部 RGB 与深度 → YOLO 分割 → 按置信度选择候选 → FoundationPose 估计物体 6DoF 位姿 → 转换到 `base_link`。

返回数据包括类别、置信度、物体位置与姿态、观测宽度。候选不再按左右臂可达性排序。每类物料仅配置 CAD 路径，抓取偏移和夹爪参数已移除。

## 运行依赖

Python 推理代码仍需要适配实际设备的 `galbot_sdk.g1`，以及 NumPy、OpenCV、PyTorch、Ultralytics、SciPy、Loguru、Trimesh 和 FoundationPose 的依赖。C++ 离线回放不需要这些依赖。

当前默认分割权重为项目内的 `assets/models/yolo11n-seg_finetuned.pt`。FoundationPose 为外部工程，`fp_root` 需按实际环境配置。SDK 版本不能从源码中的历史注释推断，集成时需单独核实。

本次只做本地源码清理与离线验证，没有执行相机采集、模型推理或机器人运动。
