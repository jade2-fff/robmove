# G1 抓取流程开发副本

从机器人 galbot-echo 的 `/home/galbot/galbot-training-intermediate` 迁移到这里。`pick_place_pipeline` 保留原有9个Python模块，`examples` 包含实际存在的单次YOLO示例。`assets` 保存这两个流程引用的分割模型、五类物体CAD和材质纹理。

主流程依次执行感知前归位、识别和三维定位、预抓取、接近、夹爪闭合、抬升、放置、回退和归位。当前默认硬件为右臂、右夹爪、头部左目RGB与FOUNDATION_STEREO双目深度。主感知链路使用YOLO分割和FoundationPose。

## 文件入口

- `pick_place_pipeline/main.py` 是主入口；`--single_grasp_no_navigation` 进入单次抓取和原地放回，循环模式需要导航开关和导航目标位姿。
- `pick_place_pipeline/task_workflows.py` 编排单次与循环任务。
- `pick_place_pipeline/pick_place_task.py` 实现抓取、抬升、放置和夹爪反馈检查。
- `pick_place_pipeline/perception.py` 负责图像、分割、FoundationPose和坐标变换。
- `pick_place_pipeline/motion_planning.py` 封装机械臂SDK调用。
- `pick_place_pipeline/config.py` 保存机械臂、夹爪、物料、模型与导航配置。
- `examples/yolo_single_pick_place.py` 是原工程独立YOLO示例，使用分割和RGB-D质心估计目标位置。

## 工作目录和依赖

将 `/home/jade/Desktop/robmove` 作为工作目录，并保留 `pick_place_pipeline`、`examples`、`assets` 的同级关系。YOLO权重和CAD相对路径依赖这一布局。

需要为实际部署机器配置匹配机型及版本的 `galbot_sdk.g1`。Python依赖包括 NumPy、OpenCV、PyTorch、Ultralytics、SciPy、Pillow、Loguru、Trimesh，以及FoundationPose依赖的Warp/CUDA扩展。

FoundationPose是外部工程。`config.py` 中的 `fp_root` 当前指向机器人路径 `/home/galbot/FoundationPose`，在新环境中应指向实际安装目录。FoundationPose本身及其两套评分/精修权重需要另外部署，参考原机外部路径：

- `/home/galbot/FoundationPose/weights/2023-10-28-18-33-37/model_best.pth`
- `/home/galbot/FoundationPose/weights/2024-01-11-20-02-45/model_best.pth`

迁移保留了原配置内容。代码中的动作入口会调用真实机械臂和夹爪，执行前需要完成环境适配、运动参数核对和现场安全确认。此次只做源码、资源和静态完整性检查。

## 接入透明瓶模型前需处理的事项

1. 当前主模型为 `yolo26s-seg_finetuned.pt`，物料配置覆盖乌龙茶、纸巾、薯片、方便面和可乐。前面训练的 `clear_bottle` 检测模型仍需与深度定位、抓取姿态和夹爪参数对接。
2. 主程序注释推荐的 `examples/single_pick_place.py` 和 `examples/cyclic_pick_place.py` 在原机缺失。当前可定位的实际入口是上述 `main.py` 与 `yolo_single_pick_place.py`。
3. 独立YOLO示例的命令行默认目标拼为 `leshishupin`，配置类别为 `leshishupian`。调用时应明确填写与模型一致的实际类别名。
4. 主流程默认目标位姿运动的 `collision_check=False`，导航位姿默认未配置。集成时需按实际工位审查这些设置。

资源及代码的来源哈希记录在 `migration_manifest.json`，用于核对迁移完整性。
