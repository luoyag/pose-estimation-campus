# 基于 OpenPose 的校园场景人体姿态估计与行为识别

## 项目简介

本项目基于 OpenPose 技术实现校园场景下的人体姿态估计和行为识别系统。能够实时检测和识别多种校园常见行为，如走路、跑步、站立、坐下、举手等。

## 功能特性

- ✅ 实时人体姿态估计（18个关键点检测）
- ✅ 多种校园行为识别（走路、跑步、站立、坐下、举手等）
- ✅ 支持视频文件和实时摄像头输入
- ✅ 可视化姿态估计结果和行为标签
- ✅ 可扩展的行为识别模型

## 环境要求

- Python 3.8+
- CUDA（可选，用于 GPU 加速）

## 安装步骤

1. 克隆项目到本地
```bash
git clone <your-repo-url>
cd <project-directory>
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 下载 OpenPose 模型（如果需要使用 MediaPipe 作为替代，已包含在依赖中）

## 使用方法

### 基本使用

```bash
# 处理视频文件
python main.py --input video.mp4 --output output.mp4

# 使用摄像头实时处理
python main.py --input 0 --output output.mp4

# 指定行为识别模型
python main.py --input video.mp4 --model behavior_model.pkl
```

### 代码示例

```python
from pose_estimation import PoseEstimator
from behavior_recognition import BehaviorRecognizer

# 初始化姿态估计器
pose_estimator = PoseEstimator()

# 初始化行为识别器
behavior_recognizer = BehaviorRecognizer()

# 处理单帧图像
keypoints = pose_estimator.estimate(image)
behavior = behavior_recognizer.recognize(keypoints)
```

## 项目结构

```
.
├── main.py                 # 主程序入口
├── pose_estimation.py      # 姿态估计模块
├── behavior_recognition.py # 行为识别模块
├── utils.py                # 工具函数
├── visualization.py        # 可视化模块
├── config.py               # 配置文件
├── requirements.txt        # 依赖列表
└── README.md              # 项目说明
```

## 行为识别类别

- 站立 (standing)
- 走路 (walking)
- 跑步 (running)
- 坐下 (sitting)
- 举手 (raising_hand)
- 看书 (reading)
- 写字 (writing)

## 技术栈

- **姿态估计**: MediaPipe Pose (OpenPose 替代方案)
- **行为识别**: 基于关键点序列的机器学习分类
- **图像处理**: OpenCV
- **深度学习**: PyTorch

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！


