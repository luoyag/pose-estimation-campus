"""
配置文件
"""
import os

# 模型路径配置
MODEL_DIR = "models"
POSE_MODEL_PATH = os.path.join(MODEL_DIR, "pose_model")
BEHAVIOR_MODEL_PATH = os.path.join(MODEL_DIR, "behavior_model.pkl")

# 姿态估计配置
POSE_CONFIDENCE_THRESHOLD = 0.5
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# 行为识别配置
BEHAVIOR_SEQUENCE_LENGTH = 30  # 用于行为识别的关键点序列长度
BEHAVIOR_CONFIDENCE_THRESHOLD = 0.6

# 关键点索引（MediaPipe Pose 18个关键点）
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# 行为类别
BEHAVIOR_CLASSES = [
    'standing',      # 站立
    'walking',       # 走路
    'running',       # 跑步
    'sitting',       # 坐下
    'raising_hand',  # 举手
    'reading',       # 看书
    'writing'        # 写字
]

# 可视化配置
VISUALIZATION_CONFIG = {
    'show_keypoints': True,
    'show_skeleton': True,
    'show_behavior_label': True,
    'keypoint_color': (0, 255, 0),
    'skeleton_color': (255, 0, 0),
    'text_color': (255, 255, 255),
    'text_scale': 0.7,
    'text_thickness': 2
}

# 视频处理配置
VIDEO_CONFIG = {
    'fps': 30,
    'width': 640,
    'height': 480
}


