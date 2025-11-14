"""
人体姿态估计模块
使用 MediaPipe Pose 进行人体姿态估计
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
import config
import mediapipe as mp


class PoseEstimator:
    """人体姿态估计器"""
    
    def __init__(self, 
                 min_detection_confidence: float = None,
                 min_tracking_confidence: float = None):
        """
        初始化姿态估计器
        
        Args:
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
        """
        self.min_detection_confidence = (
            min_detection_confidence or config.MIN_DETECTION_CONFIDENCE
        )
        self.min_tracking_confidence = (
            min_tracking_confidence or config.MIN_TRACKING_CONFIDENCE
        )
        
        # 初始化 MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            model_complexity=1  # 0, 1, 2 (2最准确但最慢)
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def estimate(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        估计图像中的人体姿态
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            关键点数组，形状为 (num_keypoints, 3) 或 None
            每行包含 [x, y, confidence]
        """
        # 转换图像格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # 进行姿态估计
        results = self.pose.process(image_rgb)
        
        # 提取关键点
        if results.pose_landmarks:
            keypoints = self._extract_keypoints(results.pose_landmarks, image.shape)
            return keypoints
        return None
    
    def _extract_keypoints(self, landmarks, image_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        从 MediaPipe 结果中提取关键点
        
        Args:
            landmarks: MediaPipe 姿态关键点
            image_shape: 图像形状 (height, width, channels)
            
        Returns:
            关键点数组 (num_keypoints, 3)
        """
        height, width = image_shape[:2]
        keypoints = []
        
        # MediaPipe Pose 有33个关键点，我们选择18个主要关键点
        # 提取18个主要关键点用于行为识别
        keypoint_indices = [
            0,   # nose
            2,   # left_eye
            5,   # right_eye
            7,   # left_ear
            8,   # right_ear
            11,  # left_shoulder
            12,  # right_shoulder
            13,  # left_elbow
            14,  # right_elbow
            15,  # left_wrist
            16,  # right_wrist
            23,  # left_hip
            24,  # right_hip
            25,  # left_knee
            26,  # right_knee
            27,  # left_ankle
            28   # right_ankle
        ]
        
        for idx in keypoint_indices:
            if idx < len(landmarks.landmark):
                landmark = landmarks.landmark[idx]
                x = landmark.x * width
                y = landmark.y * height
                confidence = landmark.visibility
                keypoints.append([x, y, confidence])
            else:
                keypoints.append([0, 0, 0])
        
        return np.array(keypoints, dtype=np.float32)
    
    def draw_landmarks(self, image: np.ndarray, landmarks) -> np.ndarray:
        """
        在图像上绘制姿态关键点和骨架
        
        Args:
            image: 输入图像
            landmarks: MediaPipe 姿态关键点
            
        Returns:
            绘制后的图像
        """
        annotated_image = image.copy()
        self.mp_drawing.draw_landmarks(
            annotated_image,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=2
            ),
            self.mp_drawing.DrawingSpec(
                color=(255, 0, 0), thickness=2
            )
        )
        return annotated_image
    
    def __del__(self):
        """释放资源"""
        if hasattr(self, 'pose') and self.pose is not None:
            try:
                self.pose.close()
            except:
                # 资源可能已经被释放，忽略错误
                pass


