"""
人体姿态估计模块
使用 MediaPipe Pose 作为 OpenPose 的替代方案
如果 MediaPipe 不可用，则使用 OpenCV 的简化版本
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
import config

# 尝试导入 MediaPipe，如果失败则使用 OpenCV 版本
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("警告：MediaPipe 未安装，将使用 OpenCV 简化版本进行姿态估计")
    print("提示：如果使用 Python 3.11 或 3.12，可以安装 mediapipe 获得更好的效果")


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
        
        # 初始化 MediaPipe Pose 或 OpenCV 版本
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                model_complexity=1  # 0, 1, 2 (2最准确但最慢)
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.use_mediapipe = True
        else:
            # 使用 OpenCV 简化版本
            self.use_mediapipe = False
            self.net = None
        
    def estimate(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        估计图像中的人体姿态
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            关键点数组，形状为 (num_keypoints, 3) 或 None
            每行包含 [x, y, confidence]
        """
        if self.use_mediapipe:
            # 使用 MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            
            # 进行姿态估计
            results = self.pose.process(image_rgb)
            
            # 提取关键点
            if results.pose_landmarks:
                keypoints = self._extract_keypoints(results.pose_landmarks, image.shape)
                return keypoints
            return None
        else:
            # 使用 OpenCV 简化版本
            return self._estimate_opencv(image)
    
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
        # 对应 OpenPose 的18个关键点
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
    
    def _estimate_opencv(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        使用 OpenCV 简化方法估计姿态（当 MediaPipe 不可用时）
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            关键点数组或 None
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用背景减除或轮廓检测
        # 这里使用一个简化的方法
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        if w < 50 or h < 100:
            return None
        
        # 基于边界框估计关键点
        return self._estimate_keypoints_from_bbox(x, y, w, h, image.shape)
    
    def _estimate_keypoints_from_bbox(self, x, y, w, h, image_shape):
        """从边界框估计关键点（简化方法）"""
        height, width = image_shape[:2]
        keypoints = []
        
        center_x = x + w / 2
        center_y = y + h / 2
        
        # 基于人体比例的关键点偏移
        keypoint_offsets = [
            (0, -0.4 * h),      # 0: nose
            (-0.1 * w, -0.35 * h),  # 1: left_eye
            (0.1 * w, -0.35 * h),   # 2: right_eye
            (-0.15 * w, -0.3 * h),  # 3: left_ear
            (0.15 * w, -0.3 * h),   # 4: right_ear
            (-0.2 * w, -0.1 * h),   # 5: left_shoulder
            (0.2 * w, -0.1 * h),    # 6: right_shoulder
            (-0.25 * w, 0.1 * h),   # 7: left_elbow
            (0.25 * w, 0.1 * h),    # 8: right_elbow
            (-0.3 * w, 0.3 * h),    # 9: left_wrist
            (0.3 * w, 0.3 * h),     # 10: right_wrist
            (-0.15 * w, 0.2 * h),   # 11: left_hip
            (0.15 * w, 0.2 * h),    # 12: right_hip
            (-0.15 * w, 0.5 * h),   # 13: left_knee
            (0.15 * w, 0.5 * h),    # 14: right_knee
            (-0.15 * w, 0.85 * h),  # 15: left_ankle
            (0.15 * w, 0.85 * h),   # 16: right_ankle
        ]
        
        for offset_x, offset_y in keypoint_offsets:
            kp_x = center_x + offset_x
            kp_y = center_y + offset_y
            kp_x = max(0, min(width - 1, kp_x))
            kp_y = max(0, min(height - 1, kp_y))
            confidence = 0.6  # 简化版本的置信度较低
            keypoints.append([kp_x, kp_y, confidence])
        
        return np.array(keypoints, dtype=np.float32)
    
    def __del__(self):
        """释放资源"""
        if hasattr(self, 'pose') and self.use_mediapipe:
            self.pose.close()


