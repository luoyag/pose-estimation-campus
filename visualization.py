"""
可视化模块
用于绘制姿态估计和行为识别结果
"""
import cv2
import numpy as np
from typing import Optional, Tuple
import config


class Visualizer:
    """可视化器"""
    
    def __init__(self):
        """初始化可视化器"""
        self.config = config.VISUALIZATION_CONFIG
    
    def draw_keypoints(self, 
                      image: np.ndarray, 
                      keypoints: np.ndarray) -> np.ndarray:
        """
        在图像上绘制关键点
        
        Args:
            image: 输入图像
            keypoints: 关键点数组 (num_keypoints, 3)
            
        Returns:
            绘制后的图像
        """
        if keypoints is None:
            return image
        
        annotated_image = image.copy()
        
        if self.config['show_keypoints']:
            for i, (x, y, conf) in enumerate(keypoints):
                if conf > config.POSE_CONFIDENCE_THRESHOLD:
                    cv2.circle(
                        annotated_image,
                        (int(x), int(y)),
                        5,
                        self.config['keypoint_color'],
                        -1
                    )
                    # 可选：显示关键点编号
                    # cv2.putText(annotated_image, str(i), (int(x), int(y)),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return annotated_image
    
    def draw_skeleton(self, 
                     image: np.ndarray, 
                     keypoints: np.ndarray) -> np.ndarray:
        """
        在图像上绘制骨架
        
        Args:
            image: 输入图像
            keypoints: 关键点数组 (num_keypoints, 3)
            
        Returns:
            绘制后的图像
        """
        if keypoints is None or len(keypoints) < 17:
            return image
        
        annotated_image = image.copy()
        
        if self.config['show_skeleton']:
            # 定义骨架连接（基于 OpenPose 18个关键点的连接）
            skeleton_connections = [
                # 头部
                (0, 1), (0, 2), (1, 3), (2, 4),  # 鼻子-眼睛-耳朵
                # 躯干
                (5, 6),  # 肩膀
                (5, 11), (6, 12),  # 肩膀到髋部
                (11, 12),  # 髋部
                # 左臂
                (5, 7), (7, 9),  # 左肩膀-左肘-左腕
                # 右臂
                (6, 8), (8, 10),  # 右肩膀-右肘-右腕
                # 左腿
                (11, 13), (13, 15),  # 左髋-左膝-左踝
                # 右腿
                (12, 14), (14, 16),  # 右髋-右膝-右踝
            ]
            
            for start_idx, end_idx in skeleton_connections:
                if (start_idx < len(keypoints) and end_idx < len(keypoints)):
                    start_kp = keypoints[start_idx]
                    end_kp = keypoints[end_idx]
                    
                    if (start_kp[2] > config.POSE_CONFIDENCE_THRESHOLD and
                        end_kp[2] > config.POSE_CONFIDENCE_THRESHOLD):
                        cv2.line(
                            annotated_image,
                            (int(start_kp[0]), int(start_kp[1])),
                            (int(end_kp[0]), int(end_kp[1])),
                            self.config['skeleton_color'],
                            2
                        )
        
        return annotated_image
    
    def draw_behavior_label(self, 
                           image: np.ndarray, 
                           behavior: str, 
                           confidence: float) -> np.ndarray:
        """
        在图像上绘制行为标签
        
        Args:
            image: 输入图像
            behavior: 行为类别
            confidence: 置信度
            
        Returns:
            绘制后的图像
        """
        if not self.config['show_behavior_label']:
            return image
        
        annotated_image = image.copy()
        
        # 行为类别中文映射
        behavior_zh = {
            'standing': '站立',
            'walking': '走路',
            'running': '跑步',
            'sitting': '坐下',
            'raising_hand': '举手',
            'reading': '看书',
            'writing': '写字',
            'unknown': '未知'
        }
        
        label_text = f"{behavior_zh.get(behavior, behavior)}: {confidence:.2f}"
        
        # 计算文本大小
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config['text_scale'],
            self.config['text_thickness']
        )
        
        # 绘制背景矩形
        cv2.rectangle(
            annotated_image,
            (10, 10),
            (20 + text_width, 30 + text_height),
            (0, 0, 0),
            -1
        )
        
        # 绘制文本
        cv2.putText(
            annotated_image,
            label_text,
            (15, 25 + text_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config['text_scale'],
            self.config['text_color'],
            self.config['text_thickness']
        )
        
        return annotated_image
    
    def draw_all(self, 
                image: np.ndarray, 
                keypoints: Optional[np.ndarray],
                behavior: str = "unknown",
                confidence: float = 0.0) -> np.ndarray:
        """
        绘制所有可视化元素
        
        Args:
            image: 输入图像
            keypoints: 关键点数组
            behavior: 行为类别
            confidence: 置信度
            
        Returns:
            绘制后的图像
        """
        annotated_image = image.copy()
        
        if keypoints is not None:
            annotated_image = self.draw_skeleton(annotated_image, keypoints)
            annotated_image = self.draw_keypoints(annotated_image, keypoints)
        
        annotated_image = self.draw_behavior_label(
            annotated_image, behavior, confidence
        )
        
        return annotated_image


