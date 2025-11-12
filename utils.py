"""
工具函数模块
"""
import cv2
import numpy as np
from typing import Union, Tuple, Optional
import os


def is_video_file(filepath: str) -> bool:
    """
    判断文件是否为视频文件
    
    Args:
        filepath: 文件路径
        
    Returns:
        是否为视频文件
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    return any(filepath.lower().endswith(ext) for ext in video_extensions)


def get_video_info(video_path: Union[str, int]) -> Optional[Tuple[int, int, int, float]]:
    """
    获取视频信息
    
    Args:
        video_path: 视频路径或摄像头索引
        
    Returns:
        (宽度, 高度, 总帧数, FPS) 或 None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    return (width, height, total_frames, fps)


def create_output_directory(output_path: str):
    """
    创建输出目录
    
    Args:
        output_path: 输出路径
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


def normalize_keypoints(keypoints: np.ndarray, 
                       image_shape: Tuple[int, int]) -> np.ndarray:
    """
    归一化关键点坐标到 [0, 1]
    
    Args:
        keypoints: 关键点数组
        image_shape: 图像形状 (height, width)
        
    Returns:
        归一化后的关键点
    """
    if keypoints is None or len(keypoints) == 0:
        return keypoints
    
    height, width = image_shape
    normalized = keypoints.copy()
    normalized[:, 0] = normalized[:, 0] / width
    normalized[:, 1] = normalized[:, 1] / height
    return normalized


def denormalize_keypoints(keypoints: np.ndarray,
                          image_shape: Tuple[int, int]) -> np.ndarray:
    """
    反归一化关键点坐标
    
    Args:
        keypoints: 归一化的关键点数组
        image_shape: 图像形状 (height, width)
        
    Returns:
        反归一化后的关键点
    """
    if keypoints is None or len(keypoints) == 0:
        return keypoints
    
    height, width = image_shape
    denormalized = keypoints.copy()
    denormalized[:, 0] = denormalized[:, 0] * width
    denormalized[:, 1] = denormalized[:, 1] * height
    return denormalized


