"""
可视化模块
用于绘制姿态估计和行为识别结果
"""
import cv2
import numpy as np
from typing import Optional, Tuple
import config
from PIL import Image, ImageDraw, ImageFont
import os


class Visualizer:
    """可视化器"""
    
    def __init__(self):
        """初始化可视化器"""
        self.config = config.VISUALIZATION_CONFIG
        # 尝试加载中文字体
        self.font_path = self._find_chinese_font()
    
    def _find_chinese_font(self):
        """查找系统中文字体"""
        # Windows系统常见中文字体路径
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
            "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
            "C:/Windows/Fonts/msyhbd.ttc",  # 微软雅黑 Bold
        ]
        
        # Linux系统常见中文字体路径
        if os.name != 'nt':  # 不是Windows
            font_paths.extend([
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ])
        
        for path in font_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _draw_chinese_text(self, img, text, position, font_size=20, color=(255, 255, 255)):
        """
        在图像上绘制中文文本（使用PIL）
        
        Args:
            img: OpenCV图像 (numpy数组)
            text: 要绘制的文本
            position: 文本位置 (x, y)
            font_size: 字体大小
            color: 文本颜色 (B, G, R)
            
        Returns:
            绘制后的图像
        """
        # 转换OpenCV图像为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 加载字体
        try:
            if self.font_path:
                font = ImageFont.truetype(self.font_path, font_size)
            else:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # 绘制文本（PIL使用RGB，需要转换颜色）
        text_color = (color[2], color[1], color[0])  # BGR to RGB
        draw.text(position, text, font=font, fill=text_color)
        
        # 转换回OpenCV格式
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img_cv
    
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
            # 定义骨架连接（基于 MediaPipe Pose 18个关键点的连接）
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
                           confidence: float,
                           fps: float = None) -> np.ndarray:
        """
        在图像上绘制行为标签
        
        Args:
            image: 输入图像
            behavior: 行为类别
            confidence: 置信度 (0-1之间的数值)
            fps: 帧率（可选）
            
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
        
        # 构建显示文本
        lines = []
        
        # 行为识别结果（置信度转换为百分比显示）
        behavior_text = behavior_zh.get(behavior, behavior)
        confidence_percent = confidence * 100  # 转换为百分比
        lines.append(f"行为: {behavior_text}")
        lines.append(f"置信度: {confidence_percent:.1f}%")
        
        # 显示FPS
        if fps is not None:
            lines.append(f"FPS: {fps:.1f}")
        
        # 提示信息
        lines.append("按 Q 或 ESC 退出")
        
        # 绘制背景矩形和文本
        y_offset = 15
        max_width = 0
        total_height = 0
        
        # 先计算所有文本的尺寸
        for line in lines:
            (text_width, text_height), baseline = cv2.getTextSize(
                line,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config['text_scale'],
                self.config['text_thickness']
            )
            max_width = max(max_width, text_width)
            total_height += text_height + 5
        
        # 绘制半透明背景矩形
        overlay = annotated_image.copy()
        cv2.rectangle(
            overlay,
            (10, 10),
            (25 + max_width, 20 + total_height),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.7, annotated_image, 0.3, 0, annotated_image)
        
        # 绘制文本（根据置信度选择颜色，使用PIL支持中文）
        y_pos = 25
        font_size = int(self.config['text_scale'] * 30)  # 转换为合适的字体大小
        
        for i, line in enumerate(lines):
            # 第一行（行为类别）使用不同颜色
            if i == 0:
                # 根据置信度选择颜色：绿色（高置信度）->黄色（中）->红色（低）
                if confidence > 0.7:
                    color = (0, 255, 0)  # 绿色
                elif confidence > 0.5:
                    color = (0, 255, 255)  # 黄色
                else:
                    color = (0, 0, 255)  # 红色
            elif i == 1:  # 置信度行
                color = self.config['text_color']
            elif i == len(lines) - 1:  # 最后一行（提示信息）
                color = (128, 128, 128)  # 灰色
            else:
                color = self.config['text_color']
            
            # 检查是否包含中文字符，如果有则使用PIL绘制
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in line)
            
            if has_chinese:
                # 使用PIL绘制中文
                annotated_image = self._draw_chinese_text(
                    annotated_image, 
                    line, 
                    (15, y_pos - int(font_size * 0.8)),  # 调整Y坐标使其与OpenCV对齐
                    font_size=font_size,
                    color=color
                )
                # 估算文本高度（用于下一行位置）
                text_height = int(font_size * 1.2)
            else:
                # 使用OpenCV绘制英文和数字
                cv2.putText(
                    annotated_image,
                    line,
                    (15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config['text_scale'],
                    color,
                    self.config['text_thickness']
                )
                
                (_, text_height), _ = cv2.getTextSize(
                    line,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config['text_scale'],
                    self.config['text_thickness']
                )
            
            y_pos += text_height + 5
        
        return annotated_image
    
    def draw_all(self, 
                image: np.ndarray, 
                keypoints: Optional[np.ndarray],
                behavior: str = "unknown",
                confidence: float = 0.0,
                fps: float = None) -> np.ndarray:
        """
        绘制所有可视化元素
        
        Args:
            image: 输入图像
            keypoints: 关键点数组
            behavior: 行为类别
            confidence: 置信度 (0-1之间的数值)
            fps: 帧率（可选）
            
        Returns:
            绘制后的图像
        """
        annotated_image = image.copy()
        
        if keypoints is not None:
            annotated_image = self.draw_skeleton(annotated_image, keypoints)
            annotated_image = self.draw_keypoints(annotated_image, keypoints)
        
        annotated_image = self.draw_behavior_label(
            annotated_image, behavior, confidence, fps
        )
        
        return annotated_image


