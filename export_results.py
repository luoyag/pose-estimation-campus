"""
结果导出模块
将姿态估计和行为识别结果导出为JSON或CSV格式
"""
import json
import csv
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
import os


class ResultExporter:
    """结果导出器"""
    
    def __init__(self, output_dir: str = "results"):
        """
        初始化结果导出器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []
    
    def add_frame_result(self,
                        frame_id: int,
                        timestamp: float,
                        keypoints: Optional[np.ndarray],
                        behavior: str,
                        confidence: float,
                        image_path: Optional[str] = None):
        """
        添加一帧的结果
        
        Args:
            frame_id: 帧ID
            timestamp: 时间戳（秒）
            keypoints: 关键点数组
            behavior: 行为类别
            confidence: 置信度
            image_path: 图像路径（可选）
        """
        result = {
            'frame_id': frame_id,
            'timestamp': timestamp,
            'behavior': behavior,
            'confidence': float(confidence),
            'image_path': image_path
        }
        
        # 添加关键点信息
        if keypoints is not None:
            keypoints_list = []
            for i, (x, y, conf) in enumerate(keypoints):
                keypoints_list.append({
                    'id': i,
                    'x': float(x),
                    'y': float(y),
                    'confidence': float(conf)
                })
            result['keypoints'] = keypoints_list
        else:
            result['keypoints'] = None
        
        self.results.append(result)
    
    def export_json(self, filename: Optional[str] = None) -> str:
        """
        导出为JSON格式
        
        Args:
            filename: 输出文件名（可选）
            
        Returns:
            输出文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"results_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        output_data = {
            'export_time': datetime.now().isoformat(),
            'total_frames': len(self.results),
            'results': self.results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def export_csv(self, filename: Optional[str] = None) -> str:
        """
        导出为CSV格式
        
        Args:
            filename: 输出文件名（可选）
            
        Returns:
            输出文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"results_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入表头
            header = ['frame_id', 'timestamp', 'behavior', 'confidence']
            # 添加关键点列
            for i in range(18):  # 18个关键点
                header.extend([f'kp_{i}_x', f'kp_{i}_y', f'kp_{i}_conf'])
            writer.writerow(header)
            
            # 写入数据
            for result in self.results:
                row = [
                    result['frame_id'],
                    result['timestamp'],
                    result['behavior'],
                    result['confidence']
                ]
                
                # 添加关键点数据
                if result['keypoints']:
                    for kp in result['keypoints']:
                        row.extend([kp['x'], kp['y'], kp['confidence']])
                else:
                    row.extend([None, None, None] * 18)
                
                writer.writerow(row)
        
        return filepath
    
    def export_summary(self, filename: Optional[str] = None) -> str:
        """
        导出统计摘要
        
        Args:
            filename: 输出文件名（可选）
            
        Returns:
            输出文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"summary_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # 统计信息
        behaviors = [r['behavior'] for r in self.results]
        unique_behaviors, counts = np.unique(behaviors, return_counts=True)
        
        behavior_stats = {}
        for behavior, count in zip(unique_behaviors, counts):
            behavior_stats[behavior] = {
                'count': int(count),
                'percentage': float(count / len(self.results) * 100)
            }
        
        confidences = [r['confidence'] for r in self.results]
        
        summary = {
            'export_time': datetime.now().isoformat(),
            'total_frames': len(self.results),
            'duration_seconds': self.results[-1]['timestamp'] - self.results[0]['timestamp'] if len(self.results) > 1 else 0,
            'behavior_statistics': behavior_stats,
            'confidence_statistics': {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences))
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def clear(self):
        """清空结果"""
        self.results = []


def export_video_results(video_path: str,
                        output_path: str,
                        format: str = 'json'):
    """
    从视频导出结果（需要先处理视频）
    
    注意：这个函数需要配合main.py使用，或者需要重新处理视频
    
    Args:
        video_path: 视频路径
        output_path: 输出路径
        format: 导出格式（'json' 或 'csv'）
    """
    # 这是一个示例函数，实际使用时需要集成到main.py中
    pass

