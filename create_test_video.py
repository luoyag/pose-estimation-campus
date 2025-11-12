"""
创建一个测试视频用于演示
"""
import cv2
import numpy as np

def create_test_video(output_path="test_video.mp4", duration=10, fps=30):
    """
    创建一个简单的测试视频
    
    Args:
        output_path: 输出视频路径
        duration: 视频时长（秒）
        fps: 帧率
    """
    print(f"正在创建测试视频: {output_path}")
    print(f"时长: {duration}秒, 帧率: {fps} FPS")
    
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    for i in range(total_frames):
        # 创建一个简单的动画帧
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 绘制一个移动的圆圈
        center_x = int(width / 2 + 100 * np.sin(2 * np.pi * i / total_frames))
        center_y = int(height / 2 + 100 * np.cos(2 * np.pi * i / total_frames))
        
        cv2.circle(frame, (center_x, center_y), 50, (0, 255, 0), -1)
        
        # 添加文字
        text = f"Test Video Frame {i+1}/{total_frames}"
        cv2.putText(frame, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        
        if (i + 1) % 30 == 0:
            print(f"已生成 {i+1}/{total_frames} 帧")
    
    out.release()
    print(f"\n测试视频创建完成: {output_path}")
    print(f"可以使用以下命令测试: python main.py --input {output_path}")

if __name__ == '__main__':
    create_test_video()

