"""
主程序入口
基于 MediaPipe Pose 的校园场景人体姿态估计与行为识别
"""
import cv2
import argparse
import sys
import time
from tqdm import tqdm
from pose_estimation import PoseEstimator
from behavior_recognition import BehaviorRecognizer
from visualization import Visualizer
from utils import is_video_file, get_video_info, create_output_directory
import config


def process_video(input_path: str, 
                 output_path: str = None,
                 behavior_model_path: str = None,
                 show_preview: bool = True):
    """
    处理视频文件或摄像头输入
    
    Args:
        input_path: 输入路径（视频文件路径或摄像头索引）
        output_path: 输出视频路径（可选）
        behavior_model_path: 行为识别模型路径（可选）
        show_preview: 是否显示预览窗口
    """
    # 初始化组件
    print("初始化姿态估计器...")
    pose_estimator = PoseEstimator()
    
    print("初始化行为识别器...")
    behavior_recognizer = BehaviorRecognizer(
        model_path=behavior_model_path or config.BEHAVIOR_MODEL_PATH
    )
    
    print("初始化可视化器...")
    visualizer = Visualizer()
    
    # 保存原始输入路径，用于后续判断
    is_camera = isinstance(input_path, str) and input_path.isdigit()
    
    # 打开视频或摄像头
    if is_camera:
        camera_index = int(input_path)
        cap = cv2.VideoCapture(camera_index)
    else:
        cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"错误：无法打开输入源 {input_path}")
        print("\n可能的原因：")
        print("1. 摄像头未连接或被其他程序占用")
        print("2. 摄像头索引不正确（尝试其他索引，如 1, 2 等）")
        print("3. 视频文件路径错误或文件不存在")
        print("\n建议：")
        print("- 运行 'python test_camera.py' 检测可用摄像头")
        print("- 关闭其他使用摄像头的程序（如 Skype、Zoom、Teams 等）")
        print("- 如果使用视频文件，检查文件路径是否正确")
        print("- 尝试使用其他摄像头索引：python main.py --input 1")
        return
    
    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or config.VIDEO_CONFIG['fps']
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息: {width}x{height}, {fps} FPS, {total_frames} 帧")
    
    # 设置输出视频写入器
    writer = None
    if output_path:
        create_output_directory(output_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"输出视频将保存到: {output_path}")
    
    # 处理帧
    frame_count = 0
    should_exit = False
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0.0
    
    try:
        # 创建进度条（仅对视频文件）
        if total_frames > 0:
            pbar = tqdm(total=total_frames, desc="处理中")
        else:
            pbar = None
            # 摄像头模式下，提示如何退出
            if show_preview:
                print("\n提示：按 'Q' 或 'ESC' 键退出，或点击窗口关闭按钮")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                # 对于摄像头，ret为False可能是暂时的问题，可以尝试继续
                if is_camera:
                    continue
                else:
                    # 对于视频文件，ret为False表示结束
                    break
            
            # 姿态估计
            keypoints = pose_estimator.estimate(frame)
            
            # 行为识别
            if keypoints is not None:
                behavior, confidence = behavior_recognizer.recognize(keypoints)
            else:
                behavior, confidence = "unknown", 0.0
            
            # 计算FPS（每30帧更新一次）
            fps_counter += 1
            if fps_counter >= 30:
                fps_end_time = time.time()
                current_fps = fps_counter / (fps_end_time - fps_start_time)
                fps_counter = 0
                fps_start_time = fps_end_time
            
            # 可视化（传递当前FPS）
            annotated_frame = visualizer.draw_all(
                frame, keypoints, behavior, confidence, 
                fps=current_fps if current_fps > 0 else None
            )
            
            # 显示预览和处理按键事件
            if show_preview:
                cv2.imshow('姿态估计与行为识别', annotated_frame)
                
                # 检查窗口是否被关闭（cv2.getWindowProperty返回-1表示窗口被关闭）
                try:
                    if cv2.getWindowProperty('姿态估计与行为识别', cv2.WND_PROP_VISIBLE) < 1:
                        print("\n窗口已关闭")
                        should_exit = True
                        break
                except:
                    # 在某些系统上可能不支持这个属性
                    pass
                
                # 检查按键：Q键、ESC键或窗口关闭
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q') or key == 27:  # 27是ESC键
                    print("\n用户按下退出键（Q/ESC）")
                    should_exit = True
                    break
            
            # 写入输出视频
            if writer:
                writer.write(annotated_frame)
            
            frame_count += 1
            if pbar:
                pbar.update(1)
        
        if pbar:
            pbar.close()
        
        if should_exit:
            print(f"\n用户中断处理，共处理 {frame_count} 帧")
        else:
            print(f"\n处理完成！共处理 {frame_count} 帧")
        
    except KeyboardInterrupt:
        print("\n\n处理被用户中断（Ctrl+C）")
        should_exit = True
    except Exception as e:
        print(f"\n错误：{e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        print("\n正在释放资源...")
        
        # 释放视频捕获
        if 'cap' in locals():
            cap.release()
        
        # 释放视频写入器
        if writer:
            writer.release()
        
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        
        # 尝试关闭所有窗口（多次调用确保关闭）
        for _ in range(3):
            cv2.waitKey(1)
        
        # 释放姿态估计器资源
        if 'pose_estimator' in locals():
            if hasattr(pose_estimator, 'pose'):
                try:
                    if pose_estimator.pose is not None:
                        pose_estimator.pose.close()
                        pose_estimator.pose = None
                except:
                    # 资源可能已经被释放，忽略错误
                    pass
        
        print("资源已释放")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='基于 MediaPipe Pose 的校园场景人体姿态估计与行为识别'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='0',
        help='输入视频文件路径或摄像头索引（默认: 0）'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出视频文件路径（可选）'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='行为识别模型路径（可选）'
    )
    parser.add_argument(
        '--no-preview',
        action='store_true',
        help='不显示预览窗口'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("基于 MediaPipe Pose 的校园场景人体姿态估计与行为识别")
    print("=" * 60)
    print()
    
    # 处理视频
    process_video(
        input_path=args.input,
        output_path=args.output,
        behavior_model_path=args.model,
        show_preview=not args.no_preview
    )


if __name__ == '__main__':
    main()


