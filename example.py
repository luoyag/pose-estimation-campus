"""
使用示例
演示如何使用姿态估计和行为识别功能
"""
import cv2
from pose_estimation import PoseEstimator
from behavior_recognition import BehaviorRecognizer
from visualization import Visualizer


def example_single_image():
    """单张图像处理示例"""
    print("=" * 60)
    print("单张图像处理示例")
    print("=" * 60)
    
    # 初始化组件
    pose_estimator = PoseEstimator()
    behavior_recognizer = BehaviorRecognizer()
    visualizer = Visualizer()
    
    # 读取图像（请替换为你的图像路径）
    image_path = "example_image.jpg"
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            print("请将 example_image.jpg 放在项目根目录，或修改 image_path")
            return
    except Exception as e:
        print(f"读取图像时出错: {e}")
        return
    
    # 姿态估计
    print("\n进行姿态估计...")
    keypoints = pose_estimator.estimate(image)
    
    if keypoints is not None:
        print(f"检测到 {len(keypoints)} 个关键点")
        
        # 行为识别
        print("进行行为识别...")
        behavior, confidence = behavior_recognizer.recognize(keypoints)
        print(f"识别结果: {behavior}, 置信度: {confidence:.2f}")
        
        # 可视化
        print("生成可视化结果...")
        annotated_image = visualizer.draw_all(
            image, keypoints, behavior, confidence
        )
        
        # 保存结果
        output_path = "output_example.jpg"
        cv2.imwrite(output_path, annotated_image)
        print(f"\n结果已保存到: {output_path}")
        
        # 显示结果
        cv2.imshow('结果', annotated_image)
        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("未检测到人体姿态")


def example_webcam():
    """摄像头实时处理示例"""
    print("=" * 60)
    print("摄像头实时处理示例")
    print("=" * 60)
    print("按 'q' 键退出")
    print()
    
    # 初始化组件
    pose_estimator = PoseEstimator()
    behavior_recognizer = BehaviorRecognizer()
    visualizer = Visualizer()
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    print("开始处理...")
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 姿态估计
            keypoints = pose_estimator.estimate(frame)
            
            # 行为识别
            if keypoints is not None:
                behavior, confidence = behavior_recognizer.recognize(keypoints)
            else:
                behavior, confidence = "unknown", 0.0
            
            # 可视化
            annotated_frame = visualizer.draw_all(
                frame, keypoints, behavior, confidence
            )
            
            # 显示
            cv2.imshow('实时姿态估计与行为识别', annotated_frame)
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"已处理 {frame_count} 帧")
    
    except KeyboardInterrupt:
        print("\n处理被中断")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"共处理 {frame_count} 帧")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'webcam':
        example_webcam()
    else:
        example_single_image()
        print("\n提示：运行 'python example.py webcam' 可以使用摄像头实时处理")


