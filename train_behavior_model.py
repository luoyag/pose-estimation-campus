"""
行为识别模型训练脚本
用于训练和保存行为识别模型
"""
import numpy as np
import pickle
import os
from behavior_recognition import BehaviorRecognizer
from pose_estimation import PoseEstimator
import cv2
import config
from tqdm import tqdm
import argparse


def load_training_data(data_dir: str):
    """
    加载训练数据
    
    注意：这是一个示例函数，实际使用时需要根据你的数据格式进行修改
    数据格式应该是：
    - 每个视频/图像序列对应一个行为类别
    - 关键点数据保存为 numpy 数组
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        (X, y): 特征数组和标签数组
    """
    print("注意：这是一个示例训练脚本")
    print("你需要根据实际数据格式修改 load_training_data 函数")
    print()
    
    # 示例：生成模拟训练数据
    print("生成模拟训练数据用于演示...")
    X = []
    y = []
    
    # 为每个行为类别生成一些示例数据
    for behavior in config.BEHAVIOR_CLASSES:
        for _ in range(50):  # 每个类别50个样本
            # 生成随机关键点特征
            features = np.random.randn(54)  # 54维特征
            X.append(features)
            y.append(behavior)
    
    return np.array(X), np.array(y)


def collect_data_from_video(video_path: str, 
                           behavior_label: str,
                           output_dir: str = "training_data"):
    """
    从视频中收集训练数据
    
    Args:
        video_path: 视频文件路径
        behavior_label: 行为标签
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化姿态估计器
    pose_estimator = PoseEstimator()
    behavior_recognizer = BehaviorRecognizer()
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        return
    
    features_list = []
    frame_count = 0
    
    print(f"正在从视频 {video_path} 收集数据，标签: {behavior_label}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 估计姿态
        keypoints = pose_estimator.estimate(frame)
        
        if keypoints is not None:
            # 提取特征
            features = behavior_recognizer.extract_features(keypoints)
            features_list.append(features)
        
        frame_count += 1
        
        # 显示进度
        if frame_count % 30 == 0:
            print(f"已处理 {frame_count} 帧，收集到 {len(features_list)} 个样本")
    
    cap.release()
    
    # 保存数据
    if features_list:
        data_file = os.path.join(
            output_dir, 
            f"{behavior_label}_{len(features_list)}.npz"
        )
        np.savez(data_file, features=np.array(features_list), label=behavior_label)
        print(f"数据已保存到: {data_file}")
        print(f"共收集 {len(features_list)} 个样本")
    else:
        print("未收集到有效数据")


def train_model(data_dir: str = "training_data",
               model_path: str = None):
    """
    训练行为识别模型
    
    Args:
        data_dir: 训练数据目录
        model_path: 模型保存路径
    """
    print("=" * 60)
    print("开始训练行为识别模型")
    print("=" * 60)
    
    # 加载训练数据
    print("\n加载训练数据...")
    X, y = load_training_data(data_dir)
    
    print(f"训练样本数: {len(X)}")
    print(f"特征维度: {X.shape[1]}")
    print(f"类别数: {len(np.unique(y))}")
    print(f"类别: {np.unique(y)}")
    
    # 初始化识别器
    print("\n初始化行为识别器...")
    recognizer = BehaviorRecognizer()
    
    # 训练模型
    print("\n开始训练...")
    recognizer.train(X, y)
    
    # 保存模型
    model_path = model_path or config.BEHAVIOR_MODEL_PATH
    print(f"\n保存模型到: {model_path}")
    recognizer.save_model(model_path)
    
    print("\n训练完成！")
    
    # 评估模型（简单测试）
    print("\n模型评估:")
    X_scaled = recognizer.scaler.transform(X)
    predictions = recognizer.model.predict(X_scaled)
    accuracy = np.mean(predictions == y)
    print(f"训练准确率: {accuracy:.2%}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='训练行为识别模型'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='training_data',
        help='训练数据目录'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='模型保存路径'
    )
    parser.add_argument(
        '--collect',
        type=str,
        default=None,
        help='从视频收集数据（需要同时指定 --label）'
    )
    parser.add_argument(
        '--label',
        type=str,
        default=None,
        help='行为标签（与 --collect 一起使用）'
    )
    
    args = parser.parse_args()
    
    # 如果指定了收集数据
    if args.collect:
        if not args.label:
            print("错误：使用 --collect 时必须指定 --label")
            return
        collect_data_from_video(args.collect, args.label, args.data_dir)
    else:
        # 训练模型
        train_model(args.data_dir, args.model_path)


if __name__ == '__main__':
    main()


