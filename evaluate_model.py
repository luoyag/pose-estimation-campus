"""
模型评估脚本
用于评估行为识别模型的性能指标（准确率、召回率、F1分数等）
"""
import numpy as np
import pickle
import os
import argparse
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)
from behavior_recognition import BehaviorRecognizer
from pose_estimation import PoseEstimator
import cv2
import config
from tqdm import tqdm
import json


def load_test_data(data_dir: str):
    """
    加载测试数据
    
    Args:
        data_dir: 测试数据目录
        
    Returns:
        (X, y): 特征数组和标签数组
    """
    X = []
    y = []
    
    # 遍历数据目录
    for filename in os.listdir(data_dir):
        if filename.endswith('.npz'):
            filepath = os.path.join(data_dir, filename)
            data = np.load(filepath)
            
            features = data['features']
            label = data['label']
            
            # 如果features是序列，需要展平
            if len(features.shape) > 2:
                # 处理序列数据
                for seq in features:
                    X.append(seq.flatten())
                    y.append(label)
            else:
                X.append(features.flatten())
                y.append(label)
    
    return np.array(X), np.array(y)


def evaluate_from_video(video_path: str, 
                       model_path: str,
                       ground_truth_label: str = None):
    """
    从视频评估模型性能
    
    Args:
        video_path: 视频文件路径
        model_path: 模型路径
        ground_truth_label: 真实标签（如果已知）
    """
    # 加载模型
    recognizer = BehaviorRecognizer(model_path=model_path)
    pose_estimator = PoseEstimator()
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        return
    
    predictions = []
    confidences = []
    frame_count = 0
    
    print(f"正在处理视频: {video_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 估计姿态
        keypoints = pose_estimator.estimate(frame)
        
        if keypoints is not None:
            behavior, confidence = recognizer.recognize(keypoints)
            predictions.append(behavior)
            confidences.append(confidence)
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"已处理 {frame_count} 帧")
    
    cap.release()
    
    # 统计结果
    if predictions:
        unique, counts = np.unique(predictions, return_counts=True)
        print(f"\n预测结果统计:")
        for pred, count in zip(unique, counts):
            print(f"  {pred}: {count} 帧 ({count/len(predictions)*100:.1f}%)")
        
        avg_confidence = np.mean(confidences)
        print(f"\n平均置信度: {avg_confidence:.3f}")
        
        if ground_truth_label:
            accuracy = np.mean([p == ground_truth_label for p in predictions])
            print(f"\n准确率: {accuracy:.3f} (真实标签: {ground_truth_label})")


def evaluate_model(model_path: str, 
                  test_data_dir: str = None,
                  X_test: np.ndarray = None,
                  y_test: np.ndarray = None):
    """
    评估模型性能
    
    Args:
        model_path: 模型路径
        test_data_dir: 测试数据目录
        X_test: 测试特征（如果直接提供）
        y_test: 测试标签（如果直接提供）
    """
    print("=" * 60)
    print("模型评估")
    print("=" * 60)
    
    # 加载模型
    print(f"\n加载模型: {model_path}")
    recognizer = BehaviorRecognizer(model_path=model_path)
    
    if not recognizer.is_trained:
        print("错误：模型未训练")
        return
    
    # 加载测试数据
    if X_test is None or y_test is None:
        if test_data_dir is None:
            print("错误：需要提供测试数据目录或测试数据")
            return
        
        print(f"\n加载测试数据: {test_data_dir}")
        X_test, y_test = load_test_data(test_data_dir)
    
    print(f"测试样本数: {len(X_test)}")
    print(f"类别数: {len(np.unique(y_test))}")
    
    # 准备数据
    # 如果特征维度不匹配，需要调整
    if len(X_test.shape) == 1:
        X_test = X_test.reshape(1, -1)
    
    # 标准化
    X_test_scaled = recognizer.scaler.transform(X_test)
    
    # 预测
    print("\n进行预测...")
    y_pred = recognizer.model.predict(X_test_scaled)
    y_pred_proba = recognizer.model.predict_proba(X_test_scaled)
    
    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"准确率 (Accuracy):  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"精确率 (Precision): {precision:.4f} ({precision*100:.2f}%)")
    print(f"召回率 (Recall):    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1 分数:            {f1:.4f} ({f1*100:.2f}%)")
    
    # 分类报告
    print("\n" + "=" * 60)
    print("详细分类报告")
    print("=" * 60)
    print(classification_report(y_test, y_pred, 
                               target_names=config.BEHAVIOR_CLASSES,
                               zero_division=0))
    
    # 混淆矩阵
    print("\n" + "=" * 60)
    print("混淆矩阵")
    print("=" * 60)
    cm = confusion_matrix(y_test, y_pred, labels=config.BEHAVIOR_CLASSES)
    print("\n预测类别")
    print(" " * 15, end="")
    for label in config.BEHAVIOR_CLASSES:
        print(f"{label[:8]:>10}", end="")
    print()
    
    for i, label in enumerate(config.BEHAVIOR_CLASSES):
        print(f"{label[:14]:<15}", end="")
        for j in range(len(config.BEHAVIOR_CLASSES)):
            print(f"{cm[i][j]:>10}", end="")
        print()
    
    # 保存结果
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(
            y_test, y_pred, 
            target_names=config.BEHAVIOR_CLASSES,
            output_dict=True,
            zero_division=0
        )
    }
    
    results_path = model_path.replace('.pkl', '_evaluation.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n评估结果已保存到: {results_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='评估行为识别模型性能'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='模型路径'
    )
    parser.add_argument(
        '--test-data-dir',
        type=str,
        default=None,
        help='测试数据目录'
    )
    parser.add_argument(
        '--video',
        type=str,
        default=None,
        help='测试视频路径（用于单视频评估）'
    )
    parser.add_argument(
        '--label',
        type=str,
        default=None,
        help='视频的真实标签（与--video一起使用）'
    )
    
    args = parser.parse_args()
    
    if args.video:
        evaluate_from_video(args.video, args.model, args.label)
    else:
        evaluate_model(args.model, args.test_data_dir)


if __name__ == '__main__':
    main()

