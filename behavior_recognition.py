"""
行为识别模块
基于关键点序列识别校园场景中的行为
"""
import numpy as np
from typing import List, Optional, Tuple
from collections import deque
import config
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os


class BehaviorRecognizer:
    """行为识别器"""
    
    def __init__(self, 
                 sequence_length: int = None,
                 model_path: Optional[str] = None):
        """
        初始化行为识别器
        
        Args:
            sequence_length: 用于行为识别的关键点序列长度
            model_path: 预训练模型路径（如果存在）
        """
        self.sequence_length = sequence_length or config.BEHAVIOR_SEQUENCE_LENGTH
        self.keypoint_sequence = deque(maxlen=self.sequence_length)  # 存储特征序列
        self.raw_keypoint_sequence = deque(maxlen=self.sequence_length)  # 存储原始关键点序列
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # 加载预训练模型（如果存在）
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # 初始化默认模型
            self._init_default_model()
    
    def _init_default_model(self):
        """初始化默认的随机森林分类器"""
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False
    
    def extract_features(self, keypoints: np.ndarray) -> np.ndarray:
        """
        从关键点中提取特征
        
        Args:
            keypoints: 关键点数组 (num_keypoints, 3)
            
        Returns:
            特征向量
        """
        if keypoints is None or len(keypoints) == 0:
            return np.zeros(54)  # 默认特征维度
        
        features = []
        
        # 1. 关键点坐标（归一化）
        center_x = np.mean(keypoints[:, 0])
        center_y = np.mean(keypoints[:, 1])
        
        # 相对坐标
        relative_keypoints = keypoints[:, :2] - np.array([center_x, center_y])
        features.extend(relative_keypoints.flatten())
        
        # 2. 关键点置信度
        features.extend(keypoints[:, 2])
        
        # 3. 身体部位角度
        angles = self._calculate_angles(keypoints)
        features.extend(angles)
        
        # 4. 身体部位距离
        distances = self._calculate_distances(keypoints)
        features.extend(distances)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_angles(self, keypoints: np.ndarray) -> List[float]:
        """计算身体部位角度"""
        angles = []
        
        # 肩部角度
        if len(keypoints) > 11:
            left_shoulder = keypoints[5]  # left_shoulder
            right_shoulder = keypoints[6]  # right_shoulder
            if left_shoulder[2] > 0.5 and right_shoulder[2] > 0.5:
                angle = np.arctan2(
                    right_shoulder[1] - left_shoulder[1],
                    right_shoulder[0] - left_shoulder[0]
                )
                angles.append(angle)
            else:
                angles.append(0.0)
        
        # 肘部角度（左臂）
        if len(keypoints) > 8:
            left_shoulder = keypoints[5]
            left_elbow = keypoints[7]
            left_wrist = keypoints[9]
            if all(kp[2] > 0.5 for kp in [left_shoulder, left_elbow, left_wrist]):
                v1 = left_elbow[:2] - left_shoulder[:2]
                v2 = left_wrist[:2] - left_elbow[:2]
                angle = np.arccos(
                    np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6), -1, 1)
                )
                angles.append(angle)
            else:
                angles.append(0.0)
        
        # 肘部角度（右臂）
        if len(keypoints) > 9:
            right_shoulder = keypoints[6]
            right_elbow = keypoints[8]
            right_wrist = keypoints[10]
            if all(kp[2] > 0.5 for kp in [right_shoulder, right_elbow, right_wrist]):
                v1 = right_elbow[:2] - right_shoulder[:2]
                v2 = right_wrist[:2] - right_elbow[:2]
                angle = np.arccos(
                    np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6), -1, 1)
                )
                angles.append(angle)
            else:
                angles.append(0.0)
        
        # 膝盖角度（左腿）
        if len(keypoints) > 13:
            left_hip = keypoints[11]
            left_knee = keypoints[13]
            left_ankle = keypoints[15]
            if all(kp[2] > 0.5 for kp in [left_hip, left_knee, left_ankle]):
                v1 = left_knee[:2] - left_hip[:2]
                v2 = left_ankle[:2] - left_knee[:2]
                angle = np.arccos(
                    np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6), -1, 1)
                )
                angles.append(angle)
            else:
                angles.append(0.0)
        
        # 膝盖角度（右腿）
        if len(keypoints) > 14:
            right_hip = keypoints[12]
            right_knee = keypoints[14]
            right_ankle = keypoints[16]
            if all(kp[2] > 0.5 for kp in [right_hip, right_knee, right_ankle]):
                v1 = right_knee[:2] - right_hip[:2]
                v2 = right_ankle[:2] - right_knee[:2]
                angle = np.arccos(
                    np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6), -1, 1)
                )
                angles.append(angle)
            else:
                angles.append(0.0)
        
        return angles if angles else [0.0] * 5
    
    def _calculate_distances(self, keypoints: np.ndarray) -> List[float]:
        """计算身体部位距离"""
        distances = []
        
        if len(keypoints) < 17:
            return [0.0] * 10
        
        # 肩宽
        if keypoints[5][2] > 0.5 and keypoints[6][2] > 0.5:
            distances.append(np.linalg.norm(keypoints[5][:2] - keypoints[6][:2]))
        else:
            distances.append(0.0)
        
        # 髋宽
        if keypoints[11][2] > 0.5 and keypoints[12][2] > 0.5:
            distances.append(np.linalg.norm(keypoints[11][:2] - keypoints[12][:2]))
        else:
            distances.append(0.0)
        
        # 左臂长度
        if all(kp[2] > 0.5 for kp in [keypoints[5], keypoints[7], keypoints[9]]):
            arm_length = (np.linalg.norm(keypoints[5][:2] - keypoints[7][:2]) +
                         np.linalg.norm(keypoints[7][:2] - keypoints[9][:2]))
            distances.append(arm_length)
        else:
            distances.append(0.0)
        
        # 右臂长度
        if all(kp[2] > 0.5 for kp in [keypoints[6], keypoints[8], keypoints[10]]):
            arm_length = (np.linalg.norm(keypoints[6][:2] - keypoints[8][:2]) +
                         np.linalg.norm(keypoints[8][:2] - keypoints[10][:2]))
            distances.append(arm_length)
        else:
            distances.append(0.0)
        
        # 左腿长度
        if all(kp[2] > 0.5 for kp in [keypoints[11], keypoints[13], keypoints[15]]):
            leg_length = (np.linalg.norm(keypoints[11][:2] - keypoints[13][:2]) +
                         np.linalg.norm(keypoints[13][:2] - keypoints[15][:2]))
            distances.append(leg_length)
        else:
            distances.append(0.0)
        
        # 右腿长度
        if all(kp[2] > 0.5 for kp in [keypoints[12], keypoints[14], keypoints[16]]):
            leg_length = (np.linalg.norm(keypoints[12][:2] - keypoints[14][:2]) +
                         np.linalg.norm(keypoints[14][:2] - keypoints[16][:2]))
            distances.append(leg_length)
        else:
            distances.append(0.0)
        
        # 身体高度（从肩膀到脚踝的平均值）
        if (keypoints[5][2] > 0.5 and keypoints[6][2] > 0.5 and
            keypoints[15][2] > 0.5 and keypoints[16][2] > 0.5):
            shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
            ankle_y = (keypoints[15][1] + keypoints[16][1]) / 2
            distances.append(abs(shoulder_y - ankle_y))
        else:
            distances.append(0.0)
        
        # 手腕到肩膀的距离（左）
        if keypoints[5][2] > 0.5 and keypoints[9][2] > 0.5:
            distances.append(np.linalg.norm(keypoints[5][:2] - keypoints[9][:2]))
        else:
            distances.append(0.0)
        
        # 手腕到肩膀的距离（右）
        if keypoints[6][2] > 0.5 and keypoints[10][2] > 0.5:
            distances.append(np.linalg.norm(keypoints[6][:2] - keypoints[10][:2]))
        else:
            distances.append(0.0)
        
        # 手腕高度差（判断是否举手）
        if keypoints[9][2] > 0.5 and keypoints[10][2] > 0.5:
            distances.append(abs(keypoints[9][1] - keypoints[10][1]))
        else:
            distances.append(0.0)
        
        return distances[:10]  # 确保返回10个距离特征
    
    def recognize(self, keypoints: Optional[np.ndarray]) -> Tuple[str, float]:
        """
        识别行为
        
        Args:
            keypoints: 当前帧的关键点
            
        Returns:
            (行为类别, 置信度)
        """
        if keypoints is None:
            return "unknown", 0.0
        
        # 提取特征
        features = self.extract_features(keypoints)
        
        # 添加到序列
        self.keypoint_sequence.append(features)
        self.raw_keypoint_sequence.append(keypoints.copy() if keypoints is not None else None)
        
        # 如果序列未满，使用简单规则判断
        if len(self.keypoint_sequence) < self.sequence_length:
            return self._rule_based_recognition(keypoints)
        
        # 使用模型识别
        if self.is_trained:
            sequence_features = np.array(self.keypoint_sequence).flatten()
            sequence_features = sequence_features.reshape(1, -1)
            sequence_features = self.scaler.transform(sequence_features)
            
            prediction = self.model.predict(sequence_features)[0]
            probability = self.model.predict_proba(sequence_features)[0]
            confidence = np.max(probability)
            
            if confidence < config.BEHAVIOR_CONFIDENCE_THRESHOLD:
                return self._rule_based_recognition(keypoints)
            
            return prediction, confidence
        else:
            # 如果模型未训练，使用规则识别
            return self._rule_based_recognition(keypoints)
    
    def _rule_based_recognition(self, keypoints: np.ndarray) -> Tuple[str, float]:
        """
        基于规则的简单行为识别（用于模型未训练时）
        
        Args:
            keypoints: 关键点数组
            
        Returns:
            (行为类别, 置信度)
        """
        if keypoints is None or len(keypoints) < 17:
            return "unknown", 0.0
        
        # 检查关键点置信度
        valid_keypoints = keypoints[keypoints[:, 2] > 0.5]
        if len(valid_keypoints) < 10:
            return "unknown", 0.3
        
        # 判断是否坐下（髋部高度接近膝盖高度）
        if (keypoints[11][2] > 0.5 and keypoints[12][2] > 0.5 and
            keypoints[13][2] > 0.5 and keypoints[14][2] > 0.5):
            hip_y = (keypoints[11][1] + keypoints[12][1]) / 2
            knee_y = (keypoints[13][1] + keypoints[14][1]) / 2
            if abs(hip_y - knee_y) < 50:  # 阈值可调
                return "sitting", 0.7
        
        # 判断是否举手（手腕高于肩膀）
        if (keypoints[5][2] > 0.5 and keypoints[6][2] > 0.5 and
            keypoints[9][2] > 0.5 and keypoints[10][2] > 0.5):
            shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
            wrist_y = min(keypoints[9][1], keypoints[10][1])
            if wrist_y < shoulder_y - 50:  # 手腕明显高于肩膀
                return "raising_hand", 0.7
        
        # 判断是否跑步或走路（通过脚踝位置变化判断）
        if len(self.raw_keypoint_sequence) >= 5:
            recent_ankle_y = []
            for kp_seq in list(self.raw_keypoint_sequence)[-5:]:
                if kp_seq is not None and len(kp_seq) >= 17:
                    # 提取左右脚踝的y坐标平均值
                    left_ankle_y = kp_seq[15][1] if kp_seq[15][2] > 0.5 else None
                    right_ankle_y = kp_seq[16][1] if kp_seq[16][2] > 0.5 else None
                    
                    if left_ankle_y is not None and right_ankle_y is not None:
                        avg_ankle_y = (left_ankle_y + right_ankle_y) / 2
                        recent_ankle_y.append(avg_ankle_y)
                    elif left_ankle_y is not None:
                        recent_ankle_y.append(left_ankle_y)
                    elif right_ankle_y is not None:
                        recent_ankle_y.append(right_ankle_y)
            
            if len(recent_ankle_y) >= 3:
                ankle_variance = np.var(recent_ankle_y)
                if ankle_variance > 100:  # 脚踝位置变化大，可能是跑步
                    return "running", 0.6
                elif 30 < ankle_variance <= 100:  # 脚踝位置变化中等，可能是走路
                    return "walking", 0.6
        
        # 默认站立
        return "standing", 0.5
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        训练行为识别模型
        
        Args:
            X: 特征数组 (n_samples, n_features)
            y: 标签数组 (n_samples,)
        """
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练模型
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def save_model(self, model_path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'is_trained': self.is_trained
            }, f)
    
    def load_model(self, model_path: str):
        """加载模型"""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.is_trained = data['is_trained']


