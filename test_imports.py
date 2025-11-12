"""
测试脚本：检查所有依赖是否正确安装
"""
import sys

def test_imports():
    """测试所有必要的导入"""
    print("=" * 60)
    print("测试依赖导入")
    print("=" * 60)
    
    errors = []
    
    # 测试标准库
    print("\n1. 测试标准库...")
    try:
        import argparse
        import os
        import pickle
        from collections import deque
        from typing import List, Tuple, Optional
        print("   [OK] 标准库导入成功")
    except ImportError as e:
        errors.append(f"标准库导入失败: {e}")
        print(f"   ✗ 标准库导入失败: {e}")
    
    # 测试第三方库
    print("\n2. 测试第三方库...")
    libraries = [
        ('cv2', 'opencv-python', True),  # 必需
        ('numpy', 'numpy', True),  # 必需
        ('mediapipe', 'mediapipe', False),  # 可选（Python 3.13不支持）
        ('sklearn', 'scikit-learn', True),  # 必需
        ('tqdm', 'tqdm', True),  # 必需
    ]
    
    for lib_name, pip_name, required in libraries:
        try:
            __import__(lib_name)
            print(f"   [OK] {lib_name} 导入成功")
        except ImportError:
            if required:
                errors.append(f"{lib_name} 未安装，请运行: pip install {pip_name}")
                print(f"   [X] {lib_name} 未安装，请运行: pip install {pip_name}")
            else:
                print(f"   [!] {lib_name} 未安装（可选），代码将使用替代方案")
    
    # 测试项目模块
    print("\n3. 测试项目模块...")
    modules = [
        'config',
        'pose_estimation',
        'behavior_recognition',
        'visualization',
        'utils',
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"   [OK] {module} 导入成功")
        except ImportError as e:
            errors.append(f"{module} 导入失败: {e}")
            print(f"   [X] {module} 导入失败: {e}")
        except Exception as e:
            errors.append(f"{module} 导入时出错: {e}")
            print(f"   [X] {module} 导入时出错: {e}")
    
    # 总结
    print("\n" + "=" * 60)
    if errors:
        print("发现以下问题：")
        for error in errors:
            print(f"  - {error}")
        print("\n请先解决这些问题后再运行主程序。")
        return False
    else:
        print("[OK] 所有依赖检查通过！代码可以运行。")
        print("\n你可以运行以下命令测试：")
        print("  python main.py --input 0          # 使用摄像头")
        print("  python example.py webcam          # 示例：摄像头")
        print("  python example.py                 # 示例：单张图像")
        return True

if __name__ == '__main__':
    success = test_imports()
    sys.exit(0 if success else 1)

