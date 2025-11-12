"""
测试摄像头可用性
"""
import cv2
import sys

def test_cameras():
    """测试所有可用的摄像头索引"""
    print("正在检测可用的摄像头...")
    print("=" * 60)
    
    available_cameras = []
    
    # 测试索引 0-9
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # 尝试读取一帧来确认摄像头真的可用
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"[OK] 摄像头索引 {i}: 可用 ({width}x{height})")
                available_cameras.append(i)
            else:
                print(f"[X] 摄像头索引 {i}: 打开但无法读取")
        else:
            print(f"[X] 摄像头索引 {i}: 不可用")
        cap.release()
    
    print("=" * 60)
    
    if available_cameras:
        print(f"\n找到 {len(available_cameras)} 个可用摄像头")
        print(f"可用的摄像头索引: {available_cameras}")
        print(f"\n建议使用索引: {available_cameras[0]}")
        print(f"\n运行命令: python main.py --input {available_cameras[0]}")
        
        # 测试打开第一个可用摄像头
        print(f"\n正在测试摄像头 {available_cameras[0]}...")
        cap = cv2.VideoCapture(available_cameras[0])
        if cap.isOpened():
            print("摄像头测试成功！")
            print("按任意键关闭测试窗口...")
            
            # 显示几帧测试
            for i in range(30):
                ret, frame = cap.read()
                if ret:
                    cv2.imshow('摄像头测试 - 按任意键退出', frame)
                    if cv2.waitKey(1) & 0xFF != 255:
                        break
                else:
                    break
            
            cv2.destroyAllWindows()
            cap.release()
            print("测试完成！")
        else:
            print("警告：摄像头打开失败")
    else:
        print("\n未找到可用的摄像头！")
        print("\n可能的原因：")
        print("1. 摄像头未连接")
        print("2. 摄像头被其他程序占用")
        print("3. 摄像头驱动问题")
        print("\n建议：")
        print("- 检查摄像头是否连接")
        print("- 关闭其他使用摄像头的程序（如 Skype、Zoom 等）")
        print("- 尝试使用视频文件代替：python main.py --input video.mp4")

if __name__ == '__main__':
    test_cameras()

