# Python 版本切换指南

## 问题说明

MediaPipe 目前不支持 Python 3.13，建议使用 **Python 3.11** 或 **Python 3.12**。

## 切换 Python 版本的方法

### 方法 1：使用 pyenv（推荐，适用于 Windows/Mac/Linux）

#### Windows 用户：

1. **安装 pyenv-win**：
   ```powershell
   # 使用 Chocolatey
   choco install pyenv-win
   
   # 或使用 Git
   git clone https://github.com/pyenv-win/pyenv-win.git $HOME\.pyenv
   ```

2. **安装 Python 3.11**：
   ```powershell
   pyenv install 3.11.9
   pyenv local 3.11.9
   ```

3. **验证版本**：
   ```powershell
   python --version
   ```

#### Mac/Linux 用户：

1. **安装 pyenv**：
   ```bash
   # Mac
   brew install pyenv
   
   # Linux
   curl https://pyenv.run | bash
   ```

2. **安装 Python 3.11**：
   ```bash
   pyenv install 3.11.9
   pyenv local 3.11.9
   ```

### 方法 2：使用 conda（推荐，适用于所有平台）

1. **安装 Miniconda 或 Anaconda**：
   - 下载：https://docs.conda.io/en/latest/miniconda.html

2. **创建新环境**：
   ```bash
   conda create -n pose_estimation python=3.11
   conda activate pose_estimation
   ```

3. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

### 方法 3：直接安装 Python 3.11（Windows）

1. **下载 Python 3.11**：
   - 访问：https://www.python.org/downloads/
   - 下载 Python 3.11.x（推荐 3.11.9）

2. **安装时注意**：
   - ✅ 勾选 "Add Python to PATH"
   - ✅ 选择 "Install for all users"（可选）

3. **验证安装**：
   ```powershell
   python --version
   # 应该显示 Python 3.11.x
   ```

4. **创建虚拟环境**（推荐）：
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

## 安装依赖

切换 Python 版本后，运行：

```bash
pip install -r requirements.txt
```

## 验证安装

运行测试脚本：

```bash
python test_imports.py
```

应该看到所有依赖（包括 mediapipe）都成功安装。

## 推荐配置

- **Python 版本**：3.11.9 或 3.12.x
- **虚拟环境**：强烈建议使用虚拟环境隔离项目依赖

## 常见问题

### Q: 如何检查当前 Python 版本？
A: 运行 `python --version`

### Q: 如何切换虚拟环境中的 Python 版本？
A: 删除旧的虚拟环境，用新版本的 Python 重新创建：
```bash
# 删除旧环境
rm -rf venv  # Linux/Mac
rmdir /s venv  # Windows

# 用新版本创建
python3.11 -m venv venv
```

### Q: 多个 Python 版本如何共存？
A: 使用 pyenv 或 conda 可以轻松管理多个 Python 版本

