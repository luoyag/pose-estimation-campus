# Git 初始化手动步骤

由于 PowerShell 可能无法识别 Git 命令（需要刷新环境变量），请按以下步骤操作：

## 方法 1：重启终端（推荐）

1. **关闭当前 PowerShell 窗口**
2. **重新打开 PowerShell**
3. **进入项目目录**：
   ```powershell
   cd E:\test_cursor
   ```
4. **验证 Git**：
   ```powershell
   git --version
   ```

## 方法 2：使用 Git Bash

如果安装了 Git，通常会有 Git Bash：

1. **右键点击项目文件夹**
2. **选择 "Git Bash Here"**
3. **执行以下命令**：

```bash
# 初始化仓库
git init

# 配置用户信息（如果还没配置）
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"

# 添加文件
git add .

# 查看状态
git status

# 创建初始提交
git commit -m "初始提交：基于 MediaPipe Pose 的校园场景人体姿态估计与行为识别项目"
```

## 方法 3：刷新环境变量

在 PowerShell 中执行：

```powershell
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
git --version
```

## 方法 4：使用完整路径

找到 Git 安装路径（通常在 `C:\Program Files\Git\cmd\git.exe`），使用完整路径：

```powershell
& "C:\Program Files\Git\cmd\git.exe" --version
& "C:\Program Files\Git\cmd\git.exe" init
```

## 推荐操作流程

1. **重启 PowerShell 终端**
2. **进入项目目录**：`cd E:\test_cursor`
3. **验证 Git**：`git --version`
4. **配置用户信息**（首次使用）：
   ```powershell
   git config --global user.name "你的名字"
   git config --global user.email "你的邮箱"
   ```
5. **初始化仓库**：
   ```powershell
   git init
   git add .
   git commit -m "初始提交：基于 MediaPipe Pose 的校园场景人体姿态估计与行为识别项目"
   ```

## 检查 Git 安装位置

如果找不到 Git，检查是否安装在以下位置：

- `C:\Program Files\Git\cmd\git.exe`
- `C:\Program Files (x86)\Git\cmd\git.exe`
- `C:\Users\你的用户名\AppData\Local\Programs\Git\cmd\git.exe`

