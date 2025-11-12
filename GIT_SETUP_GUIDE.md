# Git 仓库设置指南

## 第一步：安装 Git（如果还没有安装）

### Windows 安装 Git

1. **下载 Git for Windows**：
   - 访问：https://git-scm.com/download/win
   - 或使用包管理器：
     ```powershell
     # 使用 Chocolatey
     choco install git
     
     # 或使用 winget
     winget install Git.Git
     ```

2. **安装后验证**：
   ```powershell
   git --version
   ```

3. **配置 Git（首次使用）**：
   ```powershell
   git config --global user.name "你的名字"
   git config --global user.email "你的邮箱"
   ```

## 第二步：初始化 Git 仓库

### 1. 初始化仓库

```powershell
# 进入项目目录
cd E:\test_cursor

# 初始化 Git 仓库
git init
```

### 2. 添加文件到暂存区

```powershell
# 添加所有文件（.gitignore 会自动排除不需要的文件）
git add .

# 查看将要提交的文件
git status
```

### 3. 创建初始提交

```powershell
git commit -m "初始提交：基于 OpenPose 的校园场景人体姿态估计与行为识别项目"
```

## 第三步：连接到远程仓库（可选）

### 在 GitHub/Gitee 创建仓库

1. **GitHub**：
   - 访问 https://github.com/new
   - 创建新仓库（不要初始化 README、.gitignore 或 license）

2. **Gitee（码云）**：
   - 访问 https://gitee.com/projects/new
   - 创建新仓库

### 连接远程仓库

```powershell
# 添加远程仓库（替换为你的仓库地址）
git remote add origin https://github.com/你的用户名/你的仓库名.git

# 或使用 SSH
git remote add origin git@github.com:你的用户名/你的仓库名.git

# 查看远程仓库
git remote -v
```

### 推送到远程仓库

```powershell
# 推送代码到远程仓库
git push -u origin main

# 如果默认分支是 master，使用：
git push -u origin master
```

## 常用 Git 命令

```powershell
# 查看状态
git status

# 查看提交历史
git log

# 添加文件
git add 文件名
git add .  # 添加所有文件

# 提交更改
git commit -m "提交信息"

# 推送到远程
git push

# 拉取远程更新
git pull

# 查看分支
git branch

# 创建新分支
git branch 分支名

# 切换分支
git checkout 分支名
```

## 项目文件说明

`.gitignore` 文件已经配置好，会自动排除：
- Python 缓存文件（`__pycache__/`）
- 虚拟环境（`venv/`）
- 模型文件（`models/*.pkl` 等）
- 输出视频（`*.mp4`）
- IDE 配置文件

## 注意事项

1. **不要提交大文件**：视频文件、模型文件等已在 `.gitignore` 中排除
2. **提交前检查**：使用 `git status` 查看将要提交的文件
3. **提交信息**：使用有意义的提交信息，描述本次更改的内容

## 快速开始脚本

创建 `setup_git.ps1` 脚本可以一键完成初始化：

```powershell
# 初始化仓库
git init

# 添加所有文件
git add .

# 创建初始提交
git commit -m "初始提交：基于 OpenPose 的校园场景人体姿态估计与行为识别项目"

echo "Git 仓库初始化完成！"
echo "下一步：连接到远程仓库（可选）"
```

