# Git 仓库初始化脚本

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Git 仓库初始化" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查 Git 是否安装
try {
    $gitVersion = git --version
    Write-Host "[OK] Git 已安装: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "[错误] Git 未安装！" -ForegroundColor Red
    Write-Host "请先安装 Git: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# 检查是否已经是 Git 仓库
if (Test-Path .git) {
    Write-Host "[警告] 当前目录已经是 Git 仓库" -ForegroundColor Yellow
    $continue = Read-Host "是否继续？(y/n)"
    if ($continue -ne "y" -and $continue -ne "Y") {
        exit 0
    }
} else {
    # 初始化 Git 仓库
    Write-Host "正在初始化 Git 仓库..." -ForegroundColor Cyan
    git init
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Git 仓库初始化成功" -ForegroundColor Green
    } else {
        Write-Host "[错误] Git 仓库初始化失败" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""

# 检查 .gitignore 是否存在
if (Test-Path .gitignore) {
    Write-Host "[OK] .gitignore 文件已存在" -ForegroundColor Green
} else {
    Write-Host "[警告] .gitignore 文件不存在" -ForegroundColor Yellow
}

Write-Host ""

# 添加文件
Write-Host "正在添加文件到暂存区..." -ForegroundColor Cyan
git add .
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] 文件添加成功" -ForegroundColor Green
} else {
    Write-Host "[错误] 文件添加失败" -ForegroundColor Red
    exit 1
}

Write-Host ""

# 显示将要提交的文件
Write-Host "将要提交的文件：" -ForegroundColor Cyan
git status --short

Write-Host ""

# 创建初始提交
$commitMessage = "初始提交：基于 OpenPose 的校园场景人体姿态估计与行为识别项目"
Write-Host "正在创建初始提交..." -ForegroundColor Cyan
Write-Host "提交信息: $commitMessage" -ForegroundColor Gray
git commit -m $commitMessage

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "[OK] 初始提交创建成功！" -ForegroundColor Green
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Git 仓库设置完成！" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "下一步操作：" -ForegroundColor Yellow
    Write-Host "1. 在 GitHub/Gitee 创建远程仓库" -ForegroundColor White
    Write-Host "2. 连接远程仓库：" -ForegroundColor White
    Write-Host "   git remote add origin <仓库地址>" -ForegroundColor Gray
    Write-Host "3. 推送代码：" -ForegroundColor White
    Write-Host "   git push -u origin main" -ForegroundColor Gray
    Write-Host ""
} else {
    Write-Host "[错误] 提交失败" -ForegroundColor Red
    Write-Host "提示：如果是首次使用 Git，请先配置用户信息：" -ForegroundColor Yellow
    Write-Host "  git config --global user.name `"你的名字`"" -ForegroundColor Gray
    Write-Host "  git config --global user.email `"你的邮箱`"" -ForegroundColor Gray
    exit 1
}

