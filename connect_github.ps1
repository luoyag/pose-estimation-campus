# 连接 GitHub 仓库脚本

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "连接 GitHub 仓库" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查是否已有远程仓库
$existingRemote = git remote -v
if ($existingRemote) {
    Write-Host "[警告] 已存在远程仓库：" -ForegroundColor Yellow
    Write-Host $existingRemote -ForegroundColor Gray
    $continue = Read-Host "是否要替换？(y/n)"
    if ($continue -ne "y" -and $continue -ne "Y") {
        exit 0
    }
    git remote remove origin
}

Write-Host ""
Write-Host "请输入你的 GitHub 仓库地址：" -ForegroundColor Yellow
Write-Host "示例：https://github.com/你的用户名/仓库名.git" -ForegroundColor Gray
Write-Host ""

$repoUrl = Read-Host "GitHub 仓库地址"

if ([string]::IsNullOrWhiteSpace($repoUrl)) {
    Write-Host "[错误] 仓库地址不能为空" -ForegroundColor Red
    exit 1
}

# 添加远程仓库
Write-Host ""
Write-Host "正在添加远程仓库..." -ForegroundColor Cyan
git remote add origin $repoUrl

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] 远程仓库添加成功" -ForegroundColor Green
    
    # 显示远程仓库信息
    Write-Host ""
    Write-Host "远程仓库配置：" -ForegroundColor Cyan
    git remote -v
    
    Write-Host ""
    Write-Host "正在推送代码到 GitHub..." -ForegroundColor Cyan
    Write-Host "（如果是首次推送，可能需要输入 GitHub 用户名和密码/Token）" -ForegroundColor Yellow
    Write-Host ""
    
    # 推送代码
    git push -u origin master
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "[OK] 代码推送成功！" -ForegroundColor Green
        Write-Host ""
        Write-Host "你的代码已上传到 GitHub：" -ForegroundColor Cyan
        Write-Host $repoUrl -ForegroundColor White
        Write-Host ""
    } else {
        Write-Host ""
        Write-Host "[错误] 推送失败" -ForegroundColor Red
        Write-Host ""
        Write-Host "可能的原因：" -ForegroundColor Yellow
        Write-Host "1. 需要 GitHub 认证（用户名和 Personal Access Token）" -ForegroundColor White
        Write-Host "2. 分支名称不匹配（GitHub 可能使用 main 分支）" -ForegroundColor White
        Write-Host ""
        Write-Host "解决方案：" -ForegroundColor Yellow
        Write-Host "1. 创建 Personal Access Token：" -ForegroundColor White
        Write-Host "   https://github.com/settings/tokens" -ForegroundColor Gray
        Write-Host "2. 如果分支是 main，运行：" -ForegroundColor White
        Write-Host "   git push -u origin master:main" -ForegroundColor Gray
    }
} else {
    Write-Host "[错误] 添加远程仓库失败" -ForegroundColor Red
    exit 1
}

