# Polar-VQA 上传到 GitHub 脚本
# 使用方法：在 PowerShell 中运行：.\upload_to_github.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Polar-VQA 上传到 GitHub" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查是否在正确的目录
$currentDir = Get-Location
if (-not (Test-Path "generate_stage3_qa_llava.py")) {
    Write-Host "错误: 请在 train 目录下运行此脚本" -ForegroundColor Red
    exit 1
}

# 步骤 1: 初始化 Git 仓库
Write-Host "[1/6] 初始化 Git 仓库..." -ForegroundColor Yellow
if (Test-Path ".git") {
    Write-Host "  Git 仓库已存在，跳过初始化" -ForegroundColor Gray
} else {
    git init
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  错误: Git 初始化失败" -ForegroundColor Red
        exit 1
    }
    Write-Host "  ✓ Git 仓库初始化成功" -ForegroundColor Green
}

# 步骤 2: 配置 Git 用户信息
Write-Host "[2/6] 配置 Git 用户信息..." -ForegroundColor Yellow
$gitUser = git config --global user.name
$gitEmail = git config --global user.email

if (-not $gitUser) {
    git config --global user.name "lyl-888"
    Write-Host "  ✓ 设置用户名: lyl-888" -ForegroundColor Green
} else {
    Write-Host "  用户名已配置: $gitUser" -ForegroundColor Gray
}

if (-not $gitEmail) {
    git config --global user.email "liyuliang040612@163.com"
    Write-Host "  ✓ 设置邮箱: liyuliang040612@163.com" -ForegroundColor Green
} else {
    Write-Host "  邮箱已配置: $gitEmail" -ForegroundColor Gray
}

# 步骤 3: 检查大文件
Write-Host "[3/6] 检查大文件（>100MB）..." -ForegroundColor Yellow
$largeFiles = Get-ChildItem -Recurse -File | Where-Object {$_.Length -gt 100MB}
if ($largeFiles) {
    Write-Host "  警告: 发现以下大文件（可能无法上传到 GitHub）:" -ForegroundColor Yellow
    foreach ($file in $largeFiles) {
        $sizeMB = [math]::Round($file.Length / 1MB, 2)
        Write-Host "    - $($file.FullName) ($sizeMB MB)" -ForegroundColor Yellow
    }
    Write-Host "  建议: 将这些文件添加到 .gitignore 或使用 Git LFS" -ForegroundColor Yellow
    $continue = Read-Host "  是否继续上传？(y/n)"
    if ($continue -ne "y") {
        Write-Host "  已取消上传" -ForegroundColor Red
        exit 0
    }
} else {
    Write-Host "  ✓ 未发现超大文件" -ForegroundColor Green
}

# 步骤 4: 添加文件
Write-Host "[4/6] 添加文件到 Git..." -ForegroundColor Yellow
git add .
if ($LASTEXITCODE -ne 0) {
    Write-Host "  错误: 添加文件失败" -ForegroundColor Red
    exit 1
}
Write-Host "  ✓ 文件添加成功" -ForegroundColor Green

# 步骤 5: 提交
Write-Host "[5/6] 提交更改..." -ForegroundColor Yellow
$commitMessage = "Initial commit: Polar-VQA training code and data"
git commit -m $commitMessage
if ($LASTEXITCODE -ne 0) {
    Write-Host "  错误: 提交失败（可能没有更改）" -ForegroundColor Red
    Write-Host "  提示: 如果这是第一次提交，请检查是否有文件被 .gitignore 排除" -ForegroundColor Yellow
    exit 1
}
Write-Host "  ✓ 提交成功" -ForegroundColor Green

# 步骤 6: 添加远程仓库并推送
Write-Host "[6/6] 配置远程仓库..." -ForegroundColor Yellow

# 检查是否已有远程仓库
$remoteUrl = git remote get-url origin 2>$null
if ($remoteUrl) {
    Write-Host "  远程仓库已存在: $remoteUrl" -ForegroundColor Gray
    $changeRemote = Read-Host "  是否更改远程仓库 URL？(y/n)"
    if ($changeRemote -eq "y") {
        git remote remove origin
        git remote add origin https://github.com/lyl-888/Polar-VQA.git
        Write-Host "  ✓ 远程仓库已更新" -ForegroundColor Green
    }
} else {
    git remote add origin https://github.com/lyl-888/Polar-VQA.git
    Write-Host "  ✓ 远程仓库已添加" -ForegroundColor Green
}

# 检查分支名称
$currentBranch = git branch --show-current
if (-not $currentBranch) {
    git branch -M main
    $currentBranch = "main"
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "准备推送到 GitHub" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "⚠️  重要提示:" -ForegroundColor Yellow
Write-Host "GitHub 不再支持密码认证，需要使用 Personal Access Token (PAT)" -ForegroundColor Yellow
Write-Host ""
Write-Host "如果没有 PAT，请先创建:" -ForegroundColor Yellow
Write-Host "  1. 访问: https://github.com/settings/tokens" -ForegroundColor Cyan
Write-Host "  2. 点击 'Generate new token' -> 'Generate new token (classic)'" -ForegroundColor Cyan
Write-Host "  3. 勾选 'repo' 权限" -ForegroundColor Cyan
Write-Host "  4. 生成并复制 token" -ForegroundColor Cyan
Write-Host ""

$useToken = Read-Host "是否使用 Personal Access Token 推送？(y/n)"
if ($useToken -eq "y") {
    $token = Read-Host "请输入你的 Personal Access Token" -AsSecureString
    $tokenPlain = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($token))
    
    # 使用 token 推送
    $remoteUrlWithToken = "https://$tokenPlain@github.com/lyl-888/Polar-VQA.git"
    Write-Host ""
    Write-Host "正在推送到 GitHub..." -ForegroundColor Yellow
    git push -u $remoteUrlWithToken $currentBranch
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "✓ 上传成功！" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "仓库地址: https://github.com/lyl-888/Polar-VQA" -ForegroundColor Cyan
    } else {
        Write-Host ""
        Write-Host "推送失败，请检查:" -ForegroundColor Red
        Write-Host "  1. Token 是否正确" -ForegroundColor Yellow
        Write-Host "  2. 是否有推送权限" -ForegroundColor Yellow
        Write-Host "  3. 网络连接是否正常" -ForegroundColor Yellow
    }
} else {
    Write-Host ""
    Write-Host "请手动执行以下命令推送:" -ForegroundColor Yellow
    Write-Host "  git push -u origin $currentBranch" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "当提示输入密码时，请输入你的 Personal Access Token" -ForegroundColor Yellow
}
