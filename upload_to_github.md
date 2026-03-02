# 上传到 GitHub 仓库指南

## ⚠️ 重要提示

GitHub 已经**不再支持使用密码进行 HTTPS 认证**。你需要使用以下方式之一：

### 方式 1：使用 Personal Access Token (PAT) - 推荐

1. **创建 Personal Access Token**：
   - 访问：https://github.com/settings/tokens
   - 点击 "Generate new token" -> "Generate new token (classic)"
   - 设置名称：`Polar-VQA-Upload`
   - 选择过期时间（建议 90 天或自定义）
   - 勾选权限：`repo`（完整仓库权限）
   - 点击 "Generate token"
   - **重要**：复制生成的 token（只显示一次！）

2. **使用 token 上传**（见下方命令）

### 方式 2：使用 SSH Key

如果你已经配置了 SSH key，可以使用 SSH URL：
```bash
git remote add origin git@github.com:lyl-888/Polar-VQA.git
```

## 上传步骤

### 步骤 1：初始化 Git 仓库

```bash
cd C:\Users\LY\Desktop\train
git init
```

### 步骤 2：配置 Git 用户信息（如果还没配置）

```bash
git config --global user.name "lyl-888"
git config --global user.email "liyuliang040612@163.com"
```

### 步骤 3：添加所有文件

```bash
git add .
```

### 步骤 4：提交文件

```bash
git commit -m "Initial commit: Polar-VQA training code and data"
```

### 步骤 5：添加远程仓库

```bash
git remote add origin https://github.com/lyl-888/Polar-VQA.git
```

### 步骤 6：推送到 GitHub

**使用 Personal Access Token**：
```bash
git push -u origin main
```
当提示输入用户名时：输入 `lyl-888`
当提示输入密码时：**输入你的 Personal Access Token**（不是密码！）

**或者使用 token 直接推送**：
```bash
git push https://<YOUR_TOKEN>@github.com/lyl-888/Polar-VQA.git main
```
（将 `<YOUR_TOKEN>` 替换为你的实际 token）

## 注意事项

1. **大文件限制**：GitHub 单个文件限制 100MB，仓库总大小建议 < 1GB
2. **如果文件太大**：需要使用 Git LFS（Large File Storage）
3. **敏感信息**：确保 `.gitignore` 排除了包含密码、API key 等敏感信息的文件

## 检查文件大小

在推送前，可以检查是否有大文件：

```bash
# Windows PowerShell
Get-ChildItem -Recurse -File | Where-Object {$_.Length -gt 100MB} | Select-Object FullName, @{Name="Size(MB)";Expression={[math]::Round($_.Length/1MB,2)}}
```

如果有超过 100MB 的文件，考虑：
- 使用 Git LFS
- 或从仓库中排除（添加到 .gitignore）
