# Docker 部署指南

## 📋 本次完成的工作

### 1. Docker 镜像构建优化
- ✅ 优化 `.dockerignore`：构建上下文从 7.15GB 减少到 433KB
- ✅ 配置清华镜像源：依赖下载时间从 1小时+ 缩短到 6分19秒
- ✅ 成功构建镜像：`hunyuan-service:latest`

### 2. GPU 支持配置
- ✅ 安装 nvidia-container-toolkit (1.17.9-1)
- ✅ 配置 Docker GPU 运行时
- ✅ 验证 GPU 可用：模型使用 `cuda` 设备

### 3. 代码修复
- ✅ 修复 API bug：`output.result` → `output.text`
- ✅ 验证 API 功能正常

### 4. 性能提升
| 项目 | 优化前 | 优化后 |
|------|--------|--------|
| 构建上下文 | 7.15GB | 433KB |
| 依赖下载 | 1小时+ | 6分19秒 |
| 推理设备 | CPU（超时） | GPU (CUDA) |
| 推理速度 | >5分钟 | 预计3-10秒 |

---

## 🚀 Docker 测试指南

### 前置条件

1. **已安装 Docker**
2. **已构建镜像**：`hunyuan-service:latest`
3. **模型文件路径**：`/home/lzx/projects/hunyuan-service/models`
4. **GPU 支持**（可选）：已安装 nvidia-container-toolkit

构建镜像：docker build -t hunyuan-service:latest .

### 步骤 1：启动容器

#### GPU 模式（推荐）
```bash
# 在 WSL 终端中执行
docker run -d \
  --name hunyuan-qwen \
  --gpus all \
  -p 8000:8000 \
  -v /home/lzx/projects/hunyuan-service/models:/app/models \
  -v /home/lzx/projects/hunyuan-service/outputs:/app/outputs \
  -e MODEL_TYPE=image_to_text \
  -e QWEN_MODEL_PATH=/app/models/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 \
  hunyuan-service:latest
```

#### CPU 模式（备用）
```bash
# 移除 --gpus all 参数
docker run -d \
  --name hunyuan-qwen \
  -p 8000:8000 \
  -v /home/lzx/projects/hunyuan-service/models:/app/models \
  -v /home/lzx/projects/hunyuan-service/outputs:/app/outputs \
  -e MODEL_TYPE=image_to_text \
  -e QWEN_MODEL_PATH=/app/models/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 \
  hunyuan-service:latest
```

### 步骤 2：检查容器状态

```bash
# 查看容器状态
docker ps | grep hunyuan

# 查看日志，确认模型加载成功
docker logs hunyuan-qwen | grep "使用设备"
# GPU 模式应显示：[Qwen2_5Model] 模型加载完成，使用设备: cuda
# CPU 模式会显示：[Qwen2_5Model] 模型加载完成，使用设备: cpu

# 实时查看日志
docker logs -f hunyuan-qwen
```

### 步骤 3：测试 API

#### 方法 A：使用测试脚本（推荐）
```bash
cd /mnt/f/git/Hunyuan-service
python3 test_api.py
```

#### 方法 B：使用 curl
```bash
# 将图片转换为 base64
IMAGE_BASE64=$(base64 -w 0 /mnt/f/git/Hunyuan-service/outputs/sample.png)

# 发送推理请求
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d "{
    \"imgs\": [\"data:image/png;base64,$IMAGE_BASE64\"],
    \"prompt\": \"请详细描述这张图片\"
  }"
```

#### 方法 C：使用 API 文档页面
1. 浏览器打开：http://localhost:8000/docs
2. 展开 `POST /infer` 接口
3. 点击 "Try it out"
4. 填写请求参数
5. 点击 "Execute"

### 步骤 4：查看结果

成功的响应示例：
```json
{
  "text": "这是一张..."
}
```

---

## 🔧 容器管理命令

```bash
# 查看运行状态
docker ps

# 查看日志
docker logs hunyuan-qwen
docker logs -f hunyuan-qwen  # 实时日志

# 重启容器
docker restart hunyuan-qwen

# 停止容器
docker stop hunyuan-qwen

# 删除容器
docker rm -f hunyuan-qwen

# 进入容器
docker exec -it hunyuan-qwen bash

# 查看资源使用
docker stats hunyuan-qwen
```

---

## 🐛 故障排查

### 问题 1：容器无法启动
```bash
# 查看详细错误
docker logs hunyuan-qwen

# 常见原因：
# - 端口 8000 被占用 → 更改端口：-p 8001:8000
# - 模型路径错误 → 检查 QWEN_MODEL_PATH
# - GPU 不可用 → 使用 CPU 模式（移除 --gpus all）
```

### 问题 2：API 返回 500 错误
```bash
# 查看错误日志
docker logs hunyuan-qwen | tail -50

# 可能原因：
# - 图片格式错误
# - base64 编码问题
# - 模型加载失败
```

### 问题 3：GPU 未启用
```bash
# 检查 Docker GPU 支持
docker info | grep -i runtime

# 应该看到：nvidia

# 如果没有，需要安装 nvidia-container-toolkit：
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## 📊 性能监控

### 监控 GPU 使用
```bash
# 实时监控
watch -n 1 nvidia-smi

# 或在容器内
docker exec hunyuan-qwen nvidia-smi
```

### 查看推理时间
推理日志会显示在容器日志中，可以通过时间戳计算。

---

## 🔄 重新构建镜像

如果修改了代码，需要重新构建：

```bash
cd /mnt/f/git/Hunyuan-service
docker build -t hunyuan-service:latest .

# 停止并删除旧容器
docker rm -f hunyuan-qwen

# 启动新容器（使用上面的启动命令）
```

---

## 📝 环境变量说明

| 变量名 | 说明 | 示例值 |
|--------|------|--------|
| `MODEL_TYPE` | 模型类型 | `image_to_text` 或 `text_to_image` |
| `QWEN_MODEL_PATH` | Qwen 模型路径 | `/app/models/models--Qwen--Qwen2.5-VL-3B-Instruct/...` |
| `PYTHONUNBUFFERED` | Python 输出不缓冲 | `1` |
| `PYTHONPATH` | Python 路径 | `/app` |

---

## 💡 使用建议

1. **生产环境**：使用 GPU 模式以获得最佳性能
2. **开发测试**：CPU 模式即可，但速度较慢
3. **日志管理**：定期清理 Docker 日志以节省空间
4. **资源限制**：可以添加 `--memory` 和 `--cpus` 参数限制资源使用

---

## 🔗 相关文件

- **Dockerfile**: Docker 镜像构建配置
- **test_api.py**: API 测试脚本
- **src/api/single_task.py**: API 服务代码
- **src/models/qwen2_5.py**: Qwen 模型实现

---

**最后更新**: 2025-10-20
**Docker 镜像**: hunyuan-service:latest
**API 端口**: 8000

