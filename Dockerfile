FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN pip install --no-cache-dir uv

# 复制依赖文件（利用 Docker 层缓存）
COPY pyproject.toml uv.lock ./

# 配置 pip 使用国内镜像源（加速下载）
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装依赖（这是最耗时的步骤，放在前面可以利用缓存）
# 使用清华镜像源
RUN uv sync --frozen --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 复制源代码
COPY src ./src
COPY outputs ./outputs

EXPOSE 8000

# 环境变量
ENV MODEL_TYPE=text_to_image \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 默认启动命令
CMD ["uv", "run", "uvicorn", "src.api.single_task:app", "--host", "0.0.0.0", "--port", "8000"]