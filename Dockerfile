FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN pip install --no-cache-dir uv

# 复制依赖文件
COPY pyproject.toml uv.lock ./

# 安装依赖
RUN uv sync --frozen

# 复制源代码
COPY src ./src

EXPOSE 8000

# 环境变量
ENV MODEL_TYPE=text_to_image
ENV PYTHONUNBUFFERED=1

# 默认启动命令
CMD ["uv", "run", "uvicorn", "src.api.single_task:app", "--host", "0.0.0.0", "--port", "8000"]