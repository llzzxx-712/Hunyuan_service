#!/bin/bash
# 查找 HunyuanDiT 模型的正确路径

BASE_PATH="/home/lzx/projects/hunyuan-service/models/models--Tencent-Hunyuan--HunyuanDiT-v1.1-Diffusers-Distilled"

echo "=== 检查 HunyuanDiT 模型路径 ==="
echo ""

if [ ! -d "$BASE_PATH" ]; then
    echo "❌ 模型目录不存在: $BASE_PATH"
    exit 1
fi

echo "✓ 找到模型目录: $BASE_PATH"
echo ""

SNAPSHOTS_DIR="$BASE_PATH/snapshots"

if [ ! -d "$SNAPSHOTS_DIR" ]; then
    echo "❌ snapshots 目录不存在"
    exit 1
fi

echo "检查 snapshots 目录:"
ls -lh "$SNAPSHOTS_DIR"
echo ""

# 找到第一个 snapshot
SNAPSHOT=$(ls "$SNAPSHOTS_DIR" | head -1)

if [ -z "$SNAPSHOT" ]; then
    echo "❌ 没有找到 snapshot"
    exit 1
fi

FULL_PATH="$SNAPSHOTS_DIR/$SNAPSHOT"

echo "✓ 找到 snapshot: $SNAPSHOT"
echo ""

echo "检查必要文件:"
for file in "model_index.json" "scheduler/scheduler_config.json"; do
    if [ -f "$FULL_PATH/$file" ] || [ -L "$FULL_PATH/$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (不存在)"
    fi
done

echo ""
echo "==============================================="
echo "正确的环境变量设置命令:"
echo ""
echo "export HUNYUAN_MODEL_PATH=\"$FULL_PATH\""
echo ""
echo "==============================================="
echo ""
echo "复制上面的 export 命令并执行，然后运行:"
echo "uv run python -m src.models.hunyuan"

