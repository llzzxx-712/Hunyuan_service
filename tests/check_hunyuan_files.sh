#!/bin/bash

MODEL_PATH="/home/lzx/projects/hunyuan-service/models/models--Tencent-Hunyuan--HunyuanDiT-v1.1-Diffusers-Distilled/snapshots/527cf2ecce7c04021975938f8b0e44e35d2b1ed9"

echo "=== 检查 HunyuanDiT 模型文件完整性 ==="
echo ""
echo "模型路径: $MODEL_PATH"
echo ""

# 检查必要的配置文件
echo "检查配置文件:"
for file in "model_index.json" "scheduler/scheduler_config.json" "text_encoder/config.json" "tokenizer/tokenizer_config.json" "transformer/config.json" "vae/config.json"; do
    FULL_FILE="$MODEL_PATH/$file"
    if [ -f "$FULL_FILE" ]; then
        SIZE=$(stat -c%s "$FULL_FILE" 2>/dev/null || stat -f%z "$FULL_FILE" 2>/dev/null)
        if [ "$SIZE" -eq 0 ]; then
            echo "  ✗ $file (0 bytes - 符号链接可能失效)"
        else
            echo "  ✓ $file ($SIZE bytes)"
        fi
    elif [ -L "$FULL_FILE" ]; then
        TARGET=$(readlink "$FULL_FILE")
        echo "  → $file -> $TARGET"
        if [ ! -f "$MODEL_PATH/$TARGET" ] && [ ! -f "$TARGET" ]; then
            echo "    ✗ 符号链接目标不存在!"
        fi
    else
        echo "  ✗ $file (不存在)"
    fi
done

echo ""
echo "检查目录结构:"
ls -la "$MODEL_PATH" | head -20

