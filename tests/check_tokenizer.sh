#!/bin/bash

MODEL_PATH="/home/lzx/projects/hunyuan-service/models/models--Tencent-Hunyuan--HunyuanDiT-v1.1-Diffusers-Distilled/snapshots/527cf2ecce7c04021975938f8b0e44e35d2b1ed9"

echo "=== 检查 tokenizer 配置 ==="
echo ""

# 检查 tokenizer 配置
if [ -f "$MODEL_PATH/tokenizer/tokenizer_config.json" ] || [ -L "$MODEL_PATH/tokenizer/tokenizer_config.json" ]; then
    echo "tokenizer/tokenizer_config.json 内容:"
    cat "$MODEL_PATH/tokenizer/tokenizer_config.json" 2>/dev/null || echo "无法读取（可能是符号链接）"
    echo ""
fi

# 检查实际的 blob 文件
TOKENIZER_CONFIG_LINK=$(readlink "$MODEL_PATH/tokenizer/tokenizer_config.json" 2>/dev/null)
if [ ! -z "$TOKENIZER_CONFIG_LINK" ]; then
    echo "符号链接指向: $TOKENIZER_CONFIG_LINK"
    BLOB_PATH="$MODEL_PATH/tokenizer/$TOKENIZER_CONFIG_LINK"
    if [ -f "$BLOB_PATH" ]; then
        echo "实际文件内容:"
        cat "$BLOB_PATH" | head -20
    else
        echo "blob 文件不存在: $BLOB_PATH"
        
        # 尝试在 blobs 目录找
        BASE_DIR=$(dirname $(dirname "$MODEL_PATH"))
        BLOB_FILE="$BASE_DIR/blobs/$(basename $TOKENIZER_CONFIG_LINK)"
        if [ -f "$BLOB_FILE" ]; then
            echo "在 blobs 目录找到:"
            cat "$BLOB_FILE" | head -20
        fi
    fi
fi

echo ""
echo "=== 检查是否有必要的 tokenizer 文件 ==="
cd "$MODEL_PATH/tokenizer"
ls -lh

