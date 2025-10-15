"""检查模型文件是否完整"""

from pathlib import Path

model_base = Path("F:/git/model_data/models--Qwen--Qwen2.5-VL-3B-Instruct")

print("=== 检查模型目录结构 ===\n")

# 找到 snapshot 目录
snapshots_dir = model_base / "snapshots"
if snapshots_dir.exists():
    snapshot_dirs = list(snapshots_dir.iterdir())
    if snapshot_dirs:
        snapshot_path = snapshot_dirs[0]
        print(f"✓ 找到 snapshot: {snapshot_path.name}\n")

        # 检查必需文件
        required_files = [
            "config.json",
            "tokenizer_config.json",
        ]

        model_files = [
            "model.safetensors",
            "model-00001-of-00002.safetensors",
            "pytorch_model.bin",
        ]

        print("检查配置文件:")
        for file in required_files:
            filepath = snapshot_path / file
            exists = filepath.exists()
            size = filepath.stat().st_size if exists else 0
            status = "✓" if exists and size > 0 else "✗"
            print(f"  {status} {file}: {size} bytes")

        print("\n检查模型权重文件:")
        found_model = False
        for file in model_files:
            filepath = snapshot_path / file
            if filepath.exists():
                size = filepath.stat().st_size
                status = "✓" if size > 0 else "✗ (0 bytes - 可能是符号链接)"
                print(f"  {status} {file}: {size:,} bytes")
                if size > 0:
                    found_model = True

        if not found_model:
            print("\n⚠ 警告: 所有模型文件都是 0 字节!")
            print("这通常意味着文件是符号链接，实际文件在 blobs 目录")

            blobs_dir = model_base / "blobs"
            if blobs_dir.exists():
                print("\n检查 blobs 目录:")
                blobs = list(blobs_dir.iterdir())
                total_size = sum(f.stat().st_size for f in blobs if f.is_file())
                print(f"  文件数: {len(blobs)}")
                print(f"  总大小: {total_size / (1024**3):.2f} GB")

        print("\n推荐使用的路径:")
        print(f'  export QWEN_MODEL_PATH="{snapshot_path}"')

        # 转换为 WSL2 路径
        wsl_path = str(snapshot_path).replace("F:", "/mnt/f").replace("\\", "/")
        print("\nWSL2 中使用:")
        print(f'  export QWEN_MODEL_PATH="{wsl_path}"')
else:
    print("✗ 未找到 snapshots 目录")
