"""检查 HunyuanDiT 相关依赖"""

import sys

print("=== 检查依赖版本 ===\n")

# 检查 Python 版本
print(f"Python 版本: {sys.version}")
print("")

# 检查关键库
packages = {
    "torch": None,
    "transformers": None,
    "diffusers": None,
    "accelerate": None,
    "sentencepiece": None,
    "protobuf": None,
}

for package in packages:
    try:
        module = __import__(package)
        version = getattr(module, "__version__", "unknown")
        packages[package] = version
        print(f"✓ {package:20s} : {version}")
    except ImportError:
        print(f"✗ {package:20s} : 未安装")
        packages[package] = None

print("\n=== 依赖问题诊断 ===\n")

issues = []

# 检查缺失的库
if packages["sentencepiece"] is None:
    issues.append("❌ 缺少 sentencepiece - HunyuanDiT 的 tokenizer 需要此库")

if packages["protobuf"] is None:
    issues.append("❌ 缺少 protobuf - transformers 需要此库")

# 检查版本问题
if packages["transformers"]:
    major, minor = packages["transformers"].split(".")[:2]
    if int(major) < 4 or (int(major) == 4 and int(minor) < 36):
        issues.append(f"⚠️  transformers 版本可能过低: {packages['transformers']} (建议 >= 4.36)")

if packages["diffusers"]:
    major, minor = packages["diffusers"].split(".")[:2]
    if int(major) == 0 and int(minor) < 27:
        issues.append(f"⚠️  diffusers 版本可能过低: {packages['diffusers']} (建议 >= 0.27)")

if issues:
    print("\n".join(issues))
    print("\n建议修复命令:")
    if packages["sentencepiece"] is None:
        print("  pip install sentencepiece")
    if packages["protobuf"] is None:
        print("  pip install protobuf")
else:
    print("✓ 所有依赖都已正确安装")

print("\n=== 测试 tokenizer 加载 ===\n")

# 测试是否能加载 T5 tokenizer
try:
    from transformers import T5Tokenizer

    print("✓ 可以导入 T5Tokenizer")

    # 尝试加载（如果有本地模型）
    import os

    model_path = os.getenv("HUNYUAN_MODEL_PATH")
    if model_path:
        tokenizer_path = os.path.join(model_path, "tokenizer")
        if os.path.exists(tokenizer_path):
            print(f"  尝试加载本地 tokenizer: {tokenizer_path}")
            try:
                tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
                print("  ✓ tokenizer 加载成功")
            except Exception as e:
                print(f"  ✗ tokenizer 加载失败: {e}")
        else:
            print(f"  tokenizer 目录不存在: {tokenizer_path}")
    else:
        print("  未设置 HUNYUAN_MODEL_PATH 环境变量")

except ImportError as e:
    print(f"✗ 无法导入 T5Tokenizer: {e}")
except Exception as e:
    print(f"✗ 测试失败: {e}")
