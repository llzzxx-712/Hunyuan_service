"""简单的Qwen模型测试脚本，带调试输出"""

import sys

print("开始测试...", flush=True)

try:
    print("1. 导入torch...", flush=True)
    import torch

    print(f"   - torch版本: {torch.__version__}", flush=True)
    print(f"   - CUDA可用: {torch.cuda.is_available()}", flush=True)

    print("2. 导入transformers...", flush=True)

    print("3. 导入qwen_vl_utils...", flush=True)

    print("4. 导入自定义模块...", flush=True)
    from src.models.qwen2_5 import ImageToTextInput, Qwen2_5Model

    print("5. 检查测试图片...", flush=True)
    import os

    test_image = "outputs/image_0.png"
    if os.path.exists(test_image):
        print(f"   - 图片存在: {test_image}", flush=True)
    else:
        print(f"   - 警告: 图片不存在: {test_image}", flush=True)
        sys.exit(1)

    print("6. 开始加载模型（这可能需要几分钟）...", flush=True)
    model = Qwen2_5Model()
    print("   - 模型加载完成！", flush=True)
    print(f"   - 使用设备: {model.device}", flush=True)

    print("7. 准备输入数据...", flush=True)
    test_input = ImageToTextInput(imgs=[test_image], prompt="Describe this image")
    print("   - 输入准备完成", flush=True)

    print("8. 开始推理...", flush=True)
    output = model.infer(test_input)
    print("   - 推理完成！", flush=True)

    print("\n" + "=" * 50)
    print("推理结果:")
    print(output.result)
    print("=" * 50)

except Exception as e:
    print(f"\n❌ 错误: {type(e).__name__}: {e}", flush=True)
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n✅ 测试成功完成！")
