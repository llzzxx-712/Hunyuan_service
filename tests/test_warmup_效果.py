"""
测试改进后的 warm_up 效果
验证预热后是否能直接达到 8 秒的稳定性能
"""

import time

from src.models.qwen2_5 import ImageToTextInput, Qwen2_5Model


def test_warmup_effectiveness():
    """测试 warmup 的有效性"""
    test_inputs = [
        ImageToTextInput(imgs=["outputs/image_2.png"], prompt="识别图中的文字"),
        ImageToTextInput(imgs=["outputs/sample_large.png"], prompt="描述这张图片"),
    ]

    print("=" * 60)
    print("Warm-up 效果测试")
    print("=" * 60)

    # ============ 测试编译版本 ============
    print("\n[测试] 编译版本（with 改进的 warm_up）")
    print("-" * 60)

    # 加载模型（会自动调用 warm_up，默认 warmup_runs=2）
    load_start = time.time()
    model = Qwen2_5Model(compile_model=True, compile_mode="default", do_warmup=True)
    load_time = time.time() - load_start
    print(f"\n模型加载总耗时（含预热）: {load_time:.3f}秒")

    # 测试实际推理性能
    print("\n实际推理测试：")
    times = []
    for i in range(3):
        start = time.time()
        outputs = model.batch_infer(test_inputs, max_new_tokens=100, enable_profiler=False)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  第 {i + 1} 次推理: {elapsed:.3f}秒")

    avg_time = sum(times) / len(times)

    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    print(f"平均推理时间: {avg_time:.3f}秒")

    if avg_time < 9.0:
        print("✓ 成功！预热后直接达到最佳性能（< 9秒）")
    elif avg_time < 12.0:
        print("⚠ 部分成功，性能较好但未达到最佳（9-12秒）")
    else:
        print("✗ 预热不充分，性能未达到预期（> 12秒）")

    print("\n输出示例：")
    for idx, output in enumerate(outputs):
        text_preview = output.text[:80] + "..." if len(output.text) > 80 else output.text
        print(f"  [{idx + 1}] {text_preview}")


if __name__ == "__main__":
    test_warmup_effectiveness()
