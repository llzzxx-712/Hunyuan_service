"""
测试 SDPA 初始化对性能的影响
对比在 SDPA 已预热状态下，编译版本 vs 未编译版本的性能
"""

import time

import torch

from src.models.qwen2_5 import ImageToTextInput, Qwen2_5Model


def clean_cuda_state():
    """彻底清理 CUDA 状态"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        import gc

        gc.collect()


def main():
    test_inputs = [
        ImageToTextInput(imgs=["outputs/image_2.png"], prompt="识别图中的文字"),
        ImageToTextInput(imgs=["outputs/sample_large.png"], prompt="描述这张图片"),
    ]

    print("=" * 60)
    print("SDPA 初始化影响测试")
    print("=" * 60)

    # ============ 阶段1：SDPA 预热 ============
    print("\n[阶段 1] 使用未编译版本预热 SDPA")
    print("-" * 60)
    model_warmup = Qwen2_5Model(compile_model=False, do_warmup=False)

    start = time.time()
    _ = model_warmup.batch_infer(test_inputs, max_new_tokens=100)
    warmup_time = time.time() - start
    print(f"SDPA 预热耗时: {warmup_time:.3f}秒")

    del model_warmup
    clean_cuda_state()
    time.sleep(1)

    # 验证 SDPA 已初始化
    import torch.nn.functional as F

    sdpa_initialized = (
        hasattr(F.scaled_dot_product_attention, "__wrapped__")
        if hasattr(F, "scaled_dot_product_attention")
        else False
    )
    print(f"SDPA 状态: {'已初始化 ✓' if sdpa_initialized else '未初始化'}")

    # ============ 阶段2：对比未编译版本（热启动）============
    print("\n[阶段 2] 未编译版本（SDPA 已预热）")
    print("-" * 60)
    model_baseline = Qwen2_5Model(compile_model=False, do_warmup=False)

    times_baseline = []
    for i in range(3):
        start = time.time()
        _ = model_baseline.batch_infer(test_inputs, max_new_tokens=100)
        elapsed = time.time() - start
        times_baseline.append(elapsed)
        print(f"  第 {i + 1} 次: {elapsed:.3f}秒")

    avg_baseline = sum(times_baseline) / len(times_baseline)
    print(f"平均耗时: {avg_baseline:.3f}秒")

    del model_baseline
    clean_cuda_state()
    time.sleep(1)

    # ============ 阶段3：对比编译版本（热启动）============
    print("\n[阶段 3] 编译版本（SDPA 已预热）")
    print("-" * 60)
    # 注意：do_warmup=False，因为 SDPA 已经预热过了
    model_compiled = Qwen2_5Model(compile_model=True, compile_mode="default", do_warmup=False)

    # 第一次会有 torch.compile 的编译开销
    print("首次运行（torch.compile 编译）...")
    start = time.time()
    _ = model_compiled.batch_infer(test_inputs, max_new_tokens=100)
    first_time = time.time() - start
    print(f"  首次（含编译）: {first_time:.3f}秒")

    # 后续运行使用编译缓存
    times_compiled = []
    for i in range(3):
        start = time.time()
        _ = model_compiled.batch_infer(test_inputs, max_new_tokens=100)
        elapsed = time.time() - start
        times_compiled.append(elapsed)
        print(f"  第 {i + 1} 次: {elapsed:.3f}秒")

    avg_compiled = sum(times_compiled) / len(times_compiled)
    print(f"平均耗时: {avg_compiled:.3f}秒")

    # ============ 结果对比 ============
    print("\n" + "=" * 60)
    print("性能对比结果（SDPA 已预热状态）")
    print("=" * 60)
    print(f"未编译版本: {avg_baseline:.3f}秒")
    print(f"编译版本:   {avg_compiled:.3f}秒")
    speedup = avg_baseline / avg_compiled
    print(f"加速比:     {speedup:.2f}x")
    improvement = (avg_baseline - avg_compiled) / avg_baseline * 100
    print(f"性能提升:   {improvement:.1f}%")

    print("\n" + "=" * 60)
    print("关键发现")
    print("=" * 60)
    print(f"SDPA 预热开销:        {warmup_time:.3f}秒")
    print(f"torch.compile 编译:   {first_time - avg_compiled:.3f}秒")
    print(f"纯推理性能提升:       {avg_baseline - avg_compiled:.3f}秒 (torch.compile 带来)")
    print("总开销分解:")
    print(f"  - SDPA 初始化:      {warmup_time - 8:.3f}秒 (假设纯推理 8秒)")
    print(f"  - torch.compile:    {first_time - avg_compiled:.3f}秒")
    print(f"  - 合计:             {(warmup_time - 8) + (first_time - avg_compiled):.3f}秒")


if __name__ == "__main__":
    main()
