import time

import torch

# torch.backends.cudnn.benchmark = False
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False
# import logging
# logging.basicConfig(level=logging.DEBUG)
from src.models.qwen2_5 import ImageToTextInput, Qwen2_5Model


def benchmark_model(
    model: Qwen2_5Model,
    test_inputs: list,
    num_runs: int = 3,
    enable_profiling: bool = False,
    profiler_output: str = "trace.json",
    max_new_tokens: int = 100,
):
    """测试模型性能，多次运行取平均"""
    times = []
    for i in range(num_runs):
        # 只在最后一次运行时启用 profiler
        enable_prof = enable_profiling and (i == num_runs - 1)

        start = time.time()
        batch_outputs = model.batch_infer(
            test_inputs,
            max_new_tokens=max_new_tokens,
            enable_profiler=enable_prof,
            profiler_output=profiler_output,
        )
        end = time.time()
        elapsed = end - start
        times.append(elapsed)

        if enable_prof:
            print(f"  第 {i + 1} 次运行（with profiler）: {elapsed:.3f}秒")
            print(f"  ✓ Profiler trace 已保存到: {profiler_output}")
        else:
            print(f"  第 {i + 1} 次运行: {elapsed:.3f}秒")

        # 显示最后一次的输出结果
        if i == num_runs - 1:
            for idx, output in enumerate(batch_outputs):
                text_preview = output.text[:80] + "..." if len(output.text) > 80 else output.text
                print(f"  [{idx + 1}] {text_preview}")

    avg_time = sum(times) / len(times)
    return avg_time, times


def clean_cuda_state():
    """彻底清理 CUDA 状态"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        import gc

        gc.collect()
        torch.cuda._lazy_init()


def main():
    # 准备测试数据
    # test_inputs = [
    #     ImageToTextInput(imgs=["outputs/image_2.png"], prompt="识别图中的文字"),
    #     ImageToTextInput(imgs=["outputs/sample_large.png"], prompt="描述这张图片"),
    # ]

    test_inputs_large = [
        ImageToTextInput(imgs=["outputs/classroom_small.png"], prompt="描述这张图片"),
        ImageToTextInput(imgs=["outputs/hunyuan_output_1.png"], prompt="描述这张图片"),
    ]

    # test_inputs_small = [
    #     ImageToTextInput(imgs=["outputs/sample.png"], prompt="描述这张图片"),
    #     ImageToTextInput(imgs=["outputs/To_warm.png"], prompt="描述这张图片"),
    # ]

    tem_model = Qwen2_5Model(do_warmup=False, compile_model=False)
    _ = tem_model.batch_infer(test_inputs_large, max_new_tokens=100, enable_profiler=False)
    del tem_model
    clean_cuda_state()

    print("=" * 60)
    print("Torch Compile 性能对比测试")
    print("=" * 60)

    # 测试 1: 未编译版本
    print("\n[测试 1] 未编译版本 (baseline)")
    print("-" * 60)

    load_start = time.time()
    model_baseline = Qwen2_5Model(compile_model=False, do_warmup=False)
    load_end = time.time()
    print(f"[Time] 模型加载耗时: {load_end - load_start:.3f}秒")

    avg_baseline, times_baseline = benchmark_model(
        model_baseline,
        test_inputs_large,
        num_runs=4,
        enable_profiling=False,
        profiler_output="trace_baseline_warm.json",
        max_new_tokens=100,
    )
    print(f"平均耗时: {avg_baseline:.3f}秒")

    # 清理显存
    del model_baseline
    clean_cuda_state()

    # 测试 2: 编译版本 (default mode)
    print("\n[测试 2] 编译版本 (mode=default)")
    print("-" * 60)

    load_start = time.time()
    model_compiled = Qwen2_5Model(compile_model=True, compile_mode="default", do_warmup=False)
    load_end = time.time()
    print(f"[Time] 模型加载耗时: {load_end - load_start:.3f}秒")

    print("开始测试...")
    avg_compiled, times_compiled = benchmark_model(
        model_compiled,
        test_inputs_large,
        num_runs=4,
        enable_profiling=False,
        profiler_output="trace_compiled_warm.json",
        max_new_tokens=100,
    )
    print(f"平均耗时: {avg_compiled:.3f}秒")

    # 结果对比
    print("\n" + "=" * 60)
    print("性能对比结果")
    print("=" * 60)
    print(f"未编译版本: {avg_baseline:.3f}秒")
    print(f"编译版本:   {avg_compiled:.3f}秒")
    speedup = avg_baseline / avg_compiled
    print(f"加速比:     {speedup:.2f}x")
    improvement = (avg_baseline - avg_compiled) / avg_baseline * 100
    print(f"性能提升:   {improvement:.1f}%")


if __name__ == "__main__":
    main()
