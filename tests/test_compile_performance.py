import time

from src.models.qwen2_5 import ImageToTextInput, Qwen2_5Model


def benchmark_model(
    model: Qwen2_5Model,
    test_inputs: list,
    num_runs: int = 2,
    profiler_output: str = "trace_baseline.json",
):
    """测试模型性能，多次运行取平均"""
    times = []
    for i in range(num_runs):
        start = time.time()
        model.batch_infer(
            test_inputs, max_new_tokens=100, enable_profiler=False, profiler_output=profiler_output
        )
        end = time.time()
        elapsed = end - start
        times.append(elapsed)
        print(f"  第 {i + 1} 次运行: {elapsed:.3f}秒")

    avg_time = sum(times) / len(times)
    return avg_time, times


def main():
    # 准备测试数据
    test_inputs = [
        ImageToTextInput(imgs=["outputs/image_2.png"], prompt="识别图中的文字"),
        ImageToTextInput(imgs=["outputs/sample_large.png"], prompt="描述这张图片"),
    ]

    print("=" * 60)
    print("Torch Compile 性能对比测试")
    print("=" * 60)

    # 测试 1: 未编译版本
    print("\n[测试 1] 未编译版本 (baseline)")
    print("-" * 60)
    model_baseline = Qwen2_5Model(compile_model=False, do_warmup=False)
    avg_baseline, times_baseline = benchmark_model(
        model_baseline, test_inputs, num_runs=3, profiler_output="trace_baseline.json"
    )
    print(f"平均耗时: {avg_baseline:.3f}秒")

    # 清理显存
    del model_baseline
    import torch

    torch.cuda.empty_cache()

    # 测试 2: 编译版本 (default mode)
    print("\n[测试 2] 编译版本 (mode=default)")
    print("-" * 60)
    model_compiled = Qwen2_5Model(compile_model=True, compile_mode="default", do_warmup=True)
    print("预热完成，开始测试...")
    avg_compiled, times_compiled = benchmark_model(
        model_compiled, test_inputs, num_runs=3, profiler_output="trace_compiled.json"
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
