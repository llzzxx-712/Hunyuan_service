import sys
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
    profiler_output: str = "trace_baseline.json",
    max_new_tokens: int = 100,
):
    """测试模型性能，多次运行取平均"""
    times = []
    for i in range(num_runs):
        start = time.time()
        batch_outputs = model.batch_infer(
            test_inputs,
            max_new_tokens=max_new_tokens,
            enable_profiler=False,
            profiler_output=profiler_output if i == num_runs - 1 else None,
        )
        end = time.time()
        elapsed = end - start
        times.append(elapsed)
        print(f"  第 {i + 1} 次运行: {elapsed:.3f}秒")

        if i == num_runs - 1:
            for i, output in enumerate(batch_outputs):
                print(f"  [{i + 1}] {output.text}")

    avg_time = sum(times) / len(times)
    return avg_time, times


def clean_cuda_state():
    """彻底清理 CUDA 状态"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        # 强制垃圾回收
        import gc

        gc.collect()

        # 尝试重置 CUDA 上下文（实验性）
        torch.cuda._lazy_init()


def main():
    # 准备测试数据
    test_inputs = [
        ImageToTextInput(imgs=["outputs/image_2.png"], prompt="识别图中的文字"),
        ImageToTextInput(imgs=["outputs/sample_large.png"], prompt="描述这张图片"),
    ]

    # test_inputs1 = [
    #     ImageToTextInput(imgs=["outputs/image_1.png"], prompt="识别图中的文字"),
    #     ImageToTextInput(imgs=["outputs/hunyuan_output_1_small.png"], prompt="描述这张图片"),
    # ]

    # test_inputs2 = [
    #     #ImageToTextInput(imgs=["outputs/sample_small.png"], prompt="识别图中的文字"),
    #     ImageToTextInput(imgs=["outputs/sample_large.png"], prompt="描述这张图片"),
    # ]

    print("=" * 60)
    print("Torch Compile 性能对比测试")
    print("=" * 60)

    # print("\n[测试 2] 编译版本 (mode=default)")
    # print("-" * 60)
    # model_compiled = Qwen2_5Model(compile_model=True, compile_mode="default", do_warmup=True)
    # print("预热完成，开始测试...")
    # avg_compiled, times_compiled = benchmark_model(
    #     model_compiled, test_inputs, num_runs=3, profiler_output="trace_compiled.json"
    # )
    # print(f"平均耗时: {avg_compiled:.3f}秒")

    # del model_compiled
    # clean_cuda_state()

    # 测试 1: 未编译版本
    print("\n[测试 1] 未编译版本 (baseline)")
    print("-" * 60)

    load_start = time.time()
    model_baseline = Qwen2_5Model(compile_model=False, do_warmup=False)
    load_end = time.time()
    print(f"[Time] 模型加载耗时: {load_end - load_start:.3f}秒")

    avg_baseline, times_baseline = benchmark_model(
        model_baseline,
        test_inputs,
        num_runs=3,
        profiler_output="trace_baseline.json",
        max_new_tokens=100,
    )
    print(f"平均耗时: {avg_baseline:.3f}秒")

    # 清理显存
    del model_baseline
    clean_cuda_state()

    # 检查 transformers 相关的模块
    print("\n检查 transformers 模块缓存：")
    for key in sys.modules.keys():
        if "qwen" in key.lower():
            print(f"  {key}")

    # 检查 SDPA 相关的全局状态
    import torch.nn.functional as F

    if hasattr(F, "_scaled_dot_product_attention"):
        print("SDPA 已初始化")
    else:
        print("SDPA 未初始化")

    # 测试 2: 编译版本 (default mode)
    print("\n[测试 2] 编译版本 (mode=default)")
    print("-" * 60)

    load_start = time.time()
    model_compiled = Qwen2_5Model(compile_model=True, compile_mode="default", do_warmup=True)
    load_end = time.time()
    print(f"[Time] 模型加载耗时: {load_end - load_start:.3f}秒")

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
